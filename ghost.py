#!/usr/bin/env python3
"""
End-to-End Big Data Bowl Heliocentricity Transformer

Pipeline:
1. Data Processing: Load and transform raw CSV data into ML-ready format
2. Caching: Save/load processed tensors to avoid reprocessing
3. Model Training: Train the Heliocentricity Transformer with CVAE
4. Evaluation: Evaluate predictions with metadata tracking
5. Heliocentricity: Calculate Heliocentricity scores with play/player cross-referencing

Key Features:
- Automatic caching: Processed data saved to dataset/processed/processed_data.pt
- Metadata tracking: Every prediction linked to game_id, play_id, and player_ids
- Pretrained weights: Model weights saved to dataset/pretrained/best_heliocentricity_model.pt
- Cross-referencing: Easy lookup of predictions by play and player
"""

# === Imports and Hyperparameters ===

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from sklearn.metrics import mean_squared_error

# --- Hyperparameters ---
# Data dimensions from Big Data Bowl
T_HIST = 25         # Number of historical frames (max in dataset)
T_PRED = 25         # Number of frames to predict (max in dataset)
N_AGENTS = 9        # Actual number of agents per frame in data
D_AGENT = 33        # Agent features: player_height, player_weight, s, a, dir, o, x_rel, y_rel + one-hot encoded position/side/role
D_GLOBAL = 18       # Global features: down, yards_to_go + one-hot encoded dropback_type, team_coverage_type

# Model architecture hyperparameters
D_MODEL = 128       # Transformer Embedding Dimension
D_LATENT = 32       # Latent variable Z dimension
N_HEADS = 8         # Transformer Heads
N_LAYERS = 3        # Transformer Encoder Layers
KL_BETA = 0.01      # KL Loss Weight (needs tuning/annealing)

# Training hyperparameters
NUM_EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4

# File paths
DATA_FOLDER_PATH = Path('dataset')
PROCESSED_DATA_PATH = DATA_FOLDER_PATH / Path('processed/processed_data.pt')
PRETRAINED_WEIGHTS_PATH = DATA_FOLDER_PATH / Path('pretrained/best_heliocentricity_model.pt')


# === Data Preprocessing Functions ===

def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a direction-invariant view of all plays.

    Returns a new DataFrame where:
    - x_rel=0 is at the line of scrimmage (offense behind at negative x_rel, defense ahead at positive x_rel)
    - All plays show offense driving toward increasing x (left to right / bottom to top)
    - 'left' plays are flipped since they drive toward decreasing x
    - Orientation and direction angles are properly adjusted

    Original DataFrame is not modified.
    """
    df_rel = df.copy()
    is_left = df_rel['play_direction'] == 'left'

    df_rel.loc[is_left, 'x'] = 120 - df_rel.loc[is_left, 'x']
    if 'ball_land_x' in df_rel.columns:
        df_rel.loc[is_left, 'ball_land_x'] = 120 - df_rel.loc[is_left, 'ball_land_x']

    df_rel.loc[is_left, 'y'] = 53.3 - df_rel.loc[is_left, 'y']
    if 'ball_land_y' in df_rel.columns:
        df_rel.loc[is_left, 'ball_land_y'] = 53.3 - df_rel.loc[is_left, 'ball_land_y']

    df_rel.loc[is_left, 'o'] = df_rel.loc[is_left, 'o'] - 180
    df_rel.loc[is_left, 'dir'] = df_rel.loc[is_left, 'dir'] - 180

    df_rel.loc[is_left, 'o'] = df_rel.loc[is_left, 'o'] % 360
    df_rel.loc[is_left, 'dir'] = df_rel.loc[is_left, 'dir'] % 360

    df_rel.loc[is_left, 'absolute_yardline_number'] = 120 - df_rel.loc[is_left, 'absolute_yardline_number']

    df_rel['x_rel'] = df_rel['x'] - df_rel['absolute_yardline_number']
    if 'ball_land_x' in df_rel.columns:
        df_rel['ball_land_x_rel'] = df_rel['ball_land_x'] - df_rel['absolute_yardline_number']

    df_rel['y_rel'] = df_rel['y'] - 26.65
    if 'ball_land_y' in df_rel.columns:
        df_rel['ball_land_y_rel'] = df_rel['ball_land_y'] - 26.65

    if 'ball_land_x' in df_rel.columns and 'ball_land_y' in df_rel.columns:
        df_rel['dist_to_ball'] = np.sqrt(
            (df_rel['x'] - df_rel['ball_land_x'])**2 +
            (df_rel['y'] - df_rel['ball_land_y'])**2
        )

    return df_rel


def height_to_inches(height_str):
    """Convert height string like '6-2' to inches (74)."""
    if pd.isna(height_str):
        return None
    feet, inches = height_str.split('-')
    return int(feet) * 12 + int(inches)


def process_raw_data():
    """
    Process raw CSV data into PyTorch tensors with play/player metadata.
    Returns dictionary with tensors and metadata for cross-referencing predictions.
    """
    print("=" * 60)
    print("PROCESSING RAW DATA")
    print("=" * 60)

    train_path = Path('dataset/train')
    input_files = sorted(train_path.glob('input*.csv'))

    print(f"\nLoading {len(input_files)} input files...")
    dfs = []
    for file in input_files:
        df = pd.read_csv(file)
        dfs.append(df)

    all_weeks = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(all_weeks):,}")

    print("\nStandardizing coordinates...")
    all_weeks_std = standardize(all_weeks)

    play_features = [
        'game_id', 'play_id', 'frame_id', 'nfl_id', 'player_height', 'player_weight',
        'player_position', 'player_side', 'player_role', 's', 'a', 'dir', 'o',
        'x_rel', 'y_rel', 'ball_land_x_rel', 'ball_land_y_rel'
    ]
    all_weeks_std = all_weeks_std.filter(play_features)

    print("Merging supplementary data...")
    supp = pd.read_csv('dataset/supplementary_data.csv')
    supp_features = ['game_id', 'play_id', 'down', 'yards_to_go', 'dropback_type', 'team_coverage_type']
    supp = supp.filter(supp_features)

    merged = pd.merge(left=all_weeks_std, right=supp, how='left', on=['game_id', 'play_id'])

    print("One-hot encoding categorical features...")
    merged['player_height'] = merged['player_height'].apply(height_to_inches)

    player_side_original = merged['player_side'].copy()

    categorical_cols = merged.select_dtypes(include=['object']).columns.tolist()
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_array = encoder.fit_transform(merged[categorical_cols])
    encoded_feature_names = encoder.get_feature_names_out(categorical_cols)
    encoded_df = pd.DataFrame(encoded_array, columns=encoded_feature_names, index=merged.index)

    merged_encoded = pd.concat([merged.drop(columns=categorical_cols), encoded_df], axis=1)
    merged_encoded['player_side_original'] = player_side_original
    print(f"Encoded shape: {merged_encoded.shape}")

    print("\nTransforming to ML format...")
    grouped = merged_encoded.groupby(['game_id', 'play_id'])

    agent_feature_cols = ['player_height', 'player_weight', 's', 'a', 'dir', 'o',
                          'x_rel', 'y_rel'] + [col for col in merged_encoded.columns
                                                if (col.startswith('player_position_') or
                                                    col.startswith('player_side_') or
                                                    col.startswith('player_role_')) and
                                                   col != 'player_side_original']

    global_feature_cols = ['down', 'yards_to_go'] + [col for col in merged_encoded.columns
                                                       if col.startswith('dropback_type_') or
                                                       col.startswith('team_coverage_type_')]

    trajectory_cols = ['x_rel', 'y_rel']

    historical_agent_features = []
    global_context_features = []
    ground_truth_trajectories = []
    play_metadata = []

    for (game_id, play_id), play_data in grouped:
        play_data = play_data.sort_values('frame_id')
        frames = play_data['frame_id'].unique()

        if len(frames) < 2:
            continue

        frame_data = []
        ground_truth_data = []
        player_ids = None
        player_sides = None

        for i in range(len(frames) - 1):
            current_frame = frames[i]
            next_frame = frames[i + 1]

            current_frame_players = play_data[play_data['frame_id'] == current_frame].sort_values('nfl_id')
            next_frame_players = play_data[play_data['frame_id'] == next_frame].sort_values('nfl_id')

            if player_ids is None:
                player_ids = current_frame_players['nfl_id'].astype(int).values
                player_sides = (current_frame_players['player_side_original'] == 'defense').astype(int).values

            agent_features = current_frame_players[agent_feature_cols].values
            frame_data.append(agent_features)

            next_positions = next_frame_players[trajectory_cols].values
            ground_truth_data.append(next_positions)

        historical_agent_features.append(np.array(frame_data))
        ground_truth_trajectories.append(np.array(ground_truth_data))

        global_features = play_data[global_feature_cols].iloc[0].values
        global_context_features.append(global_features)

        play_metadata.append({
            'game_id': int(game_id),
            'play_id': int(play_id),
            'player_ids': player_ids.tolist(),
            'player_sides': player_sides.tolist(),
            'n_frames': len(frame_data),
            'n_agents': len(player_ids)
        })

    print(f"Processed {len(historical_agent_features)} plays")

    print("\nConverting to PyTorch tensors...")
    print(f"DEBUG: historical_agent_features type: {type(historical_agent_features)}")
    print(f"DEBUG: First element type: {type(historical_agent_features[0])}")
    print(f"DEBUG: First element dtype: {historical_agent_features[0].dtype}")
    print(f"DEBUG: First element shape: {historical_agent_features[0].shape}")
    print(f"DEBUG: Sample values from first element:\n{historical_agent_features[0][0, 0, :]}")

    historical_agent_features_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in historical_agent_features]
    ground_truth_trajectories_tensors = [torch.tensor(arr, dtype=torch.float32) for arr in ground_truth_trajectories]
    global_context_features_tensor = torch.tensor(np.array(global_context_features), dtype=torch.float32)

    save_path = Path('dataset/processed')
    save_path.mkdir(exist_ok=True)

    torch.save({
        'historical_agent_features': historical_agent_features_tensors,
        'ground_truth_trajectories': ground_truth_trajectories_tensors,
        'global_context_features': global_context_features_tensor,
        'play_metadata': play_metadata
    }, PROCESSED_DATA_PATH)

    print(f"\n✓ Saved processed data to {PROCESSED_DATA_PATH}")
    print(f"  - {len(historical_agent_features_tensors)} plays")
    print(f"  - {len(historical_agent_features_tensors)} plays with metadata")
    print("=" * 60)

    return {
        'historical_agent_features': historical_agent_features_tensors,
        'ground_truth_trajectories': ground_truth_trajectories_tensors,
        'global_context_features': global_context_features_tensor,
        'play_metadata': play_metadata
    }


def load_or_process_data():
    """Load processed data if it exists, otherwise process raw data."""
    if PROCESSED_DATA_PATH.exists():
        print(f"✓ Loading cached data from {PROCESSED_DATA_PATH}")
        loaded_data = torch.load(PROCESSED_DATA_PATH)
        print(f"  Loaded {len(loaded_data['historical_agent_features'])} plays")
        return loaded_data
    else:
        print(f"✗ Cached data not found at {PROCESSED_DATA_PATH}")
        print("  Processing raw data...")
        return process_raw_data()


# === Model Architecture ===

class HeliocentricityTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.T_HIST, self.T_PRED, self.N_AGENTS = kwargs['T_HIST'], kwargs['T_PRED'], kwargs['N_AGENTS']
        self.D_AGENT, self.D_GLOBAL, self.D_MODEL = kwargs['D_AGENT'], kwargs['D_GLOBAL'], kwargs['D_MODEL']
        self.D_LATENT, self.N_HEADS, self.N_LAYERS = kwargs['D_LATENT'], kwargs['N_HEADS'], kwargs['N_LAYERS']
        self.KL_BETA = kwargs['KL_BETA']

        self.agent_embed = nn.Linear(self.D_AGENT, self.D_MODEL)
        self.global_embed = nn.Linear(self.D_GLOBAL, self.D_MODEL)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL,
            nhead=self.N_HEADS,
            dim_feedforward=self.D_MODEL * 4,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)

        self.mlp_prior = nn.Sequential(
            nn.Linear(self.D_MODEL, self.D_MODEL),
            nn.ReLU(),
            nn.Linear(self.D_MODEL, 2 * self.D_LATENT)
        )

        self.mlp_recognition = nn.Sequential(
            nn.Linear(self.D_MODEL + self.T_PRED * self.N_AGENTS * 2, self.D_MODEL),
            nn.ReLU(),
            nn.Linear(self.D_MODEL, 2 * self.D_LATENT)
        )

        self.mlp_decoder = nn.Sequential(
            nn.Linear(self.D_MODEL + self.D_LATENT, self.D_MODEL * 2),
            nn.ReLU(),
            nn.Linear(self.D_MODEL * 2, self.T_PRED * self.N_AGENTS * 2)
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, X_hist_agents, X_global, Y_truth=None):
        B = X_hist_agents.size(0)

        agent_emb = self.agent_embed(X_hist_agents)
        global_emb = self.global_embed(X_global)

        cls_tokens = global_emb.unsqueeze(1).unsqueeze(1).expand(-1, self.T_HIST, -1, -1)
        input_sequence = torch.cat([cls_tokens, agent_emb], dim=2)
        flat_input = input_sequence.view(B, -1, self.D_MODEL)

        encoded_output = self.transformer_encoder(flat_input)
        C = encoded_output[:, 0, :]

        mu_prior, log_var_prior = self.mlp_prior(C).chunk(2, dim=-1)

        if Y_truth is not None:
            Y_flat = Y_truth.view(B, -1)
            rec_input = torch.cat([C, Y_flat], dim=-1)
            mu_rec, log_var_rec = self.mlp_recognition(rec_input).chunk(2, dim=-1)
            Z = self.reparameterize(mu_rec, log_var_rec)
        else:
            Z = self.reparameterize(mu_prior, log_var_prior)
            mu_rec, log_var_rec = mu_prior, log_var_prior

        decoder_input = torch.cat([C, Z], dim=-1)
        Y_pred_flat = self.mlp_decoder(decoder_input)
        Y_pred = Y_pred_flat.view(B, self.T_PRED, self.N_AGENTS, 2)

        return Y_pred, mu_rec, log_var_rec, mu_prior, log_var_prior


# === Loss and Inference Functions ===

def vae_loss(Y_pred, Y_truth, mu_rec, log_var_rec, mu_prior, log_var_prior, KL_BETA):
    L_recon = F.mse_loss(Y_pred, Y_truth, reduction='sum') / Y_pred.size(0)

    kl_loss = 0.5 * torch.sum(
        log_var_prior - log_var_rec - 1
        + (torch.exp(log_var_rec) + (mu_rec - mu_prior).pow(2)) / torch.exp(log_var_prior)
    ) / Y_pred.size(0)

    total_loss = L_recon + KL_BETA * kl_loss
    return total_loss, L_recon.item(), kl_loss.item()


@torch.no_grad()
def generate_expected_trajectories(model, X_hist_agents, X_global, K=10):
    """
    Generate K diverse, plausible trajectories for the defense (E)
    by sampling the latent space Z from the prior distribution.
    """
    model.eval()
    B = X_hist_agents.size(0)

    X_hist_agents_K = X_hist_agents.repeat_interleave(K, dim=0)
    X_global_K = X_global.repeat_interleave(K, dim=0)

    Y_pred_K, _, _, _, _ = model(X_hist_agents_K, X_global_K, Y_truth=None)

    return Y_pred_K.view(B, K, model.T_PRED, model.N_AGENTS, 2)


# === Custom Dataset with Padding and Metadata Tracking ===

class FootballDataset(Dataset):
    """Dataset with padding for variable-length sequences and metadata tracking."""

    def __init__(self, hist_features, gt_trajectories, global_features, metadata,
                 max_hist_len=None, max_pred_len=None, max_n_agents=None):
        self.hist_features = hist_features
        self.gt_trajectories = gt_trajectories
        self.global_features = global_features
        self.metadata = metadata

        self.max_hist_len = max_hist_len or max(x.shape[0] for x in hist_features)
        self.max_pred_len = max_pred_len or max(y.shape[0] for y in gt_trajectories)
        self.max_n_agents = max_n_agents or max(x.shape[1] for x in hist_features)

    def __len__(self):
        return len(self.hist_features)

    def __getitem__(self, idx):
        hist = self.hist_features[idx]
        gt = self.gt_trajectories[idx]
        global_feat = self.global_features[idx]

        hist_len = hist.shape[0]
        pred_len = gt.shape[0]
        n_agents = hist.shape[1]

        if hist_len < self.max_hist_len:
            pad_hist_time = torch.zeros(self.max_hist_len - hist_len, hist.shape[1], hist.shape[2], dtype=hist.dtype)
            hist = torch.cat([hist, pad_hist_time], dim=0)
        else:
            hist = hist[:self.max_hist_len]
            hist_len = self.max_hist_len

        if n_agents < self.max_n_agents:
            pad_hist_agents = torch.zeros(hist.shape[0], self.max_n_agents - n_agents, hist.shape[2], dtype=hist.dtype)
            hist_padded = torch.cat([hist, pad_hist_agents], dim=1)
        else:
            hist_padded = hist[:, :self.max_n_agents, :]

        if pred_len < self.max_pred_len:
            pad_gt_time = torch.zeros(self.max_pred_len - pred_len, gt.shape[1], 2, dtype=gt.dtype)
            gt = torch.cat([gt, pad_gt_time], dim=0)
        else:
            gt = gt[:self.max_pred_len]
            pred_len = self.max_pred_len

        if gt.shape[1] < self.max_n_agents:
            pad_gt_agents = torch.zeros(gt.shape[0], self.max_n_agents - gt.shape[1], 2, dtype=gt.dtype)
            gt_padded = torch.cat([gt, pad_gt_agents], dim=1)
        else:
            gt_padded = gt[:, :self.max_n_agents, :]

        return hist_padded, global_feat, gt_padded, hist_len, pred_len, idx

    def get_metadata(self, idx):
        """Get play/player metadata for a specific index."""
        return self.metadata[idx]


# === Training Function ===

def create_mask(lengths, max_len, device):
    """Create attention mask: True for valid positions, False for padding."""
    batch_size = len(lengths)
    mask = torch.arange(max_len, device=device).expand(batch_size, max_len) < lengths.unsqueeze(1)
    return mask


def train_model(model, train_loader, optimizer, model_config, device, num_epochs=20):
    """Train the Heliocentricity Transformer model."""
    print(f"Starting training on {device}...")

    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_recon_loss = 0
        total_kl_loss = 0

        for batch_idx, (X_agents, X_global, Y_truth, hist_lens, pred_lens, _) in enumerate(train_loader):

            X_agents = X_agents.to(device)
            X_global = X_global.to(device)
            Y_truth = Y_truth.to(device)
            hist_lens = hist_lens.to(device)
            pred_lens = pred_lens.to(device)

            optimizer.zero_grad()

            Y_pred, mu_rec, log_var_rec, mu_prior, log_var_prior = model(X_agents, X_global, Y_truth=Y_truth)

            pred_mask = create_mask(pred_lens, model_config['T_PRED'], device)
            pred_mask_expanded = pred_mask.unsqueeze(-1).unsqueeze(-1).expand_as(Y_pred)

            Y_pred_masked = Y_pred * pred_mask_expanded
            Y_truth_masked = Y_truth * pred_mask_expanded

            L_recon = F.mse_loss(Y_pred_masked, Y_truth_masked, reduction='sum') / pred_lens.sum()

            kl_loss = 0.5 * torch.sum(
                log_var_prior - log_var_rec - 1
                + (torch.exp(log_var_rec) + (mu_rec - mu_prior).pow(2)) / torch.exp(log_var_prior)
            ) / X_agents.size(0)

            loss = L_recon + model_config['KL_BETA'] * kl_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon_loss += L_recon.item()
            total_kl_loss += kl_loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Total Loss: {total_loss / (batch_idx+1):.4f} | "
                      f"Recon: {total_recon_loss / (batch_idx+1):.4f} | "
                      f"KL: {total_kl_loss / (batch_idx+1):.4f}")

        avg_epoch_loss = total_loss / len(train_loader)
        avg_recon = total_recon_loss / len(train_loader)
        avg_kl = total_kl_loss / len(train_loader)

        history['total_loss'].append(avg_epoch_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)

        print(f"\n--- Epoch {epoch+1}/{num_epochs} Complete ---")
        print(f"Average Total Loss: {avg_epoch_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")

    return history


# === Evaluation Function ===

@torch.no_grad()
def evaluate_model(model, data_loader, dataset, device):
    """
    Evaluate the model and return results with metadata for Heliocentricity calculation.
    """
    model.eval()
    total_rmse = []
    total_recon_loss = 0
    total_kl_loss = 0

    results_for_H_calc = []

    for batch_idx, (X_agents, X_global, Y_truth, hist_lens, pred_lens, indices) in enumerate(data_loader):

        X_agents = X_agents.to(device)
        X_global = X_global.to(device)
        Y_truth = Y_truth.to(device)
        hist_lens = hist_lens.to(device)
        pred_lens = pred_lens.to(device)

        Y_pred, mu_rec, log_var_rec, mu_prior, log_var_prior = model(X_agents, X_global, Y_truth=Y_truth)

        loss, L_recon, L_KL = vae_loss(
            Y_pred, Y_truth,
            mu_rec, log_var_rec, mu_prior, log_var_prior,
            model.KL_BETA
        )
        total_recon_loss += L_recon
        total_kl_loss += L_KL

        Y_pred_np = Y_pred.cpu().numpy()
        Y_truth_np = Y_truth.cpu().numpy()

        sample_rmse = np.sqrt(mean_squared_error(Y_truth_np.reshape(-1, 1), Y_pred_np.reshape(-1, 1)))
        total_rmse.append(sample_rmse)

        K = 10
        Y_pred_K = generate_expected_trajectories(model, X_agents, X_global, K=K).cpu().numpy()

        for i in range(Y_truth_np.shape[0]):
            dataset_idx = indices[i].item()
            metadata = dataset.get_metadata(dataset_idx)

            results_for_H_calc.append({
                'Y_truth': Y_truth_np[i],
                'Y_pred': Y_pred_np[i],
                'Y_pred_K': Y_pred_K[i],
                'game_id': metadata['game_id'],
                'play_id': metadata['play_id'],
                'player_ids': metadata['player_ids'],
                'player_sides': metadata['player_sides'],
                'n_agents': metadata['n_agents'],
                'star_idx': 4  # TODO: Identify actual star receiver from metadata
            })

        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(data_loader)}")

    avg_rmse = np.mean(total_rmse)
    avg_recon = total_recon_loss / len(data_loader)
    avg_kl = total_kl_loss / len(data_loader)

    print(f"\n--- Validation Results ---")
    print(f"Avg Trajectory RMSE: {avg_rmse:.4f} meters")
    print(f"Avg Reconstruction Loss: {avg_recon:.4f}")
    print(f"Avg KL Divergence: {avg_kl:.4f}")

    return results_for_H_calc


# === Heliocentricity Calculation Utilities ===

def min_separation_distance(receiver_coords, defense_coords):
    """
    Calculate minimum separation distance between receiver and defense.

    Args:
        receiver_coords: (T_pred, 2) - receiver trajectory
        defense_coords: (T_pred, N_defenders, 2) - defense trajectories

    Returns:
        (T_pred,) array of minimum distances at each timestep
    """
    dist_to_defenders = np.linalg.norm(
        receiver_coords[:, np.newaxis, :] - defense_coords, axis=2
    )
    return np.min(dist_to_defenders, axis=1)


def compute_heliocentricity(play_data, proximity_threshold=10.0):
    """
    Compute Heliocentricity score for a single play WITH metadata tracking.

    Only considers defenders within proximity_threshold yards of the receiver at the start
    of the route to filter out irrelevant defenders.

    Args:
        play_data: Dictionary with Y_truth, Y_pred_K, star_idx, player_sides, and metadata fields
        proximity_threshold: Maximum distance (yards) at t=0 to consider a defender relevant

    Returns:
        Dictionary with H_score, H_frame_diff, and metadata
    """
    Y_truth = play_data['Y_truth']
    Y_pred_K = play_data['Y_pred_K']
    star_idx = play_data['star_idx']
    player_sides = np.array(play_data['player_sides'])
    n_agents = Y_truth.shape[1]

    def_indices = np.where(player_sides == 1)[0]
    def_indices = def_indices[def_indices < n_agents]

    if len(def_indices) == 0:
        all_indices = np.arange(n_agents)
        def_indices = all_indices[all_indices != star_idx]

    receiver_start_pos = Y_truth[0, star_idx, :]
    defenders_start_pos = Y_truth[0, def_indices, :]
    initial_distances = np.linalg.norm(defenders_start_pos - receiver_start_pos, axis=1)

    nearby_mask = initial_distances <= proximity_threshold
    relevant_def_indices = def_indices[nearby_mask]

    if len(relevant_def_indices) == 0:
        closest_indices = np.argsort(initial_distances)[:min(3, len(def_indices))]
        relevant_def_indices = def_indices[closest_indices]

    def_indices = relevant_def_indices

    actual_R_coords = Y_truth[:, star_idx, :]
    actual_D_coords = Y_truth[:, def_indices, :]
    A = min_separation_distance(actual_R_coords, actual_D_coords)

    E_K = []
    for k in range(Y_pred_K.shape[0]):
        predicted_D_coords = Y_pred_K[k][:, def_indices, :]
        E_k = min_separation_distance(actual_R_coords, predicted_D_coords)
        E_K.append(E_k)

    E_mean = np.mean(np.stack(E_K, axis=0), axis=0)

    H_frame_diff = E_mean - A
    H_score = np.mean(H_frame_diff)

    return {
        'H_score': H_score,
        'H_frame_diff': H_frame_diff,
        'game_id': play_data['game_id'],
        'play_id': play_data['play_id'],
        'player_ids': play_data['player_ids'],
        'star_player_id': play_data['player_ids'][star_idx] if star_idx < len(play_data['player_ids']) else None
    }


def compute_heliocentricity_for_all(results):
    """
    Compute Heliocentricity for all plays in evaluation results.

    Returns:
        (DataFrame with scores and metadata, list of full result dicts)
    """
    helio_results = [compute_heliocentricity(play_data) for play_data in results]

    df = pd.DataFrame([{
        'game_id': r['game_id'],
        'play_id': r['play_id'],
        'H_score': r['H_score'],
        'star_player_id': r['star_player_id']
    } for r in helio_results])

    return df, helio_results


# === Main Pipeline ===

def main():
    # --- Load / Process Data ---
    loaded_data = load_or_process_data()

    historical_agent_features = loaded_data['historical_agent_features']
    ground_truth_trajectories = loaded_data['ground_truth_trajectories']
    global_context_features = loaded_data['global_context_features']
    play_metadata = loaded_data['play_metadata']

    print(f"\nGlobal context shape: {global_context_features.shape}")
    print(f"Sample play metadata: {play_metadata[0]}")

    model_config = {
        'T_HIST': T_HIST, 'T_PRED': T_PRED, 'N_AGENTS': N_AGENTS, 'D_AGENT': D_AGENT,
        'D_GLOBAL': D_GLOBAL, 'D_MODEL': D_MODEL, 'D_LATENT': D_LATENT, 'N_HEADS': N_HEADS,
        'N_LAYERS': N_LAYERS, 'KL_BETA': KL_BETA
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # --- Build Dataset and DataLoaders ---
    dataset = FootballDataset(
        historical_agent_features,
        ground_truth_trajectories,
        global_context_features,
        play_metadata,
        max_hist_len=model_config['T_HIST'],
        max_pred_len=model_config['T_PRED'],
        max_n_agents=model_config['N_AGENTS']
    )

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"\nDataset split: Train={train_size}, Test={test_size}")
    print(f"Max hist length: {dataset.max_hist_len}, Max pred length: {dataset.max_pred_len}, Max agents: {dataset.max_n_agents}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- Initialize Model ---
    model = HeliocentricityTransformer(**model_config).to(device)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters")

    # --- Train or Load Weights ---
    PRETRAINED_WGTS = None  # Set to PRETRAINED_WEIGHTS_PATH to load saved weights

    if PRETRAINED_WGTS is None:
        print('Training model from scratch:')
        train_model(model, train_loader, optimizer, model_config, device, num_epochs=NUM_EPOCHS)
        Path('dataset/pretrained').mkdir(exist_ok=True)
        torch.save(model.state_dict(), PRETRAINED_WEIGHTS_PATH)
    else:
        print('Loading model from pretrained weights:')
        state_dict = torch.load(PRETRAINED_WGTS, map_location=device)
        model.load_state_dict(state_dict)

    # --- Evaluate ---
    all_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    results = evaluate_model(model, all_dl, dataset, device)

    # --- Compute Heliocentricity Scores ---
    helio_df, helio_results = compute_heliocentricity_for_all(results)

    print("\n=== Heliocentricity Summary ===")
    print(helio_df.describe())

    print("\n=== Top 5 Plays by Heliocentricity ===")
    print(helio_df.nlargest(5, 'H_score'))

    if helio_results:
        print("\n=== Sample Detailed Result ===")
        sample = helio_results[0]
        print(f"Game ID: {sample['game_id']}, Play ID: {sample['play_id']}")
        print(f"Star Player ID: {sample['star_player_id']}")
        print(f"Heliocentricity Score: {sample['H_score']:.4f}")
        print(f"Frame-by-frame values shape: {sample['H_frame_diff'].shape}")

    helio_df.to_csv('heliocentricity_scores.csv', index=False)
    print("\n✓ Saved heliocentricity_scores.csv")


if __name__ == '__main__':
    main()
