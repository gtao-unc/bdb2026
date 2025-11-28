# 2026 Big Data Bowl – Dataset Description

## Summary
This document provides a summary of each dataset used in the **2026 Big Data Bowl**, including key join variables and detailed descriptions of all fields.  
The **tracking data** is provided by the **NFL Next Gen Stats** team.

---

## External Data
Participants may use **external NFL data** as long as it is **free and publicly available** to all participants.  
Examples include:
- [nflverse](https://nflverse.nflverse.com/)
- [Pro Football Reference](https://www.pro-football-reference.com/)

> **Note:**  
> The `gameId` and `playId` in the Big Data Bowl data correspond to `old_game_id` and `play_id` in `nflverse` play-by-play data.

---

## Files Overview

### `train/`
Contains the input and output tracking data for each play.

#### **Input Files:** `input_2023_w[01–18].csv`
Tracking data **before the pass is thrown**.

| Variable | Description |
|-----------|--------------|
| `game_id` | Unique game identifier (numeric) |
| `play_id` | Play identifier (numeric, not unique across games) |
| `player_to_predict` | Whether the player's x/y prediction is included in output (bool) |
| `nfl_id` | Unique player identifier (numeric) |
| `frame_id` | Frame identifier (starts at 1 for each play and type) |
| `play_direction` | Direction offense is moving (`left` or `right`) |
| `absolute_yardline_number` | Distance from end zone for possession team (numeric) |
| `player_name` | Player name (text) |
| `player_height` | Height (ft-in) |
| `player_weight` | Weight (lbs) |
| `player_birth_date` | Birth date (YYYY-MM-DD) |
| `player_position` | Player’s on-field position (text) |
| `player_side` | Team side (`Offense` or `Defense`) |
| `player_role` | Role on play (`Defensive Coverage`, `Targeted Receiver`, `Passer`, `Other Route Runner`) |
| `x`, `y` | Player field coordinates in yards (`x`: long axis 0–120, `y`: short axis 0–53.3) |
| `s` | Speed (yards/second) |
| `a` | Acceleration (yards/second²) |
| `o` | Orientation (degrees) |
| `dir` | Direction of motion (degrees) |
| `num_frames_output` | Number of frames to predict in output data |
| `ball_land_x`, `ball_land_y` | Ball landing coordinates on field (yards) |

---

#### **Output Files:** `output_2023_w[01–18].csv`
Tracking data **after the pass is thrown**.

| Variable | Description |
|-----------|--------------|
| `game_id` | Unique game identifier (numeric) |
| `play_id` | Play identifier (numeric, not unique across games) |
| `nfl_id` | Player identifier (numeric) |
| `frame_id` | Frame identifier for each output sequence (matches `num_frames_output`) |
| `x`, `y` | Player field coordinates (`x`: 0–120, `y`: 0–53.3 yards) |

---

## Supplementary Data
Contains **contextual information** about the game and play.

| Variable | Description |
|-----------|--------------|
| `game_id` | Game identifier (numeric) |
| `season` | Season of the game |
| `week` | Week of the game |
| `game_date` | Game date (MM/DD/YYYY) |
| `game_time_eastern` | Start time (HH:MM:SS, EST) |
| `home_team_abbr` | Home team code (text) |
| `visitor_team_abbr` | Visiting team code (text) |
| `home_final_score` | Home team points (numeric) |
| `visitor_final_score` | Visitor team points (numeric) |
| `play_id` | Play identifier (numeric) |
| `play_description` | Play description (text) |
| `quarter` | Game quarter (numeric) |
| `game_clock` | Time on clock (MM:SS) |
| `down` | Down number (numeric) |
| `yards_to_go` | Distance needed for first down (numeric) |
| `possession_team` | Offensive team (abbr) |
| `defensive_team` | Defensive team (abbr) |
| `yardline_side` | Team code for line of scrimmage |
| `yardline_number` | Yard line at line of scrimmage (numeric) |
| `pre_snap_home_score` | Home team score before play |
| `pre_snap_visitor_score` | Visitor team score before play |
| `pass_result` | Dropback outcome (`C`, `I`, `S`, `IN`, `R`) |
| `play_nullified_by_penalty` | Whether play was canceled by penalty (`Y`/`N`) |
| `pass_length` | Air yards beyond LOS (negative if behind LOS) |
| `offense_formation` | Offensive formation (text) |
| `receiver_alignment` | Receiver alignment (`0x0`, `1x0`, …, `3x2`) |
| `route_of_targeted_receiver` | Route run by targeted receiver |
| `play_action` | Whether play used play-action (binary) |
| `dropback_type` | QB dropback type (e.g. `Traditional`, `Scramble`, `Designed Rollout`) |
| `dropback_distance` | QB dropback distance (yards) |
| `pass_location_type` | QB throw location (`Inside Tackle Box`, `Outside Left`, `Outside Right`, `Unknown`) |
| `defenders_in_the_box` | Number of defenders near LOS |
| `team_coverage_man_zone` | Type of coverage (`Man`/`Zone`) |
| `team_coverage_type` | Specific coverage type (text) |
| `penalty_yards` | Penalty yardage gained (numeric) |
| `pre_penalty_yards_gained` | Yards gained before penalty (numeric) |
| `yards_gained` | Total yards gained including penalty (numeric) |
| `expected_points` | Expected points for the play (numeric) |
| `expected_points_added` | Change in expected points (numeric) |
| `pre_snap_home_team_win_probability` | Win probability for home team before play (numeric) |
| `pre_snap_visitor_team_win_probability` | Win probability for visiting team before play (numeric) |
| `home_team_win_probability_added` | Change in home win probability (numeric) |
| `visitor_team_win_probility_added` | Change in visitor win probability (numeric) |

---

**© 2026 Big Data Bowl – NFL Next Gen Stats**
