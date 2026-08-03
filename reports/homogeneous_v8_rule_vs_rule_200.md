# Homogeneous v8 rule-vs-rule attackability report

- config: `configs/homogeneous_3v3_main_v8.yaml`
- config_sha256: `ab921d4ad7d1569b8aa2b881af9030d41995f1ecd1710fbdba767294411a1533`
- episodes: 200
- seed_start: 200000
- num_envs/env_workers: 8 / 4
- red_policy: `paper_nearest_pursuit_v1`
- blue_policy: `paper_nearest_pursuit_v1`
- classification: **A1**

## Outcomes

- environment_red_outcome_rate: 0.0
- environment_blue_outcome_rate: 1.0
- draw_rate: 0.0
- neutral_rule_red_win_rate: 0.0
- neutral_rule_blue_win_rate: 0.0
- neutral_rule_draw_rate: 1.0

## Attack statistics

- red_any_attack_kill_rate: 0.32
- blue_any_attack_kill_rate: 0.32
- red_attack_kill_count_distribution: `{'0': 136, '1': 64, '2': 0, '3': 0}`
- blue_attack_kill_count_distribution: `{'0': 136, '1': 64, '2': 0, '3': 0}`
- mean_red_first_attack_kill_step: 428.25
- mean_blue_first_attack_kill_step: 428.25
- mean_red_remaining_steps_after_first_kill: 171.75
- mean_blue_remaining_steps_after_first_kill: 171.75

## Tactical-window rates

- red_r3_active_step_rate: 0.04471666666666667
- blue_r3_active_step_rate: 0.04471666666666667
- red_r41_active_step_rate: 0.0
- blue_r41_active_step_rate: 0.0
- red_r42_active_step_rate: 0.0
- blue_r42_active_step_rate: 0.0
- red_attack_window_step_rate: 0.0005333333333333334
- blue_attack_window_step_rate: 0.0005333333333333334

This report is generated from the JSON payload. It diagnoses attackability only; it does not justify changing rewards, attack ranges, angle gates, or max_steps.
