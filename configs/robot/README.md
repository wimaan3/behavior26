# Robot configs

Put the exact robot config used for evaluation here. It is a **required submission artifact**.

- Start from `OmniGibson/omnigibson/eval/r1pro.yaml` in the BEHAVIOR-1K checkout and copy it in as `r1pro.yaml`.
- The 2026 challenge does **not** fix the embodiment — `--robot-config` accepts any OmniGibson-supported robot.
  In practice stay on R1Pro, since all 20,000 demos are R1Pro and switching discards the training set.
- The `controller_config` block *is* worth tuning (joint vs IK vs base controllers). Cheap knob, few teams touch it.
- Whatever action space you choose, the vector your policy returns must match `robot.action_dim` exactly.

Required keys: `model`, `name`, `controller_config`, `obs_modalities`, and an
`eval.camera_sensor_names` block mapping the `head`, `left_wrist`, `right_wrist` roles
(`--write-video` needs all three).
