"""
FastUMI Data Config for StarVLA.

Copy this class into starVLA/dataloader/gr00t_lerobot/data_config.py
and register it in ROBOT_TYPE_CONFIG_MAP at the bottom of that file.

Robot: FastUMI Pro (single-arm, Kinova Kortex, wrist camera)
State: [x, y, z, rot6d(6), gripper] = 10D absolute EEF pose
Action: [rel_x, rel_y, rel_z, rel_rot6d(6), gripper] = 10D pre-computed relative
Camera: 1x wrist (256x256)

Key design decisions:
  - action_mode should be "abs" because actions are ALREADY relative
    (inv(base_pose) @ target_pose). No further delta computation needed.
  - rotation is stored as rotation_6d (continuous representation, no gimbal lock).
    We use target_rotations to tell StarVLA about the rotation encoding.
  - gripper is binary (0=open, 1=closed) so we use "binary" normalization.
  - eef_pos and eef_rot6d use "min_max" normalization to scale to [-1, 1].
  - action horizon = 16 (matching the Diffusion Policy training config).
"""

# ---- Paste the following into data_config.py ----

# Add this class BEFORE the ROBOT_TYPE_CONFIG_MAP dict:

class FastUMIDataConfig:
    """FastUMI Pro: single-arm robot with wrist camera, 10D EEF pose + rot6d."""

    video_keys = [
        "video.wrist",
    ]
    state_keys = [
        "state.eef_pos",
        "state.eef_rot6d",
        "state.gripper",
    ]
    action_keys = [
        "action.eef_pos",
        "action.eef_rot6d",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        video_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.video_keys,
        )
        state_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.state_keys,
        )
        action_modality = ModalityConfig(
            delta_indices=self.action_indices,
            modality_keys=self.action_keys,
        )
        language_modality = ModalityConfig(
            delta_indices=self.observation_indices,
            modality_keys=self.language_keys,
        )
        modality_configs = {
            "video": video_modality,
            "state": state_modality,
            "action": action_modality,
            "language": language_modality,
        }
        return modality_configs

    def transform(self):
        transforms = [
            # state transforms
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(
                apply_to=self.state_keys,
                normalization_modes={
                    "state.eef_pos": "min_max",
                    "state.eef_rot6d": "min_max",
                    "state.gripper": "binary",
                },
            ),
            # action transforms
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.eef_pos": "min_max",
                    "action.eef_rot6d": "min_max",
                    "action.gripper": "binary",
                },
            ),
        ]
        return ComposedModalityTransform(transforms=transforms)


# ---- Then add this line inside ROBOT_TYPE_CONFIG_MAP: ----
#
# ROBOT_TYPE_CONFIG_MAP = {
#     ...existing entries...
#     "fastumi": FastUMIDataConfig(),   # <-- add this
# }
