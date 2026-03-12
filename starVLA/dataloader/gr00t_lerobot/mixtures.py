"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],
    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "fourier_gr1_unified_1000": [
        ("gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
        ("gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000", 1.0, "fourier_gr1_arms_waist"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],


    "SO101_pick": [
        ("pick_dataset_name", 1.0, "SO101"),
    ],

    "arx_x5": [
        ("arx_x5", 1.0, "arx_x5"),
    ],

    "robotwin": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_bell", 1.0, "robotwin"),
        ("dump_bin_bigbin", 1.0, "robotwin"),
        ("grab_roller", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("hanging_mug", 1.0, "robotwin"),
        ("lift_pot", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_playingcard_away", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("open_laptop", 1.0, "robotwin"),
        ("open_microwave", 1.0, "robotwin"),
        ("pick_diverse_bottles", 1.0, "robotwin"),
        ("pick_dual_bottles", 1.0, "robotwin"),
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
        ("place_bread_basket", 1.0, "robotwin"),
        ("place_bread_skillet", 1.0, "robotwin"),
        ("place_burger_fries", 1.0, "robotwin"),
        ("place_can_basket", 1.0, "robotwin"),
        ("place_cans_plasticbox", 1.0, "robotwin"),
        ("place_container_plate", 1.0, "robotwin"),
        ("place_dual_shoes", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_fan", 1.0, "robotwin"),
        ("place_mouse_pad", 1.0, "robotwin"),
        ("place_object_basket", 1.0, "robotwin"),
        ("place_object_scale", 1.0, "robotwin"),
        ("place_object_stand", 1.0, "robotwin"),
        ("place_phone_stand", 1.0, "robotwin"),
        ("place_shoe", 1.0, "robotwin"),
        ("press_stapler", 1.0, "robotwin"),
        ("put_bottles_dustbin", 1.0, "robotwin"),
        ("put_object_cabinet", 1.0, "robotwin"),
        ("rotate_qrcode", 1.0, "robotwin"),
        ("scan_object", 1.0, "robotwin"),
        ("shake_bottle", 1.0, "robotwin"),
        ("shake_bottle_horizontally", 1.0, "robotwin"),
        ("stack_blocks_three", 1.0, "robotwin"),
        ("stack_blocks_two", 1.0, "robotwin"),
        ("stack_bowls_three", 1.0, "robotwin"),
        ("stack_bowls_two", 1.0, "robotwin"),
        ("stamp_seal", 1.0, "robotwin"),
        ("turn_switch", 1.0, "robotwin"),
    ],

    "robotwin_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],
    "robotwin_task2": [
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
    ],

    "custom_all": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("adjust_bottle_wp1", 1.0, "robotwin"),
        ("adjust_bottle_wp2", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("beat_block_hammer_wp1", 1.0, "robotwin"),
        ("beat_block_hammer_wp2", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_rgb_wp1", 1.0, "robotwin"),
        ("blocks_ranking_rgb_wp2", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),
        ("blocks_ranking_size_wp1", 1.0, "robotwin"),
        ("blocks_ranking_size_wp2", 1.0, "robotwin"),
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_alarmclock_wp1", 1.0, "robotwin"),
        ("click_alarmclock_wp2", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_block_wp1", 1.0, "robotwin"),
        ("handover_block_wp2", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),
        ("handover_mic_wp1", 1.0, "robotwin"),
        ("handover_mic_wp2", 1.0, "robotwin"),
        ("move_can_pot", 1.0, "robotwin"),
        ("move_can_pot_wp1", 1.0, "robotwin"),
        ("move_can_pot_wp2", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),
        ("move_pillbottle_pad_wp1", 1.0, "robotwin"),
        ("move_pillbottle_pad_wp2", 1.0, "robotwin"),
        ("move_stapler_pad", 1.0, "robotwin"),
        ("move_stapler_pad_wp1", 1.0, "robotwin"),
        ("move_stapler_pad_wp2", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_empty_cup_wp1", 1.0, "robotwin"),
        ("place_empty_cup_wp2", 1.0, "robotwin"),
    ],

    # Debug subset: just 1 custom task
    "custom_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],

    # ── Custom v0218: 27 variants (removed wp1/wp2 for 3 tasks with obstacle issues) ──
    "custom_v0218": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("adjust_bottle_wp1", 1.0, "robotwin"),
        ("adjust_bottle_wp2", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("beat_block_hammer_wp1", 1.0, "robotwin"),
        ("beat_block_hammer_wp2", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_rgb_wp1", 1.0, "robotwin"),
        ("blocks_ranking_rgb_wp2", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),           # clean only
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_alarmclock_wp1", 1.0, "robotwin"),
        ("click_alarmclock_wp2", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_block_wp1", 1.0, "robotwin"),
        ("handover_block_wp2", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),                  # clean only
        ("move_can_pot", 1.0, "robotwin"),
        ("move_can_pot_wp1", 1.0, "robotwin"),
        ("move_can_pot_wp2", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),           # clean only
        ("move_stapler_pad", 1.0, "robotwin"),
        ("move_stapler_pad_wp1", 1.0, "robotwin"),
        ("move_stapler_pad_wp2", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_empty_cup_wp1", 1.0, "robotwin"),
        ("place_empty_cup_wp2", 1.0, "robotwin"),
    ],

    # Debug subset: just 1 v0218 task
    "custom_v0218_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],

    # ── Custom v0225_v3: 20 variants (clean + hv/lv, 10 tasks) ──
    # 5 tasks with clean/hv/lv, 5 tasks with clean only
    "custom_v0225_v3": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("adjust_bottle_hv", 1.0, "robotwin"),
        ("adjust_bottle_lv", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("beat_block_hammer_hv", 1.0, "robotwin"),
        ("beat_block_hammer_lv", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_rgb_hv", 1.0, "robotwin"),
        ("blocks_ranking_rgb_lv", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),           # clean only
        ("handover_block", 1.0, "robotwin"),
        ("handover_block_hv", 1.0, "robotwin"),
        ("handover_block_lv", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),                  # clean only
        ("move_can_pot", 1.0, "robotwin"),
        ("move_can_pot_hv", 1.0, "robotwin"),
        ("move_can_pot_lv", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),           # clean only
        ("move_stapler_pad", 1.0, "robotwin"),              # clean only
        ("place_empty_cup", 1.0, "robotwin"),               # clean only
    ],

    # Debug subset: just 1 v0225_v3 task
    "custom_v0225_v3_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],

    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        # ("OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],

    # ── FastUMI: single-arm, wrist camera, 10D EEF + rot6d ──
    "fastumi_pickandplace": [
        ("pickandplace_vla", 1.0, "fastumi"),
    ],

    "fastumi_pickandplace_real_0307": [
        ("pickandplace-real-0307", 1.0, "fastumi"),
    ],

    # Debug subset: just 1 FastUMI task
    "fastumi_task1": [
        ("pickandplace_vla", 1.0, "fastumi"),
    ],

    # Multi-task (add more as you convert them)
    "fastumi_all": [
        ("pickandplace_vla", 1.0, "fastumi"),
        # ("handover_all",   1.0, "fastumi"),
    ],
}
