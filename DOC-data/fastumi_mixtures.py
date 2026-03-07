"""
FastUMI mixture registration for StarVLA.

Add these entries to starVLA/dataloader/gr00t_lerobot/mixtures.py
inside the DATASET_NAMED_MIXTURES dict.

Each tuple: (folder_name_under_data_root_dir, sampling_weight, robot_type)
"""

# ---- Add inside DATASET_NAMED_MIXTURES in mixtures.py ----

# Single task
"fastumi_pickandplace": [
    ("pickandplace_vla", 1.0, "fastumi"),
],

# Multiple tasks (when you have more categories converted)
"fastumi_all": [
    ("pickandplace_vla", 1.0, "fastumi"),
    ("handover_all",     1.0, "fastumi"),
    # ("scene1_div1_2",  1.0, "fastumi"),
    # ("scene2_div1_2",  1.0, "fastumi"),
    # ...add more task folders here
],
