"""Generate OFT Stage-B YAMLs (main + b0, 5 cats) from the oft_baseline templates.

main: warm-start from the token-VQA Stage-A ckpt, condition on pseudo-pref suffix
      (pref_hdf5_stageb + with_pref_suffix=true + pseudo_label_cache).
b0:   warm-start from the OFT-baseline Stage-A ckpt, no pref suffix (control).
Both: L_action only (plain QwenOFT, no VQA), ~1500 steps, LR /10 vs Stage A.
pretrained_checkpoint uses the repo-relative results/ symlink so it resolves on
whichever machine holds that Stage-A ckpt.
"""
from omegaconf import OmegaConf
from pathlib import Path
from examples.preference.dataset.prompt import PREF_CATEGORIES

D = Path("examples/preference/train_files")
TASKB_GROUP = {
    "height": "place_playingcards1_box", "orient": "move_can5_away",
    "contact": "put_boxdrink3_plate", "place": "place_soap2_stand", "hvlv": "stamp_seal6",
}
CKPT = "results/Checkpoints/{rid}/checkpoints/steps_10000_pytorch_model.pt"
written = []

for cat in ["height", "orient", "contact", "place", "hvlv"]:
    pc = PREF_CATEGORIES[cat]
    base = OmegaConf.load(D / f"starvla_pref_stage_a_oft_baseline_{cat}.yaml")
    for mode in ["main", "b0"]:
        c = OmegaConf.create(OmegaConf.to_yaml(base))  # deep copy
        c.run_id = f"pref_stageb_{mode}_{cat}"
        c.trackers = ["jsonl", "wandb"]
        # --- dataset: Stage-B firewall ---
        v = c.datasets.vla_data
        v.dataset_py = "pref_hdf5_stageb"
        v.data_root_dir = f"/mnt/localssd/kaiwenh/pref/data/0526/{cat}/taskB"
        v.data_mix = f"{cat}_taskB_{mode}"
        v.task_groups = [TASKB_GROUP[cat]]
        v.pref_keys = list(pc.pref_keys)
        v.split_val_fraction = 0.0
        v.with_pref_suffix = (mode == "main")
        if mode == "main":
            v.pseudo_label_cache = f"r-preference/eval/pref_pseudo_labels_{cat}_B.json"
            stageA = f"pref_oftvqa_token_{cat}_10k"
        else:
            stageA = f"pref_oft_baseline_{cat}_10k"
        # --- trainer: warm-start + LR/10 + short ---
        t = c.trainer
        t.pretrained_checkpoint = CKPT.format(rid=stageA)
        t.max_train_steps = 1500
        t.num_warmup_steps = 150
        t.save_interval = 250
        t.eval_interval = 100000
        t.learning_rate.base = 1.0e-6
        t.learning_rate.qwen_vl_interface = 1.0e-6
        t.learning_rate.action_model = 1.0e-5
        if "vqa_state_proj" in t.learning_rate:
            del t.learning_rate["vqa_state_proj"]
        t.scheduler_specific_kwargs.min_lr = 1.0e-7
        p = D / f"starvla_pref_stageb_{mode}_{cat}.yaml"
        OmegaConf.save(c, p); written.append(str(p))

print("\n".join(written))
print(f"\nTotal: {len(written)} Stage-B YAMLs")
