"""Generate the 10 overnight Stage-A YAMLs (5 cats x {token-VQA, OFT-baseline})
from the place templates. OFT warm-start backbone, 10k steps, 0526 data."""
from omegaconf import OmegaConf
from pathlib import Path

D = Path("examples/preference/train_files")
TOKEN_TMPL = D / "starvla_pref_stage_a_oftvqa_token_place.yaml"
BASE_TMPL = D / "conv_oft_warmstart_place_0526.yaml"
CATS = ["height", "orient", "contact", "hvlv", "place"]


def set_data(cfg, cat):
    cfg.datasets.vla_data.data_root_dir = f"/mnt/localssd/kaiwenh/pref/data/0526/{cat}"
    cfg.datasets.vla_data.data_mix = f"{cat}_v1"
    cfg.datasets.vla_data.stats_json_path = f"examples/preference/dataset/stats_{cat}_0526_joint.json"
    cfg.datasets.vla_data.pref_category = cat


written = []
for cat in CATS:
    # --- token-VQA (place template already exists) ---
    if cat != "place":
        c = OmegaConf.load(TOKEN_TMPL)
        c.run_id = f"pref_oftvqa_token_stage_a_{cat}"
        c.framework.vqa.category = cat
        set_data(c, cat)
        p = D / f"starvla_pref_stage_a_oftvqa_token_{cat}.yaml"
        OmegaConf.save(c, p); written.append(str(p))
    # --- OFT baseline (no VQA) ---
    b = OmegaConf.load(BASE_TMPL)
    b.run_id = f"pref_oft_baseline_stage_a_{cat}"
    b.trackers = ["jsonl", "wandb"]
    b.trainer.max_train_steps = 10000
    b.trainer.num_warmup_steps = 1000
    b.trainer.save_interval = 5000
    set_data(b, cat)
    p = D / f"starvla_pref_stage_a_oft_baseline_{cat}.yaml"
    OmegaConf.save(b, p); written.append(str(p))

print("\n".join(written))
print(f"\nTotal written: {len(written)} (+ place token already present = 10)")
