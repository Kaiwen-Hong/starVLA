#!/usr/bin/env python3
"""
Attention comparison: Does the avoid-obstacle model actually look at obstacles?

Compares 4 models on the SAME task (place_stapler_stand):
  v41 (trained w/ cup_tray + avoid-obstacle data)   evaluated on wp4 (avoid instr)
  v42 (trained w/ cup_tray + standard data)          evaluated on wp5 (standard instr)
  v61 (trained w/ cup5_tray5 + avoid-obstacle data)  evaluated on wp4 (avoid instr)
  v62 (trained w/ cup5_tray5 + standard data)        evaluated on wp5 (standard instr)

Output: 3 focused figures per timestep + 1 summary figure.

Usage:
    python compare_attention.py \
        --v41_dir results/attention_analysis/XXX/v41 \
        --v42_dir results/attention_analysis/XXX/v42 \
        --v61_dir results/attention_analysis/XXX/v61 \
        --v62_dir results/attention_analysis/XXX/v62 \
        --output_dir results/attention_analysis/XXX/comparison_figures
"""

import argparse
import glob
import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image as PILImage

warnings.filterwarnings("ignore", category=UserWarning)

# ── Labels ─────────────────────────────────────────────────────────────────────

PAIRS = [
    # (avoid_model, standard_model, pair_label)
    ("v41", "v42", "cup_tray pair"),
    ("v61", "v62", "cup5_tray5 pair"),
]
MODEL_ORDER = ["v41", "v42", "v61", "v62"]
SHORT = {
    "v41": "v41 (avoid)", "v42": "v42 (std)",
    "v61": "v61 (avoid)", "v62": "v62 (std)",
}
COLOR = {
    "v41": "#D62728", "v42": "#1F77B4",
    "v61": "#D62728", "v62": "#1F77B4",
}
CAMERA_NAMES = ["Head Cam", "Left Cam", "Right Cam"]


# ── Normalization ──────────────────────────────────────────────────────────────

def _norm(attn, method="robust"):
    a = np.asarray(attn, dtype=np.float64)
    if a.size == 0:
        return a
    if method == "log":
        a = np.log(np.clip(a, 1e-8, None))
    if method == "power":
        a = np.power(a - a.min() + 1e-8, 2.0)
    if method == "robust":
        med = np.median(a)
        q75, q25 = np.percentile(a, [75, 25])
        iqr = q75 - q25
        if iqr > 1e-12:
            a = np.clip((a - med) / iqr, -3, 3)
            return (a + 3) / 6.0
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo + 1e-12)


def _resize(smap, h, w):
    """Resize a 2-D attention map to (h, w)."""
    return np.array(
        PILImage.fromarray((smap * 255).astype(np.uint8))
        .resize((w, h), PILImage.BILINEAR)
    ).astype(float) / 255.0


# ── Data loading ───────────────────────────────────────────────────────────────

def _load(d):
    files = sorted(glob.glob(os.path.join(d, "step_*.npz")))
    out = []
    for f in files:
        npz = np.load(f, allow_pickle=True)
        entry = {}
        for k in npz.files:
            entry[k] = npz[k]
        out.append(entry)
    return out


def _last_layer(entry):
    idxs = set()
    for k in entry:
        if k.startswith("agg_layer_"):
            idxs.add(int(k.split("_")[2]))
    return f"layer_{max(idxs)}" if idxs else None


def _agg(entry):
    lk = _last_layer(entry)
    if not lk:
        return {}
    return {
        "image": float(entry.get(f"agg_{lk}_image", 0)),
        "text":  float(entry.get(f"agg_{lk}_text", 0)),
        "action": float(entry.get(f"agg_{lk}_action", 0)),
    }


def _smap(entry, cam):
    k = f"spatial_img{cam}"
    if k not in entry:
        return None
    s = entry[k]
    return s[0] if s.ndim == 3 else s


def _img(entry, cam):
    return entry.get(f"raw_img{cam}")


def _ncams(entry):
    return sum(1 for k in entry if k.startswith("spatial_img"))


# ── Figure 1 (per step): Hero figure ──────────────────────────────────────────
# For each camera: Raw | Avoid heatmap | Standard heatmap | Difference
# Two rows of this (one per pair: v41/v42, v61/v62)

def fig_hero(all_data, step_idx, path, norm="robust"):
    """The main figure: side-by-side heatmaps + difference for each pair."""
    ref = next((all_data[n][step_idx] for n in MODEL_ORDER
                if step_idx < len(all_data[n])), None)
    if ref is None:
        return
    nc = _ncams(ref)
    if nc == 0:
        return

    # Layout: rows = pairs * cameras, cols = Raw | Avoid | Standard | Diff
    nrows = len(PAIRS) * nc
    fig, axes = plt.subplots(nrows, 4, figsize=(20, 4 * nrows), squeeze=False)

    for pi, (av, st, plabel) in enumerate(PAIRS):
        av_data = all_data[av]
        st_data = all_data[st]
        if step_idx >= len(av_data) or step_idx >= len(st_data):
            continue
        av_e, st_e = av_data[step_idx], st_data[step_idx]

        for cam in range(nc):
            row = pi * nc + cam
            ax_raw, ax_av, ax_st, ax_diff = axes[row]
            cname = CAMERA_NAMES[cam] if cam < len(CAMERA_NAMES) else f"Cam{cam}"

            raw = _img(av_e, cam)
            av_s = _smap(av_e, cam)
            st_s = _smap(st_e, cam)

            # Col 0: Raw image
            if raw is not None:
                ax_raw.imshow(raw)
            ax_raw.set_xticks([]); ax_raw.set_yticks([])

            if av_s is None or st_s is None:
                for ax in (ax_av, ax_st, ax_diff):
                    ax.axis("off")
                continue

            av_n = _norm(av_s, norm)
            st_n = _norm(st_s, norm)

            h, w = (raw.shape[:2] if raw is not None
                    else (av_n.shape[0] * 14, av_n.shape[1] * 14))

            # Col 1: Avoid model heatmap
            if raw is not None:
                ax_av.imshow(raw)
            ax_av.imshow(_resize(av_n, h, w), cmap="inferno", alpha=0.6, vmin=0, vmax=1)

            # Col 2: Standard model heatmap
            if raw is not None:
                ax_st.imshow(raw)
            ax_st.imshow(_resize(st_n, h, w), cmap="inferno", alpha=0.6, vmin=0, vmax=1)

            # Col 3: Difference (avoid - standard)
            # Resize to same grid first, then diff
            tgt_h, tgt_w = max(av_n.shape[0], st_n.shape[0]), max(av_n.shape[1], st_n.shape[1])
            av_r = _resize(av_n, tgt_h, tgt_w) if av_n.shape != (tgt_h, tgt_w) else av_n
            st_r = _resize(st_n, tgt_h, tgt_w) if st_n.shape != (tgt_h, tgt_w) else st_n
            diff = av_r - st_r
            if raw is not None:
                ax_diff.imshow(raw, alpha=0.4)
            ax_diff.imshow(_resize(diff, h, w), cmap="RdBu_r", alpha=0.7, vmin=-0.6, vmax=0.6)

            for ax in (ax_av, ax_st, ax_diff):
                ax.set_xticks([]); ax.set_yticks([])

            # Row label
            ax_raw.set_ylabel(f"{plabel}\n{cname}", fontsize=10, fontweight="bold")

    # Column titles
    col_titles = ["Observation", "Avoid-Obstacle Model", "Standard Model",
                  "Difference (Avoid − Std)\nRed=avoid looks MORE"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=11, fontweight="bold",
                             color="#D62728" if j == 1 else "#1F77B4" if j == 2 else "black")

    step_num = int(ref.get("step", step_idx))
    fig.suptitle(
        f"Where Does the Model Look? (inference call #{step_idx}, env step ~{step_num})",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2 (per step): Text vs Image attention ─────────────────────────────
# Simple grouped bars: does the avoid model rely more on text?

def fig_text_vs_image(all_data, step_idx, path):
    """Bar chart: does avoid-obstacle model attend more to text (instruction)?"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for pi, (av, st, plabel) in enumerate(PAIRS):
        ax = axes[pi]
        models = [av, st]
        x = np.arange(3)  # image, text, action
        width = 0.35

        for mi, name in enumerate(models):
            data = all_data[name]
            if step_idx >= len(data):
                continue
            a = _agg(data[step_idx])
            vals = [a.get("image", 0), a.get("text", 0), a.get("action", 0)]
            bars = ax.bar(x + mi * width, vals, width,
                          label=SHORT[name], color=COLOR[name], alpha=0.85)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(["Image\n(vision)", "Text\n(instruction)", "Action\n(self)"],
                           fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Attention Proportion" if pi == 0 else "")
        ax.set_title(plabel, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Does the avoid-obstacle model attend more to the instruction? (step {step_idx})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 3 (per step): Per-word text attention ──────────────────────────────
# THE key figure: which words in the instruction get the most attention?

OBSTACLE_KEYWORDS = {"avoid", "avoiding", "obstacle", "obstacles", "past", "around", "steer"}

def fig_per_word(all_data, step_idx, path):
    """
    Bar chart: attention weight per word in instruction.
    Highlights obstacle-related words in red.
    One subplot per model, so you can directly see if "avoid"/"obstacle" stand out.
    """
    models_with_text = []
    for name in MODEL_ORDER:
        data = all_data[name]
        if step_idx < len(data) and "text_token_words" in data[step_idx]:
            models_with_text.append(name)

    if not models_with_text:
        return

    n = len(models_with_text)
    fig, axes = plt.subplots(n, 1, figsize=(18, 3.5 * n), squeeze=False)

    for row, name in enumerate(models_with_text):
        ax = axes[row, 0]
        entry = all_data[name][step_idx]

        words = entry["text_token_words"]  # array of strings
        attn = entry["text_token_attention"]  # (num_text_tokens,)

        # Merge subword tokens into whole words for readability
        merged_words, merged_attn = _merge_subwords(words, attn)

        x = np.arange(len(merged_words))
        # Color obstacle-related words red, others blue
        colors = []
        for w in merged_words:
            w_clean = w.strip().lower().rstrip(".,;:!?")
            if w_clean in OBSTACLE_KEYWORDS:
                colors.append("#D62728")  # red
            else:
                colors.append(COLOR[name])

        ax.bar(x, merged_attn, color=colors, alpha=0.85, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(merged_words, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Attention", fontsize=10)
        ax.set_title(SHORT[name], fontsize=11, fontweight="bold", color=COLOR[name])
        ax.grid(axis="y", alpha=0.3)

        # Mark obstacle words with a star
        for i, w in enumerate(merged_words):
            w_clean = w.strip().lower().rstrip(".,;:!?")
            if w_clean in OBSTACLE_KEYWORDS:
                ax.annotate("*", (i, merged_attn[i]), ha="center", va="bottom",
                            fontsize=14, fontweight="bold", color="#D62728")

    fig.suptitle(
        f"Per-Word Text Attention (step {step_idx})\n"
        "Red bars / * = obstacle-related words. Do they get more attention in avoid models?",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _merge_subwords(words, attn):
    """Merge subword tokens that start with non-space into previous word."""
    merged_w, merged_a = [], []
    for w, a in zip(words, attn):
        w_str = str(w)
        # Qwen tokenizer: subwords that continue a word typically don't start with space
        if merged_w and not w_str.startswith(" ") and not w_str.startswith("<"):
            merged_w[-1] += w_str
            merged_a[-1] += a
        else:
            merged_w.append(w_str.strip())
            merged_a.append(a)
    return merged_w, np.array(merged_a)


# ── Figure 4 (summary): Per-word attention averaged over all steps ────────────

def fig_per_word_avg(all_data, path):
    """
    Time-averaged per-word attention. Reduces noise from single timesteps.
    Two rows: one for each pair (avoid vs standard).
    """
    fig, axes = plt.subplots(len(PAIRS), 1, figsize=(18, 5 * len(PAIRS)), squeeze=False)

    for pi, (av, st, plabel) in enumerate(PAIRS):
        ax = axes[pi, 0]
        # Collect and average per-word attention across all steps for each model
        for mi, name in enumerate([av, st]):
            data = all_data[name]
            all_words = None
            all_attn = []
            for entry in data:
                if "text_token_words" not in entry:
                    continue
                words = entry["text_token_words"]
                attn = entry["text_token_attention"]
                mw, ma = _merge_subwords(words, attn)
                if all_words is None:
                    all_words = mw
                if len(mw) == len(all_words):
                    all_attn.append(ma)

            if all_words is None or not all_attn:
                continue

            avg_attn = np.mean(all_attn, axis=0)
            x = np.arange(len(all_words))
            offset = mi * 0.35
            colors = []
            for w in all_words:
                wc = w.strip().lower().rstrip(".,;:!?")
                if wc in OBSTACLE_KEYWORDS:
                    colors.append("#D62728" if mi == 0 else "#FF7F7F")
                else:
                    colors.append(COLOR[name])

            ax.bar(x + offset, avg_attn, 0.35, color=colors, alpha=0.8,
                   label=SHORT[name], edgecolor="none")

        if all_words:
            ax.set_xticks(np.arange(len(all_words)) + 0.175)
            ax.set_xticklabels(all_words, rotation=60, ha="right", fontsize=8)
        ax.set_ylabel("Avg Attention", fontsize=10)
        ax.set_title(plabel, fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Time-Averaged Per-Word Attention\n"
        "Red = obstacle words. Does the avoid model attend more to them?",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 5 (summary): Attention evolution over the entire rollout ───────────

def fig_timeline(all_data, path):
    """
    Two panels:
      Left:  Text attention over time (4 lines) — is avoid consistently higher?
      Right: Image attention over time (4 lines)
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
    ls = {"v41": "-", "v42": "--", "v61": "-.", "v62": ":"}
    mk = {"v41": "o", "v42": "s", "v61": "^", "v62": "D"}

    for ax, tt, title in zip(axes,
                              ["text", "image"],
                              ["Attention → Text (instruction)",
                               "Attention → Image (vision)"]):
        for name in MODEL_ORDER:
            data = all_data[name]
            if not data:
                continue
            steps = [int(e.get("step", i)) for i, e in enumerate(data)]
            vals = [_agg(e).get(tt, 0) for e in data]
            ax.plot(steps, vals, linestyle=ls[name], marker=mk[name],
                    color=COLOR[name], markersize=4, linewidth=2,
                    label=SHORT[name], alpha=0.85)

        ax.set_xlabel("Environment Step", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)

    axes[0].set_ylabel("Attention Proportion", fontsize=11)

    fig.suptitle(
        "Does the avoid-obstacle model consistently attend more to the instruction?",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Figure 4 (summary): Average difference heatmap across all steps ──────────

def fig_avg_diff(all_data, path, norm="robust"):
    """
    Time-averaged spatial difference: where does the avoid model
    consistently look MORE than the standard model?
    """
    ref_entry = None
    for n in MODEL_ORDER:
        if all_data[n]:
            ref_entry = all_data[n][0]
            break
    if ref_entry is None:
        return
    nc = _ncams(ref_entry)
    if nc == 0:
        return

    fig, axes = plt.subplots(len(PAIRS), nc, figsize=(6 * nc, 5 * len(PAIRS)), squeeze=False)

    for pi, (av, st, plabel) in enumerate(PAIRS):
        av_data, st_data = all_data[av], all_data[st]
        n_common = min(len(av_data), len(st_data))
        if n_common == 0:
            continue

        for cam in range(nc):
            ax = axes[pi, cam]
            diffs = []
            raw_img_last = None
            for i in range(n_common):
                av_s = _smap(av_data[i], cam)
                st_s = _smap(st_data[i], cam)
                if av_s is None or st_s is None:
                    continue
                av_n = _norm(av_s, norm)
                st_n = _norm(st_s, norm)
                # Align shapes
                tgt_h = max(av_n.shape[0], st_n.shape[0])
                tgt_w = max(av_n.shape[1], st_n.shape[1])
                av_r = _resize(av_n, tgt_h, tgt_w) if av_n.shape != (tgt_h, tgt_w) else av_n
                st_r = _resize(st_n, tgt_h, tgt_w) if st_n.shape != (tgt_h, tgt_w) else st_n
                diffs.append(av_r - st_r)
                raw_img_last = _img(av_data[i], cam)

            if not diffs:
                ax.axis("off")
                continue

            avg_diff = np.mean(diffs, axis=0)

            h, w = (raw_img_last.shape[:2] if raw_img_last is not None
                    else (avg_diff.shape[0] * 14, avg_diff.shape[1] * 14))
            if raw_img_last is not None:
                ax.imshow(raw_img_last, alpha=0.4)
            im = ax.imshow(_resize(avg_diff, h, w), cmap="RdBu_r", alpha=0.75,
                           vmin=-0.4, vmax=0.4)
            ax.set_xticks([]); ax.set_yticks([])

            cname = CAMERA_NAMES[cam] if cam < len(CAMERA_NAMES) else f"Cam{cam}"
            if pi == 0:
                ax.set_title(cname, fontsize=11, fontweight="bold")
            if cam == 0:
                ax.set_ylabel(plabel, fontsize=11, fontweight="bold")

    fig.suptitle(
        f"Time-Averaged Attention Difference (over {n_common} steps)\n"
        "Persistent red = avoid model consistently looks here more",
        fontsize=13, fontweight="bold", y=1.03,
    )
    fig.tight_layout()
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-0.4, 0.4)),
        cax=cbar_ax, label="Avoid − Standard",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--v41_dir", type=str, required=True)
    parser.add_argument("--v42_dir", type=str, required=True)
    parser.add_argument("--v61_dir", type=str, required=True)
    parser.add_argument("--v62_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--norm_method", type=str, default="robust",
                        choices=["min_max", "log", "robust", "power"])
    parser.add_argument("--max_steps", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading attention data...")
    dirs = {"v41": args.v41_dir, "v42": args.v42_dir,
            "v61": args.v61_dir, "v62": args.v62_dir}
    all_data = {}
    for name, d in dirs.items():
        data = _load(d)
        print(f"  {name}: {len(data)} snapshots")
        all_data[name] = data

    total = sum(len(v) for v in all_data.values())
    if total == 0:
        print("ERROR: No attention data found.")
        return

    # Pick representative steps
    lengths = [len(v) for v in all_data.values() if v]
    max_len = max(lengths) if lengths else 0
    if max_len <= args.max_steps:
        rep = list(range(max_len))
    else:
        rep = np.linspace(0, max_len - 1, args.max_steps, dtype=int).tolist()

    print(f"\n--- Generating {len(rep)} per-step figures + 2 summary figures ---")

    per_step = os.path.join(args.output_dir, "per_step")
    os.makedirs(per_step, exist_ok=True)

    for idx in rep:
        tag = f"step_{idx:03d}"
        print(f"  {tag}...")

        # Fig 1: Hero — Raw | Avoid | Standard | Difference
        fig_hero(all_data, idx, os.path.join(per_step, f"{tag}_hero.png"), args.norm_method)

        # Fig 2: Text vs Image bar chart
        fig_text_vs_image(all_data, idx, os.path.join(per_step, f"{tag}_text_vs_image.png"))

        # Fig 3: Per-word text attention (THE key figure)
        fig_per_word(all_data, idx, os.path.join(per_step, f"{tag}_per_word.png"))

    # Summary figures
    print("  summary: attention timeline...")
    fig_timeline(all_data, os.path.join(args.output_dir, "timeline.png"))

    print("  summary: average difference heatmap...")
    fig_avg_diff(all_data, os.path.join(args.output_dir, "avg_difference.png"), args.norm_method)

    print("  summary: per-word attention (averaged)...")
    fig_per_word_avg(all_data, os.path.join(args.output_dir, "per_word_avg.png"))

    print(f"\nDone! Figures in {args.output_dir}")
    print(f"  Per-step:  {per_step}/")
    print(f"  Summary:   {args.output_dir}/timeline.png")
    print(f"             {args.output_dir}/avg_difference.png")
    print(f"             {args.output_dir}/per_word_avg.png  <-- KEY FIGURE")


if __name__ == "__main__":
    main()
