"""
Stage B dataset for the preference-conditioned VLA continue-training pipeline.

Wraps PrefHDF5Dataset's HDF5 / action / image loading, but REPLACES the
prompt-pref source with a pre-cached pseudo-label JSON (main mode) or
strips the pref suffix entirely (B0 mode).

Critical firewall (see r-preference/doc/0524-stageB-contact-plan.md §2.3):
  - This dataset's __getitem__ must NEVER use the dir-name-parsed pref_key
    for label/prompt construction. TaskB dir names (e.g.
    `put_boxdrink3_plate_25/_75`) contain ground-truth labels that would
    contaminate the "unlabeled B" claim if read.
  - The cache loader is a strict whitelist: only `action_prompt_label`
    and `decision` (+ `pref_key`, the cache's own re-derivation of the
    label) are read. `gt_*` fields, if present in the cache, are IGNORED
    by this dataset. (They exist in the cache solely for the §3.2
    launch-gate verification step, which is OUT of the train path.)
  - Defensive runtime assert in __init__:
      main mode  (with_pref_suffix=True)  → cache_path required, must exist
      B0 mode    (with_pref_suffix=False) → cache_path may be None
  - Rejected episodes (cache `decision != "keep"`) are SKIPPED from
    `_index` entirely — they neither appear in __len__ nor are returned
    by __getitem__.

B0 behaviour:
  - The prompt is built WITHOUT the " Preference: ..." suffix.
  - Internally we use `pref_key="25"` as a placeholder when calling
    `build_action_prompt` to retrieve the base template, then strip
    the trailing suffix. (The base template is pref-independent —
    confirmed by leak-regex check on the contact template.)
  - This means B0 dataset doesn't even read the cache.

Usage (set via YAML `datasets.vla_data`):
  dataset_py: pref_hdf5_stageb
  task_groups: ["put_boxdrink3_plate"]    # contact taskB
  pref_keys: ["25", "75"]                  # enumerated for index building only
  pref_category: contact
  pseudo_label_cache: r-preference/eval/pref_pseudo_labels_contact_B.json
  with_pref_suffix: true                   # main; false = B0
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .pref_hdf5_dataset import PrefHDF5Dataset
from .prompt import PREF_CATEGORIES, build_action_prompt


# Whitelist of cache fields this dataset may read. Defensive against the
# cache JSON containing `gt_pref_key` / `gt_match` / etc (which it does,
# for §3.2 launch-gate purposes — but only the labeler script touches those).
_CACHE_WHITELIST = frozenset({"action_prompt_label", "decision", "pref_key"})


class PrefHDF5StageBDataset(PrefHDF5Dataset):
    """Stage B continue-train dataset with GT-leakage firewall.

    Two modes (set by `with_pref_suffix` ctor arg):
      - main (with_pref_suffix=True): lang = "<base>. Preference: <cache_label>"
      - B0   (with_pref_suffix=False): lang = "<base>"  (no Preference suffix)
    """

    def __init__(
        self,
        *args,
        pseudo_label_cache: Optional[str | Path] = None,
        with_pref_suffix: bool = True,
        **kwargs,
    ):
        # Force a deterministic "placeholder" pref_key path. We need
        # _task_dirs_for_groups to enumerate ALL dirs (both _25 and _75)
        # so we can iterate every episode regardless of GT pref. The pref
        # we ASSIGN to each episode at __getitem__ time comes from the
        # cache, never from the dir-name parse.
        super().__init__(*args, **kwargs)

        self.with_pref_suffix = bool(with_pref_suffix)
        self.pseudo_label_cache_path = (
            Path(pseudo_label_cache) if pseudo_label_cache else None
        )

        # Defensive asserts (firewall §2.3)
        if self.with_pref_suffix:
            if self.pseudo_label_cache_path is None:
                raise ValueError(
                    "PrefHDF5StageBDataset(with_pref_suffix=True) requires "
                    "pseudo_label_cache JSON; got None. (Main mode reads "
                    "pseudo-labels from cache; never from dir name.)"
                )
            if not self.pseudo_label_cache_path.exists():
                raise FileNotFoundError(
                    f"pseudo_label_cache not found: {self.pseudo_label_cache_path}"
                )
            self._cache: Dict[str, Dict[str, str]] = self._load_cache(
                self.pseudo_label_cache_path
            )
        else:
            # B0 mode: no cache, no pref suffix. Defensive guard.
            if self.pseudo_label_cache_path is not None:
                print(
                    f"[PrefHDF5StageBDataset/B0] WARNING: "
                    f"pseudo_label_cache={self.pseudo_label_cache_path!s} provided "
                    f"but with_pref_suffix=False — cache will NOT be read."
                )
            self._cache = {}

        # Pre-compute the prompt-stripping regex for B0 mode (strip
        # trailing " Preference: <label>" if any sneaks through).
        self._pref_suffix_re = re.compile(r"\s+Preference:\s+.*$")

        # Now filter _index based on cache `decision` (main) or keep all (B0).
        if self.with_pref_suffix:
            self._filter_index_by_cache()

        # Re-derive bookkeeping
        self._summarize_index()

    # -- cache loading + filtering -----------------------------------------

    @staticmethod
    def _load_cache(path: Path) -> Dict[str, Dict[str, str]]:
        """Load + whitelist-filter cache. Raises if entries malformed."""
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"cache root must be dict; got {type(raw).__name__}")
        filtered: Dict[str, Dict[str, str]] = {}
        for ep_key, entry in raw.items():
            if not isinstance(entry, dict):
                raise ValueError(
                    f"cache entry for {ep_key!r} must be dict; got {type(entry).__name__}"
                )
            # WHITELIST: drop everything except the safe fields. Even if the
            # cache JSON has `gt_pref_key`, we don't load it.
            safe = {k: v for k, v in entry.items() if k in _CACHE_WHITELIST}
            # Required fields check
            if "decision" not in safe:
                raise ValueError(f"cache entry {ep_key!r} missing 'decision'")
            if safe["decision"] == "keep":
                if "action_prompt_label" not in safe:
                    raise ValueError(
                        f"cache entry {ep_key!r} decision=keep but no action_prompt_label"
                    )
            filtered[ep_key] = safe
        # Loud rejection check: anyone reading the cache dict won't be
        # able to see gt_* even if they typo'd
        for ep_key, safe in filtered.items():
            for k in safe:
                if k not in _CACHE_WHITELIST:
                    raise RuntimeError(
                        f"firewall broken: cache key {k!r} not in whitelist"
                    )
        return filtered

    def _filter_index_by_cache(self) -> None:
        """Drop _index entries whose episode is rejected by cache."""
        before = len(self._index)
        new_index: List[Tuple[str, str, str, int, int]] = []
        n_no_cache = 0
        n_rejected = 0
        for entry in self._index:
            task_dir, tg, pk_dir, ep_id, frame_idx = entry
            ep_key = f"{task_dir}/episode{ep_id}"
            cached = self._cache.get(ep_key)
            if cached is None:
                n_no_cache += 1
                continue
            if cached["decision"] != "keep":
                n_rejected += 1
                continue
            new_index.append(entry)
        self._index = new_index
        after = len(self._index)
        print(
            f"[PrefHDF5StageBDataset/{self.split}] cache filter: "
            f"frame-samples {before} -> {after} "
            f"(no_cache={n_no_cache}, rejected={n_rejected})"
        )

    def _summarize_index(self) -> None:
        unique_eps = set((td, ep) for td, _, _, ep, _ in self._index)
        per_task: Dict[str, int] = {}
        for td, _, _, _, _ in self._index:
            per_task[td] = per_task.get(td, 0) + 1
        print(
            f"[PrefHDF5StageBDataset/{self.split}] "
            f"unique_eps={len(unique_eps)}  frame_samples={len(self._index)}  "
            f"per_task_dir={per_task}  "
            f"with_pref_suffix={self.with_pref_suffix}"
        )

    # -- skip paraphrase loading (taskB has no instructions/, and we
    #    overwrite lang anyway) -----------------------------------------------

    def _load_paraphrase(self, task_dir: str, ep_id: int):
        """Stage B taskB has no instructions/ — always return None.

        The parent's `_load_paraphrase` opens an instructions JSON
        unconditionally, which doesn't exist for taskB. We short-circuit
        before parent touches the disk. The returned None feeds parent's
        `build_action_prompt(..., paraphrase=None)`, which for contact
        (template-only path) goes to clean_template. The resulting `lang`
        is then OVERWRITTEN in our __getitem__ anyway — but having parent
        not crash is required to reach our overwrite point.
        """
        return None

    # -- __getitem__ override (the firewall point) -------------------------

    def __getitem__(self, idx: int) -> dict:
        # First, do everything the parent does EXCEPT prompt construction.
        # We need: image, action, state from parent; lang we'll override.
        task_dir, tg, pk_DIR_IGNORE_THIS, ep_id, frame_idx = self._index[idx]
        # Note: pk_DIR_IGNORE_THIS is parsed from dir name — DO NOT USE
        # for prompt construction. We get the label from cache instead.

        # Call parent's __getitem__ for image/action/state.
        # Parent will internally use pk_DIR for its own prompt — we then
        # OVERWRITE 'lang' to remove the leakage path. Slightly wasteful
        # (parent builds a prompt we discard) but cheap (string ops) and
        # keeps the action/image loading logic in one place.
        sample = super().__getitem__(idx)

        # === FIREWALL POINT: rebuild lang from cache, NOT from pk_DIR ===
        if self.with_pref_suffix:
            ep_key = f"{task_dir}/episode{ep_id}"
            cached = self._cache.get(ep_key)
            if cached is None:
                raise RuntimeError(
                    f"firewall: _index has {ep_key!r} but cache doesn't — "
                    f"_filter_index_by_cache should have removed this. "
                    f"This is a bug."
                )
            # cache provides the safe pref label directly
            base_prompt = self._get_base_template(tg)
            lang = f"{base_prompt} Preference: {cached['action_prompt_label']}"
        else:
            # B0 mode: just the base template, NO Preference suffix
            base_prompt = self._get_base_template(tg)
            lang = base_prompt
            # Sanity: assert no Preference: snuck in
            if "Preference:" in lang:
                # Strip it (defensive, but shouldn't happen given template_only path)
                lang = self._pref_suffix_re.sub("", lang)

        sample["lang"] = lang
        sample["language"] = lang
        sample["robot_tag"] = "pref_hdf5_stageb"
        return sample

    def _get_base_template(self, task_group: str) -> str:
        """Get the pref-independent base template for a task_group.

        For contact (template-only path), this is just
        PREF_CATEGORIES["contact"].clean_templates[task_group]. The
        helper exists for clarity and future-cat extension.
        """
        cat_cfg = PREF_CATEGORIES[self.category]
        if task_group not in cat_cfg.clean_templates:
            raise KeyError(
                f"task_group {task_group!r} not in PREF_CATEGORIES[{self.category!r}].clean_templates"
            )
        return cat_cfg.clean_templates[task_group]


def get_pref_stageb_dataset(data_cfg, mode: str = "train", **kwargs) -> PrefHDF5StageBDataset:
    """Factory mirroring `get_pref_dataset` signature."""
    def _g(name, default):
        return data_cfg.get(name, default) if hasattr(data_cfg, "get") else default

    chunk_size = int(_g("future_action_window_size", 15)) + 1
    task_groups = _g("task_groups", None)
    pref_keys = _g("pref_keys", None)
    # Stage B uses ALL episodes (no val split — see §4.4). Default 0.0 here;
    # YAML can override if you really want a val split for something.
    val_fraction = float(_g("split_val_fraction", 0.0))
    return PrefHDF5StageBDataset(
        data_root_dir=data_cfg.data_root_dir,
        split=mode,
        chunk_size=chunk_size,
        past_window=int(_g("past_action_window_size", 0)),
        include_state=bool(_g("include_state", False)),
        cameras=tuple(_g("cameras", ("head_camera", "left_camera", "right_camera"))),
        image_size=tuple(_g("image_size", (224, 224))),
        stats_json_path=_g("stats_json_path", None),
        category=str(_g("pref_category", "contact")),
        action_space=str(_g("action_space", "ee")),
        task_groups=tuple(task_groups) if task_groups else None,
        pref_keys=tuple(pref_keys) if pref_keys else None,
        split_val_fraction=val_fraction,
        # Stage B specific
        pseudo_label_cache=_g("pseudo_label_cache", None),
        with_pref_suffix=bool(_g("with_pref_suffix", True)),
        data_cfg=data_cfg,
        **kwargs,
    )


def collate_fn(batch):
    return batch
