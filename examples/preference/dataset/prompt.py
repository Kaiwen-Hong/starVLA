"""
Prompt assembly for the preference-conditioned VLA baseline / main-method.

Supports 4 categories with distinct strip strategies:
  giveobj (legacy)  v5 hybrid regex strip + 8-task CLEAN_TEMPLATE fallback
  height            template-only (paraphrase ignored; 81.5% scan pass below 99% gate)
  hvlv              broader-sep first-split + leak regex (100% pre-flight pass)
  orient            broader-sep first-split + tight leak regex (100% pre-flight pass)

Pre-flight scan results (2026-05-23 across 160k paraphrases per category):
  hvlv broader-sep:                                 160k/160k = 100.000%
  orient broader-sep + tight LEAK_RE (drop side):   160k/160k = 100.000%
  height broader-sep + tight LEAK_RE:               130.5k/160k = 81.562%  → template_only

The strip's only job is to produce a base prompt with NO pref leak. Pref enters
the prompt at exactly one place: the appended ' Preference: <label>' suffix.
Otherwise the baseline vs main-method (+VQA) ablation is contaminated.

Public API:
  PREF_CATEGORIES[category]: PrefCategory dataclass
  build_action_prompt(task_group, pref_key, paraphrase=None, category="giveobj") -> str
  strip_v5(phrase, task_group) -> (base, mode)   # legacy giveobj path only
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# ============================================================
# giveobj (legacy contact category — same data, old naming)
# ============================================================

GIVEOBJ_PREF_LABELS = {"25": "low contact", "75": "high contact"}

GIVEOBJ_TASK_GROUPS = (
    "give_boxdrink", "give_callbell", "give_fork", "give_screwdriver",
    "put_boxdrink_dustbin", "put_callbell_dustbin",
    "put_fork_dustbin", "put_screwdriver_dustbin",
)

GIVEOBJ_CLEAN_TEMPLATE = {
    "give_boxdrink":          "Hand the box drink over to the other side of the table.",
    "give_callbell":          "Hand the call bell over to the other side of the table.",
    "give_fork":              "Hand the fork over to the other side of the table.",
    "give_screwdriver":       "Hand the screwdriver over to the other side of the table.",
    "put_boxdrink_dustbin":   "Put the box drink into the dustbin.",
    "put_callbell_dustbin":   "Put the call bell into the dustbin.",
    "put_fork_dustbin":       "Put the fork into the dustbin.",
    "put_screwdriver_dustbin":"Put the screwdriver into the dustbin.",
}

# v5 hybrid: broad regex catching "grasping/holding ... top/bottom [of X]".
_STRIP_RE = re.compile(
    r"(,\s*)?(?:while\s+)?(?:by\s+|and\s+|before\s+)?"
    r"(?:grasping|gripping|holding|grabbing|picking(?:\s+up)?|taking|lifting|"
    r"grasped|gripped|held|grabbed|picked|lifted|grab|pick|hold|grip|lift|"
    r"hand(?:ing)?\s+(?:it\s+)?over|carry(?:ing)?|carrying|carried)\s+"
    r"(?:it\s+|the\s+object\s+|the\s+\w+\s+|of\s+(?:the\s+)?\w+\s+)?"
    r"(?:on|from|at|by|near|along|around|with|in|via|using)?\s*"
    r"(?:its\s+|the\s+)?(?:top|bottom|upper|lower|higher|high|low|base)"
    r"(?:\s+(?:part|end|section|portion|side|half|region|area|of|tip|cap))?"
    r"[^\.,]*?(?=[\.,]|$)",
    re.IGNORECASE,
)
_LEAK_WORDS_GIVEOBJ = (
    "top", "bottom", "low", "high", "upper", "lower", "height", "tall",
    "short", "above", "below", "elevated", "raised", "halfway", "midway",
    "base",
)
_LEAK_RE_GIVEOBJ = re.compile(r"\b(" + "|".join(_LEAK_WORDS_GIVEOBJ) + r")\b", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# ============================================================
# height (16 tasks: move|place × 4 obj × {high,low})
# Strategy: template_only — paraphrase ignored.
# Reason: pre-flight 81.5% pass even with broader sep + tightened leak
# regex; 16.75% paraphrases have no `,|and|while|...` separator at all,
# and the structural leaks like "Set the mouse on the stand from above"
# can't be cleanly stripped. Templates are simpler and 100% clean.
# ============================================================

HEIGHT_PREF_LABELS = {"high": "high drop", "low": "low drop"}

HEIGHT_TASK_GROUPS = (
    "move_mouse_pad", "move_pillbottle_pad",
    "move_playingcards_pad", "move_soap_pad",
    "place_mouse_stand", "place_pillbottle_stand",
    "place_playingcards_stand", "place_soap_stand",
)

HEIGHT_CLEAN_TEMPLATE = {
    "move_mouse_pad":           "Move the mouse onto the pad.",
    "move_pillbottle_pad":      "Move the pill bottle onto the pad.",
    "move_playingcards_pad":    "Move the playing cards onto the pad.",
    "move_soap_pad":            "Move the soap onto the pad.",
    "place_mouse_stand":        "Place the mouse on the stand.",
    "place_pillbottle_stand":   "Place the pill bottle on the stand.",
    "place_playingcards_stand": "Place the playing cards on the stand.",
    "place_soap_stand":         "Place the soap on the stand.",
}

# ============================================================
# hvlv (16 tasks: place × 4 obj × {plate,right} × {hv,lv})
# Strategy: broader-sep first-split + leak regex (100% scan pass).
# ============================================================

HVLV_PREF_LABELS = {"hv": "wide detour", "lv": "narrow detour"}

HVLV_TASK_GROUPS = (
    "place_apple_plate", "place_apple_right",
    "place_cup_plate", "place_cup_right",
    "place_hamburg_plate", "place_hamburg_right",
    "place_seal_plate", "place_seal_right",
)

HVLV_CLEAN_TEMPLATE = {
    # Fallback only (100% scan pass so paraphrase path almost always taken)
    "place_apple_plate":   "Place the apple on the plate.",
    "place_apple_right":   "Place the apple on the right side.",
    "place_cup_plate":     "Place the cup on the plate.",
    "place_cup_right":     "Place the cup on the right side.",
    "place_hamburg_plate": "Place the hamburger on the plate.",
    "place_hamburg_right": "Place the hamburger on the right side.",
    "place_seal_plate":    "Place the seal stamp on the plate.",
    "place_seal_right":    "Place the seal stamp on the right side.",
}

_LEAK_RE_HVLV = re.compile(
    # \w* suffix catches adverb forms (closer, closely, narrower, etc).
    r"\b(far|close|near|wide|narrow|detour|berth|arc|gap|distance|obstacle|around)\w*",
    re.IGNORECASE,
)

# ============================================================
# orient (16 tasks: place × 4 obj × {box,left} × {0,90})
# Strategy: broader-sep first-split + tight leak regex (drop side/above/
# top/down which appear in base sentences for left-side-placing tasks).
# 100% scan pass.
# ============================================================

ORIENT_PREF_LABELS = {"0": "horizontal grasp", "90": "vertical grasp"}

ORIENT_TASK_GROUPS = (
    "place_bottle_box", "place_bottle_left",
    "place_callbell_box", "place_callbell_left",
    "place_can_box", "place_can_left",
    "place_chipstub_box", "place_chipstub_left",
)

ORIENT_CLEAN_TEMPLATE = {
    "place_bottle_box":      "Place the bottle into the box.",
    "place_bottle_left":     "Place the bottle on the left side.",
    "place_callbell_box":    "Place the call bell into the box.",
    "place_callbell_left":   "Place the call bell on the left side.",
    "place_can_box":         "Place the can into the box.",
    "place_can_left":        "Place the can on the left side.",
    "place_chipstub_box":    "Place the chip tube into the box.",
    "place_chipstub_left":   "Place the chip tube on the left side.",
}

_LEAK_RE_ORIENT = re.compile(
    # \w* suffix catches adverb forms ("vertically", "horizontally", "flatly").
    # Deliberately drops side/above/top/down (occur in base task descriptors like
    # "left side", "into the box from above") — verified ≥99% clean by scan.
    r"\b(horizontal|vertical|sideways|level|flat)\w*",
    re.IGNORECASE,
)

# ============================================================
# place (16 tasks: move|place × 4 obj × {pad,tray} × {center,corner})
# Strategy: template_only — paraphrase ignored.
# Reason: pre-flight 137,200 paraphrases × 4 strip strategies, max pass
# 47.74% (broad-sep). ~52% of paraphrases lack any separator and embed
# the pref word inline ("Place the pillbottle at the center of the
# tray."). Templates are the only ≥99% clean option.
# ============================================================

PLACE_PREF_LABELS = {"center": "center placement", "corner": "corner placement"}

PLACE_TASK_GROUPS = (
    "move_mouse_pad", "move_pillbottle_pad",
    "move_playingcards_pad", "move_soap_pad",
    "place_mouse_tray", "place_pillbottle_tray",
    "place_playingcards_tray", "place_soap_tray",
)

PLACE_CLEAN_TEMPLATE = {
    "move_mouse_pad":          "Move the mouse onto the pad.",
    "move_pillbottle_pad":     "Move the pill bottle onto the pad.",
    "move_playingcards_pad":   "Move the playing cards onto the pad.",
    "move_soap_pad":           "Move the soap onto the pad.",
    "place_mouse_tray":        "Place the mouse on the tray.",
    "place_pillbottle_tray":   "Place the pill bottle on the tray.",
    "place_playingcards_tray": "Place the playing cards on the tray.",
    "place_soap_tray":         "Place the soap on the tray.",
}

# Broader separator shared by hvlv + orient.
_SEP_RE_BROAD = re.compile(r",|\s+(?:and|while|by|before|after)\s+", re.IGNORECASE)
MIN_BASE_LEN = 15


# ============================================================
# Registry
# ============================================================

@dataclass(frozen=True)
class PrefCategory:
    """Category-level config for prompt assembly.

    sep_re/leak_re convention:
      - both None: template-only (height; or giveobj which has its own
        v5 hybrid strip path branched in build_action_prompt).
      - both set: use _broad_first_split + leak check (hvlv, orient).
    """
    name: str
    task_groups: Tuple[str, ...]
    pref_keys: Tuple[str, ...]
    pref_labels: Dict[str, str]
    clean_templates: Dict[str, str]
    sep_re: Optional[re.Pattern]
    leak_re: Optional[re.Pattern]


PREF_CATEGORIES: Dict[str, PrefCategory] = {
    "giveobj": PrefCategory(
        name="giveobj",
        task_groups=GIVEOBJ_TASK_GROUPS,
        pref_keys=("25", "75"),
        pref_labels=GIVEOBJ_PREF_LABELS,
        clean_templates=GIVEOBJ_CLEAN_TEMPLATE,
        sep_re=None,
        leak_re=None,
    ),
    # `contact` is the canonical name for the legacy `giveobj` category
    # (same data, same prompts, same pref keys). Disk path renamed
    # /mnt/.../pref/data/giveobj -> /mnt/.../pref/data/contact;
    # `giveobj/` is kept as a backward-compat symlink. Use this key in
    # new YAML / launch scripts; `giveobj` remains for old ckpts.
    "contact": PrefCategory(
        name="contact",
        task_groups=GIVEOBJ_TASK_GROUPS,
        pref_keys=("25", "75"),
        pref_labels=GIVEOBJ_PREF_LABELS,
        clean_templates=GIVEOBJ_CLEAN_TEMPLATE,
        sep_re=None,
        leak_re=None,
    ),
    "height": PrefCategory(
        name="height",
        task_groups=HEIGHT_TASK_GROUPS,
        pref_keys=("high", "low"),
        pref_labels=HEIGHT_PREF_LABELS,
        clean_templates=HEIGHT_CLEAN_TEMPLATE,
        sep_re=None,
        leak_re=None,
    ),
    "hvlv": PrefCategory(
        name="hvlv",
        task_groups=HVLV_TASK_GROUPS,
        pref_keys=("hv", "lv"),
        pref_labels=HVLV_PREF_LABELS,
        clean_templates=HVLV_CLEAN_TEMPLATE,
        sep_re=_SEP_RE_BROAD,
        leak_re=_LEAK_RE_HVLV,
    ),
    "orient": PrefCategory(
        name="orient",
        task_groups=ORIENT_TASK_GROUPS,
        pref_keys=("0", "90"),
        pref_labels=ORIENT_PREF_LABELS,
        clean_templates=ORIENT_CLEAN_TEMPLATE,
        sep_re=_SEP_RE_BROAD,
        leak_re=_LEAK_RE_ORIENT,
    ),
    "place": PrefCategory(
        name="place",
        task_groups=PLACE_TASK_GROUPS,
        pref_keys=("center", "corner"),
        pref_labels=PLACE_PREF_LABELS,
        clean_templates=PLACE_CLEAN_TEMPLATE,
        sep_re=None,
        leak_re=None,
    ),
}

# ============================================================
# Legacy module-level exports (giveobj backward-compat)
# Existing pref_hdf5_dataset / precompute_stats / smoke imports still work.
# ============================================================

PREF_LABELS = GIVEOBJ_PREF_LABELS
TASK_GROUPS = GIVEOBJ_TASK_GROUPS
CLEAN_TEMPLATE = GIVEOBJ_CLEAN_TEMPLATE
_LEAK_RE = _LEAK_RE_GIVEOBJ

OBJ_PRETTY = {
    "boxdrink": "box drink",
    "callbell": "call bell",
    "fork": "fork",
    "screwdriver": "screwdriver",
}


# ============================================================
# Strip / build functions
# ============================================================

def strip_v5(phrase: str, task_group: str) -> Tuple[str, str]:
    """giveobj v5 hybrid strip — kept unchanged for ablation lock."""
    if task_group not in GIVEOBJ_CLEAN_TEMPLATE:
        raise KeyError(
            f"strip_v5 expects a giveobj task_group; got {task_group!r}. "
            f"Other categories use build_action_prompt with category arg."
        )
    stripped, n_sub = _STRIP_RE.subn("", phrase)
    stripped = _WS_RE.sub(" ", stripped).strip()
    if not stripped.endswith("."):
        stripped += "."
    if n_sub > 0 and not _LEAK_RE_GIVEOBJ.search(stripped) and len(stripped) > MIN_BASE_LEN:
        return stripped, "stripped"
    return GIVEOBJ_CLEAN_TEMPLATE[task_group], "fallback"


def _broad_first_split(phrase: str, sep_re: re.Pattern) -> Optional[str]:
    """Split phrase at the first sep match; return the (period-terminated) base."""
    m = sep_re.search(phrase)
    if not m:
        return None
    base = phrase[:m.start()].rstrip()
    return base + "." if not base.endswith(".") else base


def build_action_prompt(
    task_group: str,
    pref_key: str,
    paraphrase: Optional[str] = None,
    category: str = "giveobj",
) -> str:
    """Assemble: '<clean base prompt> Preference: <pref label>'.

    Per-category strip strategy:
      giveobj  v5 hybrid strip → fallback to CLEAN_TEMPLATE on leak/miss.
      height   ignore paraphrase; always use CLEAN_TEMPLATE.
      hvlv     broader-sep first-split; fallback on no-sep/short/leak.
      orient   same as hvlv with tighter leak regex.
      place    ignore paraphrase; always use CLEAN_TEMPLATE (same as height).

    Args:
      task_group: e.g. 'give_boxdrink' (giveobj), 'move_mouse_pad' (height
                  or place), 'place_apple_plate' (hvlv), 'place_bottle_box'
                  (orient), 'place_mouse_tray' (place).
      pref_key:   per-category pref keys — see PREF_CATEGORIES[c].pref_keys.
      paraphrase: optional 'seen' phrase; if None or fallback triggered,
                  uses CLEAN_TEMPLATE.
      category:   one of {'giveobj','contact','height','hvlv','orient','place'}.
    """
    cat = PREF_CATEGORIES.get(category)
    if cat is None:
        raise KeyError(
            f"Unknown category {category!r}; expected one of {tuple(PREF_CATEGORIES)}"
        )
    if pref_key not in cat.pref_labels:
        raise KeyError(
            f"Unknown pref_key {pref_key!r} for category {category!r}; "
            f"expected one of {tuple(cat.pref_labels)}"
        )
    if task_group not in cat.clean_templates:
        raise KeyError(
            f"Unknown task_group {task_group!r} for category {category!r}; "
            f"expected one of {tuple(cat.clean_templates)}"
        )

    if category == "giveobj":
        if paraphrase is None:
            base = cat.clean_templates[task_group]
        else:
            base, _ = strip_v5(paraphrase, task_group)
    elif cat.sep_re is None:
        # template-only (height)
        base = cat.clean_templates[task_group]
    else:
        # broader-sep + leak-check (hvlv, orient)
        base = None
        if paraphrase is not None:
            cand = _broad_first_split(paraphrase, cat.sep_re)
            if cand is not None and len(cand) >= MIN_BASE_LEN and not cat.leak_re.search(cand):
                base = cand
        if base is None:
            base = cat.clean_templates[task_group]

    return f"{base} Preference: {cat.pref_labels[pref_key]}"


# ============================================================
# Sanity tests
# ============================================================

def _sanity():
    # ----- giveobj (legacy) -----
    s, m = strip_v5(
        "Handover the boxdrink to the other side of the table, grasping on the bottom of the boxdrink.",
        "give_boxdrink",
    )
    assert m == "stripped" and not _LEAK_RE_GIVEOBJ.search(s), (s, m)

    p25 = build_action_prompt(
        "give_boxdrink", "25",
        paraphrase="Handover the boxdrink to the other side of the table, grasping on the bottom of the boxdrink.",
        category="giveobj",
    )
    assert p25.endswith(" Preference: low contact"), p25

    # Backward-compat: no category kwarg defaults to giveobj.
    p25_legacy = build_action_prompt("give_boxdrink", "25",
        paraphrase="Handover the boxdrink to the other side of the table, grasping on the bottom of the boxdrink.")
    assert p25_legacy == p25

    # ----- height (template only, paraphrase ignored) -----
    p_h_high = build_action_prompt("move_mouse_pad", "high",
        paraphrase="Whatever paraphrase, ignored.", category="height")
    assert p_h_high == "Move the mouse onto the pad. Preference: high drop", p_h_high
    p_h_low = build_action_prompt("place_pillbottle_stand", "low", paraphrase=None, category="height")
    assert p_h_low == "Place the pill bottle on the stand. Preference: low drop", p_h_low

    # ----- hvlv (broader sep + leak re) -----
    # Successful split case.
    p_v = build_action_prompt(
        "place_apple_plate", "hv",
        paraphrase="Put the apple on the plate, staying well far from the can obstacle.",
        category="hvlv",
    )
    base_v = p_v.split(" Preference:")[0]
    assert "far" not in base_v.lower() and "obstacle" not in base_v.lower(), p_v
    assert p_v.endswith(" Preference: wide detour"), p_v

    # "and" separator works too.
    p_v2 = build_action_prompt(
        "place_cup_plate", "lv",
        paraphrase="Place the cup on the plate and remain close to the obstacle.",
        category="hvlv",
    )
    base_v2 = p_v2.split(" Preference:")[0]
    assert "close" not in base_v2.lower() and "obstacle" not in base_v2.lower(), p_v2

    # Fallback when no sep at all (rare; should fallback to template).
    p_v3 = build_action_prompt(
        "place_hamburg_right", "hv",
        paraphrase="No separator paraphrase here.",
        category="hvlv",
    )
    # Without sep, fallback template kicks in.
    assert p_v3.startswith(HVLV_CLEAN_TEMPLATE["place_hamburg_right"]), p_v3

    # paraphrase=None -> template
    p_v4 = build_action_prompt("place_seal_plate", "lv", paraphrase=None, category="hvlv")
    assert p_v4 == HVLV_CLEAN_TEMPLATE["place_seal_plate"] + " Preference: narrow detour", p_v4

    # ----- orient (tight leak regex; 'side' deliberately removed) -----
    # "side" must NOT be flagged as leak in base.
    p_o = build_action_prompt(
        "place_bottle_left", "0",
        paraphrase="Place the bottle on the toy car's left side, holding it horizontally.",
        category="orient",
    )
    base_o = p_o.split(" Preference:")[0]
    assert "horizontal" not in base_o.lower(), p_o
    assert "left side" in base_o.lower(), p_o  # base must keep 'side' as action descriptor
    assert p_o.endswith(" Preference: horizontal grasp"), p_o

    # "vertical" in base => leak => fallback.
    p_o2 = build_action_prompt(
        "place_bottle_box", "90",
        paraphrase="Place the bottle into the box vertically and let go.",
        category="orient",
    )
    # Note: "and let go" should be picked as sep — "Place the bottle into the box vertically." has leak ('vertical')
    # so fallback template kicks in.
    base_o2 = p_o2.split(" Preference:")[0]
    assert "vertical" not in base_o2.lower(), p_o2

    # ----- error paths -----
    try:
        build_action_prompt("give_boxdrink", "high", paraphrase=None, category="giveobj")
        assert False, "should have raised on wrong pref_key"
    except KeyError:
        pass
    try:
        build_action_prompt("move_mouse_pad", "0", paraphrase=None, category="height")
        assert False, "should have raised on wrong pref_key for height"
    except KeyError:
        pass
    try:
        build_action_prompt("move_mouse_pad", "high", paraphrase=None, category="bogus")
        assert False, "should have raised on unknown category"
    except KeyError:
        pass

    print("prompt.py sanity OK (giveobj + height + hvlv + orient)")


if __name__ == "__main__":
    _sanity()
