#!/bin/bash
# 2026-06-12 hvlv: b0 (Naive FT) arm, paired hv/lv, topdown camera + SWAP_RB
export STARVLA_SWAP_RB=1
bash /root/ar-research/policy/starvla_joint/run_pref_control.sh hvlv stamp_seal6_hv stamp_seal6 \
     pref_stageb_b0_hvlv 10098 1 6 0 hv lv
