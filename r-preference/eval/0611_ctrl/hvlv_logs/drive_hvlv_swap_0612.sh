#!/bin/bash
# 2026-06-12 hvlv diagnosis run 1: main_hvlv_geom@1500, paired hv/lv,
# WITH STARVLA_SWAP_RB=1 (deploy obs R/B-swapped to match the BGR-stored training JPEGs)
export STARVLA_SWAP_RB=1
bash /root/ar-research/policy/starvla_joint/run_pref_control.sh hvlv stamp_seal6_hv stamp_seal6 \
     pref_stageb_main_hvlv_geom 10097 1 6 0 hv lv
