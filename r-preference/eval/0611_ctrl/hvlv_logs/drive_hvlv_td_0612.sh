#!/bin/bash
# 2026-06-12 hvlv fix run: main_hvlv_geom@1500, paired hv/lv,
# TOPDOWN head camera (stamp_seal6_*.yml embodiment -> aloha-agilex-topdown,
# matching the 0528 hvlv collection) + STARVLA_SWAP_RB=1 (match BGR-stored training JPEGs)
export STARVLA_SWAP_RB=1
bash /root/ar-research/policy/starvla_joint/run_pref_control.sh hvlv stamp_seal6_hv stamp_seal6 \
     pref_stageb_main_hvlv_geom 10097 1 6 0 hv lv
