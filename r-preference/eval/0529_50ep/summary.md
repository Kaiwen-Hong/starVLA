| ckpt (cat) | scale | metric | prompt-A (mean,n,succ) | prompt-B (mean,n,succ) | separation | follow (midpoint) | follow (FIXED src-thr) |
|---|---|---|---|---|---|---|---|
| pref_stageb_main_height | 50ep | drop_height (m) | high: 0.1327 (n=49, 50/50) | low: 0.0622 (n=49, 41/50) | 0.0705 | 0.81 | 0.84 |
| pref_stageb_b0_height | 10ep-0611 | drop_height (m) | high: 0.0885 (n=10, 11/11) | low: 0.0601 (n=10, 11/11) | 0.0283 | 0.8 | 0.55 |
| pref_stageb_main_orient | 50ep | grasp_tilt ee_x (deg) | 0: 81.0812 (n=49, 48/50) | 90: 33.1912 (n=49, 43/50) | 47.89 | 0.81 | 0.81 |
| pref_stageb_main_orient_geom | 10ep-0611 | grasp_tilt ee_x (deg) | 0: 81.6097 (n=10, 11/11) | 90: 3.5406 (n=10, 9/11) | 78.0692 | 1.0 | 1.0 |
| pref_stageb_b0_orient | 10ep-0611 | grasp_tilt ee_x (deg) | 0: 81.8369 (n=10, 10/11) | 90: 4.1136 (n=10, 10/11) | 77.7233 | 1.0 | 1.0 |
| pref_stageb_main_contact_geom | 5ep-sample | EE.z-obj.z @grasp (m) | 25: 0.0882 (n=4, 4/5) | 75: 0.1093 (n=4, 4/5) | 0.021 | 0.75 | — |
| pref_stageb_main_hvlv_geom | 5ep-sample | max EE detour (m) | hv: 0.0439 (n=4, 0/5) | lv: 0.045 (n=4, 0/5) | -0.0011 | 0.5 | 0.5 |
