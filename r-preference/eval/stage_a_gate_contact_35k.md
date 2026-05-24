# Stage A gate — `contact`

- baseline ckpt: `/mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/pref_baseline_stage_a_v1_noVQA_contact/checkpoints/steps_35000_pytorch_model.pt`
- VQA ckpt:      `/mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/pref_main_stage_a_v1_VQA_contact/checkpoints/steps_35000_pytorch_model.pt`
- taskA val sample: 80 eps
- taskB sample:    100 eps

## GATE: **RED: no taskB task ≥90%, diagnose before Stage B**

- passed (≥0.90 on taskB): `[]`

## ① VQA per-task accuracy — **taskB** (GATE METRIC)
Overall: **0.610** (N=100); confidence mean=0.998, p10=1.000

| task_group | acc | N | pred_dist |
|---|---:|---:|---|
| `put_boxdrink3_plate` | 0.610 | 100 | `{'25': 89, '75': 11}` |

## ② VQA per-task accuracy — taskA val (sanity)
Overall: **0.863** (N=80); confidence mean=0.991

| task_group | acc | N |
|---|---:|---:|
| `give_boxdrink` | 1.000 | 10 |
| `give_callbell` | 1.000 | 10 |
| `give_fork` | 1.000 | 10 |
| `give_screwdriver` | 0.900 | 10 |
| `put_boxdrink_dustbin` | 1.000 | 10 |
| `put_callbell_dustbin` | 1.000 | 10 |
| `put_fork_dustbin` | 0.500 | 10 |
| `put_screwdriver_dustbin` | 0.500 | 10 |

## ③ Counterfactual action MSE — baseline ckpt (taskA val)
Overall counterfactual MSE (normalized 20D): **0.0311**

| task_group | counterfactual MSE | sign acc (strong) | N |
|---|---:|---:|---:|
| `give_boxdrink` | 0.0374 | — | 10 |
| `give_callbell` | 0.0343 | — | 10 |
| `give_fork` | 0.0418 | — | 10 |
| `give_screwdriver` | 0.0391 | — | 10 |
| `put_boxdrink_dustbin` | 0.0220 | 0.600 | 10 |
| `put_callbell_dustbin` | 0.0239 | 0.600 | 10 |
| `put_fork_dustbin` | 0.0258 | 0.700 | 10 |
| `put_screwdriver_dustbin` | 0.0245 | — | 10 |

## ④ Counterfactual action MSE — VQA ckpt (taskA val)
Overall counterfactual MSE (normalized 20D): **0.0321**

| task_group | counterfactual MSE | sign acc (strong) | N |
|---|---:|---:|---:|
| `give_boxdrink` | 0.0349 | — | 10 |
| `give_callbell` | 0.0398 | — | 10 |
| `give_fork` | 0.0477 | — | 10 |
| `give_screwdriver` | 0.0343 | — | 10 |
| `put_boxdrink_dustbin` | 0.0225 | 0.700 | 10 |
| `put_callbell_dustbin` | 0.0259 | 0.400 | 10 |
| `put_fork_dustbin` | 0.0279 | 0.500 | 10 |
| `put_screwdriver_dustbin` | 0.0239 | — | 10 |

## ⑤ Baseline vs VQA action drift (sanity — should be small)
Overall drift (normalized): **0.0586**, N=80

| task_group | drift |
|---|---:|
| `give_boxdrink` | 0.0652 |
| `give_callbell` | 0.0702 |
| `give_fork` | 0.0868 |
| `give_screwdriver` | 0.0748 |
| `put_boxdrink_dustbin` | 0.0420 |
| `put_callbell_dustbin` | 0.0436 |
| `put_fork_dustbin` | 0.0424 |
| `put_screwdriver_dustbin` | 0.0437 |

---

Notes:
- Counterfactual MSE ≈ baseline≈VQA is **expected** (both action heads read pref from prompt token). VQA's value is in (1).
- Sign accuracy reported only for contact strong tasks (`put_boxdrink_dustbin`, `put_callbell_dustbin`, `put_fork_dustbin`) per doc 0522 §2.6.
