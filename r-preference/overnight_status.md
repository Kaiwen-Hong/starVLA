# Overnight closed-loop status

Last updated: 2026-05-25T12:41:33Z
Log root: (not yet created — `~/closed_loop_logs/` does not exist on 5090)

## Status: WAITING FOR LAUNCH

The wrapper has not yet created `~/closed_loop_logs/` on `kaiwen@100.97.239.33`.
No running processes match `closed.loop|robotwin|eval|policy_server` on that host.
Checkpoints are staged at `/home/kaiwen/starVLA_runs/results/Checkpoints/`:
- `pref_baseline_stage_a_v1_noVQA_contact`
- `pref_main_stage_a_v1_VQA_contact`
- `pref_baseline_stage_a_v1_noVQA_place`

Monitoring agent will keep polling every ~10 min and update this doc when the run starts.

## Per-arm progress

| Arm | Task | Eps done | Success rate | Last log line |
|---|---|---|---|---|
| a_baseline | put_boxdrink3_plate_25 | -/10 | - | (not started) |
| a_baseline | put_boxdrink3_plate_75 | -/10 | - | (not started) |
| a_main | put_boxdrink3_plate_25 | -/10 | - | (not started) |
| a_main | put_boxdrink3_plate_75 | -/10 | - | (not started) |
| place_baseline | place_soap2_stand_center | -/10 | - | (not started) |
| place_baseline | place_soap2_stand_corner | -/10 | - | (not started) |

## Server status

| Arm | Server log tail (last 3 lines) | Errors detected? |
|---|---|---|
| a_baseline | (no log yet) | n/a |
| a_main | (no log yet) | n/a |
| place_baseline | (no log yet) | n/a |

## Anomalies / failures

- 2026-05-25T12:41:33Z: log root `~/closed_loop_logs/` not yet created on 5090. Run not started.

## Wall time / ETA

- Monitoring agent started: 2026-05-25T12:41:33Z
- Run started: (pending)
- Elapsed: 0
- Remaining (estimate): unknown until run starts
