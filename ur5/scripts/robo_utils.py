"""Robot control utilities: math, model loading, robot helpers, control loops."""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image

from starVLA.model.framework.base_framework import baseframework

# ── Repo / module setup ─────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))

import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera

import modular_policy
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), "real_world", "robot_extrinsics"
)

# ── Constants ────────────────────────────────────────────────────────
BASE_IN_WORLD = {
    "left":  np.load(os.path.join(_extrinsics_dir, "left_base_pose_in_world.npy")),
    "right": np.load(os.path.join(_extrinsics_dir, "right_base_pose_in_world.npy")),
}
ROBOT_IPS = {"left": "192.168.0.3", "right": "192.168.0.2"}
HOME_POSES_WORLD = {
    "left":  [0.30, -0.2, 0.25, -2.2192, 2.2148, 0.0091, 1],
    "right": [-0.1, -0.3, 0.25,  2.2419, -2.1984, 0.0166, 1],
}
CONTROL_HZ = 20
INTERP_MULT = 5
SERVO_HZ = CONTROL_HZ * INTERP_MULT  # 100 Hz
Z_MIN_WORLD = 0.001793 - 0.02


# ── Coordinate transforms ───────────────────────────────────────────

def base_to_world(pose_base, T_bw):
    p = list(pose_base)
    p[0] += T_bw[0, 3]; p[1] += T_bw[1, 3]; p[2] += T_bw[2, 3]
    return p


def world_to_base(pose_world, T_bw):
    p = list(pose_world)
    p[0] -= T_bw[0, 3]; p[1] -= T_bw[1, 3]; p[2] -= T_bw[2, 3]
    return p


# ── Rotation conversions ────────────────────────────────────────────

def rot6d_to_axisangle(d6: np.ndarray) -> np.ndarray:
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=0)
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def axisangle_to_rot6d(rx, ry, rz):
    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    return R[:2, :].flatten().astype(np.float32)


# ── Action conversions ──────────────────────────────────────────────

def ee_pose_to_state10d(pose_6d, gripper: float) -> np.ndarray:
    x, y, z, rx, ry, rz = pose_6d[:6]
    rot6d = axisangle_to_rot6d(rx, ry, rz)
    return np.array([x, y, z, *rot6d, gripper], dtype=np.float32)


def action_10d_to_delta7d(action_10d: np.ndarray) -> np.ndarray:
    delta = np.zeros(7, dtype=np.float32)
    delta[:3] = action_10d[:3]
    delta[3:6] = rot6d_to_axisangle(action_10d[3:9])
    delta[6] = action_10d[9]
    return delta


def actions_10d_to_7d(actions_10d: np.ndarray) -> np.ndarray:
    T = actions_10d.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        out[t, :3] = actions_10d[t, :3]
        out[t, 3:6] = rot6d_to_axisangle(actions_10d[t, 3:9])
        out[t, 6] = actions_10d[t, 9]
    return out


def accumulate_deltas(current_pose_world, deltas_7d):
    T = deltas_7d.shape[0]
    poses = np.zeros((T, 7), dtype=np.float64)
    pos = np.array(current_pose_world[:3], dtype=np.float64)
    rot = np.array(current_pose_world[3:6], dtype=np.float64)
    for t in range(T):
        pos = pos + deltas_7d[t, :3]
        rot = rot + deltas_7d[t, 3:6]
        poses[t, :3] = pos
        poses[t, 3:6] = rot
        poses[t, 6] = deltas_7d[t, 6]
    return poses


# ── Interpolation & timing ──────────────────────────────────────────

def interpolate_waypoints(start_pose, waypoints, mult):
    all_poses = []
    prev = start_pose
    for wp in waypoints:
        for j in range(1, mult + 1):
            alpha = j / mult
            all_poses.append(prev + alpha * (wp - prev))
        prev = wp
    return np.array(all_poses, dtype=np.float64)


def precise_wait(t_end: float, slack: float = 0.001):
    remaining = t_end - time.monotonic()
    if remaining > 0:
        if remaining > slack:
            time.sleep(remaining - slack)
        while time.monotonic() < t_end:
            pass


# ── Model loading ────────────────────────────────────────────────────

def _detect_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path: str):
    import torch
    from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
    from starVLA.model.framework import build_framework

    print(f"Loading model from: {checkpoint_path}")
    t0 = time.time()
    checkpoint_pt = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    config.framework.qwenvl.attn_implementation = _detect_attn()

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats
    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda").eval()

    print(f"Model loaded in {time.time() - t0:.1f}s ({config.framework.name})")
    return model


def get_action_stats(model):
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")
    return action_stats, dataset_key


def build_example(image: Image.Image, instruction: str,
                  state_10d: np.ndarray = None) -> dict:
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example


# ── Robot connection & control ───────────────────────────────────────

def connect_robot(arm):
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    ip = ROBOT_IPS[arm]
    print(f"Connecting to {arm} arm at {ip}...")
    rtde_c = RTDEControlInterface(ip)
    rtde_r = RTDEReceiveInterface(ip)
    return rtde_c, rtde_r, ip


def connect_gripper(robot_ip):
    from robotiq_gripper import RobotiqGripper
    print("Connecting to gripper...")
    g = RobotiqGripper()
    g.connect(hostname=robot_ip, port=63352)
    return g


def go_home(rtde_c, rtde_r, arm, T_bw, robot_ip):
    home = HOME_POSES_WORLD[arm]
    home_base = world_to_base(home[:6], T_bw)
    target_joints = rtde_c.getInverseKinematics(home_base)
    print(f"Moving to home pose (world): {[round(x, 2) for x in home[:6]]}")
    rtde_c.moveJ(target_joints, 1.0, 1.0)

    current_base = rtde_r.getActualTCPPose()
    current_world = base_to_world(current_base, T_bw)
    print(f"Reached: {[round(x, 4) for x in current_world]}")

    from robotiq_gripper import RobotiqGripper
    g = RobotiqGripper()
    g.connect(hostname=robot_ip, port=63352)
    g.move(int((1.0 - home[6]) * 255), 255, 150)
    g.disconnect()
    print("Gripper opened.")
    return home[6]


def compute_waypoints(current_pos, actions_10d, n_exec, fix_rotation=True):
    """Convert delta actions to absolute world-frame waypoints.

    Returns: (waypoints (n_exec, 6), gripper_transitions [(index, value), ...])
    """
    waypoints = np.zeros((n_exec, 6), dtype=np.float64)
    gripper_transitions = []
    pos = current_pos.copy()

    for i in range(n_exec):
        delta = action_10d_to_delta7d(actions_10d[i])
        if fix_rotation:
            delta[3:6] = 0.0
        pos = pos + delta[:6]
        if pos[2] < Z_MIN_WORLD:
            print(f"  [SAFETY] z clamped: {pos[2]:.4f} -> {Z_MIN_WORLD:.4f}")
            pos[2] = Z_MIN_WORLD
        waypoints[i] = pos
        gripper_transitions.append((i, float(delta[6])))

    return waypoints, gripper_transitions


def execute_servo(rtde_c, current_pos, waypoints, T_bw,
                  gripper_hw, current_gripper, gripper_transitions):
    """Interpolate waypoints to 100 Hz and execute via servoL (sync, with servoStop).

    Returns: (exec_ms, new_gripper_value)
    """
    servo_dt = 1.0 / SERVO_HZ
    interp = interpolate_waypoints(current_pos, waypoints, INTERP_MULT)

    for _, new_val in gripper_transitions:
        if (new_val > 0.5) != (current_gripper > 0.5):
            grip_pos = int(new_val * 255)
            label = "CLOSE" if new_val > 0.5 else "OPEN"
            print(f"  Gripper -> {label} (pos={grip_pos})")
            gripper_hw.move(grip_pos, 255, 150)
            current_gripper = new_val

    t_start = time.monotonic()
    for i, pose_w in enumerate(interp):
        target_base = world_to_base(pose_w.tolist(), T_bw)
        rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)
        precise_wait(t_start + (i + 1) * servo_dt)
    rtde_c.servoStop()

    exec_ms = (time.monotonic() - t_start) * 1000
    return exec_ms, current_gripper


def check_grasp_done(pose_world, current_gripper, visited_near_table,
                     threshold=0.04):
    """Check early-stop: grasped + lifted high enough.

    Returns: (done, visited_near_table)
    """
    z_above = pose_world[2] - Z_MIN_WORLD
    if not visited_near_table and z_above < 0.05:
        visited_near_table = True
        print(f"  [INFO] Near table (z={pose_world[2]:.4f}), lift check armed")
    if visited_near_table and current_gripper > 0.5:
        if z_above >= threshold:
            print(f"\n[DONE] Lifted {z_above:.3f}m >= {threshold:.2f}m. Stopping.")
            return True, visited_near_table
    return False, visited_near_table


# ═════════════════════════════════════════════════════════════════════
#  Internal helpers for control loops
# ═════════════════════════════════════════════════════════════════════

def _read_state(rtde_r, T_bw, current_gripper):
    """Read current robot state. Returns (pose_base, pose_world, state_10d, current_pos)."""
    pose_base = rtde_r.getActualTCPPose()
    pose_world = base_to_world(pose_base, T_bw)
    state_10d = ee_pose_to_state10d(pose_world, current_gripper)
    current_pos = np.array(pose_world, dtype=np.float64)
    return pose_base, pose_world, state_10d, current_pos


def _print_step(step, wall_ms, obs_ms, infer_ms, send_ms,
                pose_world, current_gripper, actions, n_exec, suffix=""):
    deltas_3d = np.array([action_10d_to_delta7d(actions[i])[:3] for i in range(n_exec)])
    max_delta = np.abs(deltas_3d).max()
    total_disp = np.linalg.norm(deltas_3d.sum(axis=0))
    z_above = pose_world[2] - Z_MIN_WORLD
    hz = 1000.0 / wall_ms if wall_ms > 0 else 0
    print(f"[step {step:4d}]  wall={wall_ms:5.0f}ms ({hz:4.1f}Hz)  "
          f"obs={obs_ms:5.0f}ms  infer={infer_ms:5.0f}ms  "
          f"send={send_ms:5.0f}ms  "
          f"pos=[{pose_world[0]:.3f}, {pose_world[1]:.3f}, {pose_world[2]:.3f}]  "
          f"z={z_above:.3f}m  disp={total_disp:.4f}m  max_d={max_delta:.4f}m  "
          f"grip={'C' if current_gripper > 0.5 else 'O'}{suffix}")


def _build_step_data(step, wall_time, pose_base, pose_world, gripper,
                     state_10d, normalized, actions_10d, poses,
                     n_exec, waypoints, obs_ms, infer_ms, send_ms,
                     exec_ms, wall_ms, inference_delay=None):
    d = {
        "step": step, "wall_time": wall_time,
        "ee_pose_base": list(pose_base), "ee_pose_world": list(pose_world),
        "gripper_state": gripper, "state_10d": state_10d.tolist(),
        "pred_normalized": normalized.tolist(),
        "pred_actions_10d": actions_10d.tolist(),
        "pred_trajectory_world": poses.tolist(),
        "n_actions_executed": n_exec,
        "executed_targets_world": waypoints.tolist(),
        "obs_ms": round(obs_ms, 1), "infer_ms": round(infer_ms, 1),
        "send_ms": round(send_ms, 1), "exec_ms": round(exec_ms, 1),
        "wall_ms": round(wall_ms, 1),
        "viz_file": f"images/step_{step:04d}_viz.png",
    }
    if inference_delay is not None:
        d["inference_delay"] = inference_delay
    return d


def _save_rollout_step(saver, mode, step, wall_time, pose_base, pose_world,
                       current_gripper, state_10d, instruction,
                       obs_ms, infer_ms, send_ms, exec_ms, wall_ms, n_exec,
                       waypoints, pil_img,
                       # sync-only
                       pred_normalized=None, pred_actions=None,
                       # async/rtc-only
                       executing_normalized=None, current_actions=None,
                       action_stats=None, actual_delay=0):
    """Save viz + step data to rollout saver."""
    from scripts.log_utils import visualize_step

    if mode == "sync":
        pred_deltas = actions_10d_to_7d(pred_actions)
        pred_poses = accumulate_deltas(pose_world, pred_deltas)
        viz_path = str(saver.dir / "images" / f"step_{step:04d}_viz.png")
        saver.submit_viz(visualize_step, pil_img.copy(), list(pose_world),
                         n_exec, step, viz_path, instruction, "sync",
                         pred_poses=pred_poses.copy())
        saver.save_step(_build_step_data(
            step, wall_time, pose_base, pose_world, current_gripper,
            state_10d, pred_normalized, pred_actions, pred_poses,
            n_exec, waypoints, obs_ms, infer_ms, send_ms, exec_ms, wall_ms))
    else:
        exec_actions = baseframework.unnormalize_actions(executing_normalized, action_stats)
        exec_poses = accumulate_deltas(pose_world, actions_10d_to_7d(exec_actions))
        new_poses = accumulate_deltas(pose_world, actions_10d_to_7d(current_actions))
        viz_path = str(saver.dir / "images" / f"step_{step:04d}_viz.png")
        viz_mode = "rtc" if mode == "rtc" else "async"
        saver.submit_viz(visualize_step, pil_img.copy() if pil_img else None,
                         list(pose_world), n_exec, step, viz_path, instruction,
                         viz_mode, exec_poses=exec_poses.copy(),
                         new_poses=new_poses.copy(), inference_delay=actual_delay)
        saver.save_step(_build_step_data(
            step, wall_time, pose_base, pose_world, current_gripper,
            state_10d, executing_normalized, exec_actions, exec_poses,
            n_exec, waypoints, obs_ms, infer_ms, send_ms, exec_ms, wall_ms,
            inference_delay=actual_delay if mode == "rtc" else None))


# ═════════════════════════════════════════════════════════════════════
#  Sync control loop
# ═════════════════════════════════════════════════════════════════════

def run_sync(args, model, action_stats, cam, rtde_c, rtde_r, gripper_hw,
             T_bw, infer_kwargs, saver, timer):
    """Synchronous: observe -> infer -> execute, no overlap."""
    current_gripper = 0.0
    visited_near_table = False
    step = 0

    while args.max_steps == 0 or step < args.max_steps:
        loop_t0 = time.monotonic()
        wall_time = time.time()

        pose_base, pose_world, state_10d, current_pos = _read_state(rtde_r, T_bw, current_gripper)
        done, visited_near_table = check_grasp_done(
            pose_world, current_gripper, visited_near_table, args.grasp_lift_threshold)
        if args.stop_when_grasping and done:
            break

        t_obs = time.monotonic()
        pil_img = cam.grab_pil()
        if pil_img is None:
            print("[WARN] Camera frame dropped, retrying...")
            continue
        obs_ms = (time.monotonic() - t_obs) * 1000

        state_for_model = state_10d if args.include_state else None
        example = build_example(pil_img, args.instruction, state_10d=state_for_model)
        t_infer = time.monotonic()
        output = model.predict_action(examples=[example], **infer_kwargs)
        infer_ms = (time.monotonic() - t_infer) * 1000

        pred_normalized = output["normalized_actions"][0].astype(np.float32)
        pred_actions = baseframework.unnormalize_actions(pred_normalized, action_stats)
        n_exec = min(args.n_actions, len(pred_actions))

        t_send = time.monotonic()
        waypoints, grip_trans = compute_waypoints(current_pos, pred_actions, n_exec, args.fix_rotation)
        exec_ms, current_gripper = execute_servo(
            rtde_c, current_pos, waypoints, T_bw, gripper_hw, current_gripper, grip_trans)
        send_ms = (time.monotonic() - t_send) * 1000

        wall_ms = (time.monotonic() - loop_t0) * 1000
        timer.record(obs_ms, infer_ms, send_ms, wall_ms)
        _print_step(step, wall_ms, obs_ms, infer_ms, send_ms,
                    pose_world, current_gripper, pred_actions, n_exec)

        if saver:
            _save_rollout_step(saver, "sync", step, wall_time, pose_base, pose_world,
                               current_gripper, state_10d, args.instruction,
                               obs_ms, infer_ms, send_ms, exec_ms, wall_ms,
                               n_exec, waypoints, pil_img,
                               pred_normalized=pred_normalized, pred_actions=pred_actions)
        step += 1

    return step


# ═════════════════════════════════════════════════════════════════════
#  Async / RTC control loop (continuous servo, fixed-cadence inference)
# ═════════════════════════════════════════════════════════════════════

def run_async(args, model, action_stats, cam, rtde_c, rtde_r, gripper_hw,
              T_bw, infer_kwargs, saver, timer):
    """Async execution with continuous gap-free servo + fixed-cadence inference.

    Follows Algorithm 1 from the RTC paper:

      ServoRunner (GETACTION):
        - 100Hz daemon thread, consumes from buffer, increments action_t.
        - action_t counts 20Hz actions consumed (= paper's t).

      Inferencer (INFERENCELOOP):
        - Persistent worker thread, watches servo.action_t.
        - Waits until action_t >= chunk_start_t + n_actions (paper: t >= s_min).
        - Snapshots s = actions consumed, builds A_prev = A_cur[s:] (shifted prefix).
        - Runs camera capture + inference with prefix.
        - Posts result with actual_delay from the shift.

      Main loop:
        - Collects inference result.
        - Swaps in new chunk (paper line 20: A_cur = A_new).
        - Pushes free actions (skip prefix) into servo buffer.
        - Submits next job with the new chunk.

    Execution rhythm (chunk_len=16, n_actions=8, inference_delay=8):

      t=0 :  sync inference -> 16 actions.
             Push first n_actions into buffer, start servo + inferencer.
             Submit first job (inferencer waits for action_t=8).
      t=8 :  inferencer fires, runs inference.
             Main loop collects result, pushes 8 free actions, submits next.
      t=16:  repeat.
    """
    from scripts.servo import ServoRunner
    from scripts.inferencer import Inferencer

    use_rtc = (args.mode == "rtc")
    visited_near_table = False

    # ── Step 0: sync initial inference (no prefix) ────────────────────
    pose_base, pose_world, state_10d, current_pos = _read_state(rtde_r, T_bw, 0.0)
    pil_img = cam.grab_pil()
    while pil_img is None:
        pil_img = cam.grab_pil()

    state_for_model = state_10d if args.include_state else None
    example = build_example(pil_img, args.instruction, state_10d=state_for_model)
    t_infer = time.monotonic()
    output = model.predict_action(examples=[example], **infer_kwargs)
    print(f"[init] inference={((time.monotonic() - t_infer) * 1000):.0f}ms (sync, no prefix)")

    current_normalized = output["normalized_actions"][0].astype(np.float32)
    current_actions = baseframework.unnormalize_actions(current_normalized, action_stats)
    chunk_len = current_normalized.shape[0]
    n_actions = min(args.n_actions, chunk_len)

    # ── Push first n_actions into buffer (servo not started yet) ──────
    servo = ServoRunner(rtde_c, T_bw, gripper_hw)
    waypoints, grip_trans = compute_waypoints(
        current_pos, current_actions, n_actions, args.fix_rotation)
    servo.push_waypoints(current_pos, waypoints, grip_trans)

    # ── Start servo + inferencer ──────────────────────────────────────
    servo.start(current_pos)

    inferencer = Inferencer(
        model, cam, servo, n_actions, args.inference_delay,
        args.instruction, infer_kwargs, use_rtc=use_rtc)

    # Submit first job — inferencer watches servo.action_t, will wait
    # for n_actions consumed, then build prefix from current_normalized
    state_for_model = state_10d if args.include_state else None
    inferencer.submit(state_for_model, current_normalized, chunk_len)

    step = 0

    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()
            wall_time = time.time()

            # 1. Wait for inferencer result
            #    (inferencer internally waits for cadence boundary,
            #     builds prefix, runs camera + inference, posts result)
            result = inferencer.wait_result(timeout=60.0)

            # 2. Read robot state
            pose_base, pose_world, state_10d, current_pos = (
                _read_state(rtde_r, T_bw, servo.current_gripper))
            done, visited_near_table = check_grasp_done(
                pose_world, servo.current_gripper, visited_near_table,
                args.grasp_lift_threshold)
            if args.stop_when_grasping and done:
                break

            # 3. Process inference result
            executing_normalized = current_normalized.copy()
            if result is None:
                print("[ERROR] Inference timed out, reusing current chunk")
                obs_ms, infer_ms, pil_img = 0.0, 0.0, None
                actual_delay = 0
            else:
                obs_ms = result.obs_ms
                infer_ms = result.infer_ms
                pil_img = result.camera_image
                actual_delay = result.actual_delay
                # Paper line 20: A_cur = A_new (swap in new chunk)
                current_normalized = result.output["normalized_actions"][0].astype(np.float32)
                current_actions = baseframework.unnormalize_actions(
                    current_normalized, action_stats)

            # 4. Push free actions (skip prefix) into servo buffer
            actions_to_push = current_actions[actual_delay:]
            n_push = min(n_actions, len(actions_to_push))
            actions_to_push = actions_to_push[:n_push]
            start_pos = servo.last_pose if servo.last_pose is not None else current_pos
            waypoints, grip_trans = compute_waypoints(
                start_pos, actions_to_push, n_push, args.fix_rotation)
            servo.push_waypoints(start_pos, waypoints, grip_trans)

            # 5. Submit next inference job with the new chunk
            #    (inferencer will wait for next n_actions boundary,
            #     then build prefix from current_normalized internally)
            state_for_model = state_10d if args.include_state else None
            inferencer.submit(state_for_model, current_normalized, chunk_len)

            # 6. Logging
            wall_ms = (time.monotonic() - loop_t0) * 1000
            send_ms = 0.0
            timer.record(obs_ms, infer_ms, send_ms, wall_ms)

            current_action_t = servo.action_t
            buf_len = servo.buffer_len
            suffix = f"  buf={buf_len}  t={current_action_t}"
            if use_rtc:
                suffix += f"  delay={actual_delay}"
            if servo.starve_count > 0:
                suffix += f"  starved={servo.starve_count}"
            _print_step(step, wall_ms, obs_ms, infer_ms, send_ms,
                        pose_world, servo.current_gripper,
                        actions_to_push, n_push, suffix)

            if saver:
                _save_rollout_step(saver, args.mode, step, wall_time, pose_base, pose_world,
                                   servo.current_gripper, state_10d, args.instruction,
                                   obs_ms, infer_ms, send_ms, 0.0, wall_ms,
                                   n_push, waypoints, pil_img,
                                   executing_normalized=executing_normalized,
                                   current_actions=current_actions,
                                   action_stats=action_stats, actual_delay=actual_delay)
            step += 1

    finally:
        inferencer.stop()
        servo.stop()

    return step


# ═════════════════════════════════════════════════════════════════════
#  Main harness (shared setup + cleanup)
# ═════════════════════════════════════════════════════════════════════

def run_main(args, model, action_stats, dataset_key,
             infer_kwargs, splash_title, method_name,
             extra_config=None):
    """Shared entry point: hardware setup -> loop -> cleanup.

    Called by both run_dd.py and run_fm.py after parsing args and loading model.
    """
    from scripts.log_utils import TimingTracker, StopwatchServer, RolloutSaver

    stopwatch = StopwatchServer(port=args.stopwatch_port)
    sw_url = stopwatch.serve(open_browser=True)
    T_bw = BASE_IN_WORLD[args.arm]

    rtde_c, rtde_r, robot_ip = connect_robot(args.arm)
    gripper_hw = connect_gripper(robot_ip)

    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    print("Camera ready.")

    if not args.no_go_home:
        go_home(rtde_c, rtde_r, args.arm, T_bw, robot_ip)

    # Print config
    print(f"\n{'=' * 60}")
    print(f"  Method:             {method_name} {args.mode}")
    print(f"  Arm:                {args.arm}")
    print(f"  Instruction:        {args.instruction}")
    print(f"  Actions/step:       {args.n_actions}")
    if args.mode == "rtc":
        print(f"  Inference delay:    {args.inference_delay}")
    if args.mode in ("async", "rtc"):
        print(f"  Servo:              {SERVO_HZ}Hz continuous (gap-free)")
    else:
        print(f"  Servo:              {CONTROL_HZ}Hz x {INTERP_MULT} = {SERVO_HZ}Hz")
    print(f"  Z safety:           {Z_MIN_WORLD:.4f}m")
    print(f"  Include state:      {args.include_state}")
    for k, v in (extra_config or {}).items():
        print(f"  {k + ':':20s}{v}")
    print(f"  fix_rotation:       {args.fix_rotation}")
    print(f"  stop_when_grasping: {args.stop_when_grasping} (>={args.grasp_lift_threshold:.2f}m)")
    print(f"  stopwatch:          {sw_url}")
    print(f"{'=' * 60}")

    # Rollout saver
    saver = None
    if args.save_rollout:
        ts = time.strftime("%Y%m%d_%H%M%S")
        prefix = method_name.lower().replace(" ", "_")
        rollout_dir = Path(args.rollout_dir) if args.rollout_dir else (
            Path("ur5") / "rollouts" / f"{prefix}_{args.mode}_{ts}")
        run_config = {
            "mode": args.mode, "method": method_name,
            "checkpoint": args.checkpoint, "arm": args.arm,
            "instruction": args.instruction, "n_actions": args.n_actions,
            "inference_delay": args.inference_delay if args.mode == "rtc" else None,
            "max_steps": args.max_steps, "include_state": args.include_state,
            "control_hz": CONTROL_HZ, "servo_hz": SERVO_HZ,
            "interp_mult": INTERP_MULT, "dataset_key": dataset_key,
            **(extra_config or {}),
        }
        saver = RolloutSaver(rollout_dir, run_config)

    timer = TimingTracker()
    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    stopwatch.start_timer(splash_title=splash_title)
    print("[STOPWATCH] Started")

    step = 0
    loop_fn = run_sync if args.mode == "sync" else run_async
    try:
        step = loop_fn(args, model, action_stats, cam,
                       rtde_c, rtde_r, gripper_hw, T_bw,
                       infer_kwargs, saver, timer)
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        stopwatch.stop_timer()
        print("[STOPWATCH] Stopped")
        print("Cleaning up...")
        try: rtde_c.servoStop()
        except Exception: pass
        try: rtde_c.stopScript()
        except Exception: pass
        try: gripper_hw.disconnect()
        except Exception: pass
        cam.close()
        if saver:
            saver.finalize()
        timer.summary()
        print(f"Done. Executed {step} inference steps.")
