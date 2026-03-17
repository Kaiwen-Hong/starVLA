#!/usr/bin/env python3
# ============ DATA PATH CONFIG (edit here) ============
DATA_BASE = "/home/kaiwen/Desktop/research/fastumipro-collection/data_collector_opt/DATA"
DATA_SESSION = "left_hand_250801DR48FP25002960/pickandplace-314/session_20260314_172507"
DATA_PATH = f"{DATA_BASE}/{DATA_SESSION}"
# ======================================================
"""
Dataset Replay Script for UR5e Robot

Replays recorded demonstrations from zarr datasets on real UR5e robot.
Supports both joint and end-effector control modes with raw and processed data formats.
"""

import os
import sys
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

import subprocess

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as Rot
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
from rtde_io import RTDEIOInterface
import modular_policy

# Resolve fastumipro-collection root from the installed modular_policy package
project_root = Path(modular_policy.__file__).parent.parent

# Load robot extrinsics (base pose in world frame)
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Information about a loaded dataset."""
    data: Dict[str, np.ndarray]
    episode_ends: np.ndarray
    format_type: str  # 'raw' or 'processed'
    available_keys: List[str]


@dataclass
class ActionData:
    """Container for extracted action and gripper data."""
    actions: np.ndarray
    gripper_actions: Optional[np.ndarray] = None


class DatasetLoader:
    """Handles loading and parsing of zarr datasets."""
    
    @staticmethod
    def load(zarr_path: str) -> DatasetInfo:
        """Load zarr dataset and return structured information."""
        logger.info(f"Loading dataset from: {zarr_path}")
        
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Dataset not found at {zarr_path}")
        
        try:
            group = zarr.open_group(zarr_path, 'r')
            data_group = group['data']
            episode_ends = group['meta']['episode_ends'][:]
        except Exception as e:
            raise ValueError(f"Failed to load dataset: {e}")
        
        # Detect and normalize data format
        if 'observations' in data_group:
            logger.info("Detected raw data format")
            data_dict = DatasetLoader._normalize_raw_data(data_group)
            format_type = 'raw'
        else:
            logger.info("Detected processed data format")
            data_dict = dict(data_group)
            format_type = 'processed'
        
        available_keys = [k for k in data_dict.keys() if not k.startswith('camera')]
        logger.info(f"Dataset loaded - Episodes: {len(episode_ends)}, Keys: {available_keys}")
        
        return DatasetInfo(
            data=data_dict,
            episode_ends=episode_ends,
            format_type=format_type,
            available_keys=available_keys
        )
    
    @staticmethod
    def _normalize_raw_data(data_group) -> Dict[str, np.ndarray]:
        """Normalize raw data format to processed format structure."""
        normalized = {}
        
        # Copy observation data to top level
        if 'observations' in data_group:
            for key, value in data_group['observations'].items():
                normalized[key] = value
        
        # Copy action and other data
        for key, value in data_group.items():
            if key != 'observations':
                normalized[key] = value
                
        return normalized


class SessionLoader:
    """Load EEF poses directly from a raw session directory (no zarr needed).

    Automatically runs slam_analyze_and_vel.py if the downsampled pose file
    doesn't exist yet, then loads the resulting txt file.
    """

    SLAM_SCRIPT = os.path.join(project_root, 'data_collector_opt', 'slam_analyze_and_vel.py')
    POSE_FILENAME = 'slam_raw_pose_worldframe_downsampled.txt'

    @staticmethod
    def is_session_dir(path: str) -> bool:
        """Check if path is a raw session directory (contains SLAM_Poses/)."""
        return os.path.isdir(path) and os.path.isdir(os.path.join(path, 'SLAM_Poses'))

    @staticmethod
    def load(session_dir: str) -> ActionData:
        """Load EEF actions from a session directory.

        1. Find or generate the downsampled worldframe pose file
        2. Parse it into ActionData (eef poses + gripper)
        """
        slam_dir = os.path.join(session_dir, 'SLAM_Poses')
        pose_file = os.path.join(slam_dir, SessionLoader.POSE_FILENAME)

        # Run slam processing if downsampled file doesn't exist
        if not os.path.isfile(pose_file):
            SessionLoader._run_slam_processing(session_dir)

        if not os.path.isfile(pose_file):
            raise FileNotFoundError(
                f"Pose file not found after processing: {pose_file}\n"
                f"Check slam_analyze_and_vel.py output.")

        return SessionLoader._load_pose_txt(pose_file)

    @staticmethod
    def _run_slam_processing(session_dir: str):
        """Run slam_analyze_and_vel.py on the session's slam_raw.txt."""
        slam_raw = os.path.join(session_dir, 'SLAM_Poses', 'slam_raw.txt')
        if not os.path.isfile(slam_raw):
            raise FileNotFoundError(f"slam_raw.txt not found: {slam_raw}")

        logger.info(f"Running SLAM processing on: {slam_raw}")
        result = subprocess.run(
            [sys.executable, SessionLoader.SLAM_SCRIPT, '-i', slam_raw],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            logger.error(f"SLAM processing failed:\n{result.stderr}")
            raise RuntimeError("slam_analyze_and_vel.py failed")
        logger.info("SLAM processing completed.")

    # SLAM device → UR5 EEF rotation mapping (validated against calibration data)
    # eef_x = slam_y, eef_y = slam_z, eef_z = slam_x
    T_SLAM_GRIPPER = np.array([
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ])

    @staticmethod
    def _slam_to_gripper_rotation(rotvecs: np.ndarray) -> np.ndarray:
        """Convert SLAM device axis-angle to gripper axis-angle.

        The worldframe file has SLAM device orientation. The robot EEF needs
        gripper orientation. Correction: R_eef = R_slam @ T^(-1)
        """
        T_inv = SessionLoader.T_SLAM_GRIPPER.T  # orthogonal, so T^(-1) = T^T
        corrected = np.zeros_like(rotvecs)
        for i in range(len(rotvecs)):
            R_slam = Rot.from_rotvec(rotvecs[i]).as_matrix()
            R_eef = R_slam @ T_inv
            corrected[i] = Rot.from_matrix(R_eef).as_rotvec()
        return corrected

    @staticmethod
    def _load_pose_txt(pose_file: str) -> ActionData:
        """Parse downsampled pose txt: time x y z rx ry rz clamp_open."""
        logger.info(f"Loading poses from: {pose_file}")
        rows = []
        with open(pose_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) < 8:
                    continue
                # skip timestamp (col 0), take x y z rx ry rz clamp_open (cols 1-7)
                rows.append([float(v) for v in parts[1:8]])

        if not rows:
            raise ValueError(f"No valid pose data in {pose_file}")

        data = np.array(rows, dtype=np.float64)  # (N, 7): x y z rx ry rz clamp
        actions = data[:, :6]       # (N, 6): x y z rx ry rz
        gripper = data[:, 6:7]      # (N, 1): clamp_open

        # Correct rotation: SLAM device orientation → gripper orientation
        actions[:, 3:6] = SessionLoader._slam_to_gripper_rotation(actions[:, 3:6])
        logger.info(f"Applied SLAM→Gripper rotation correction")

        # Convert clamp_open → Robotiq: binary open/close
        # clamp_open < 0.8 means grasping → full close (1.0), else full open (0.0)
        GRASP_THRESHOLD = 0.8
        gripper = np.where(gripper < GRASP_THRESHOLD, 1.0, 0.0)
        logger.info(f"Gripper: {int((gripper > 0.5).sum())}/{len(gripper)} steps closed")

        logger.info(f"Loaded {len(actions)} poses from txt (EEF mode)")
        return ActionData(actions=actions, gripper_actions=gripper)


class ActionExtractor:
    """Handles extraction of actions and gripper data from datasets."""
    
    @staticmethod
    def extract(dataset_info: DatasetInfo, episode_slice: slice, 
                ctrl_mode: str, joint_source: str = 'robot_joint') -> ActionData:
        """Extract actions based on control mode."""
        
        if ctrl_mode == 'joint':
            return ActionExtractor._extract_joint_actions(
                dataset_info, episode_slice, joint_source)
        else:  # eef mode
            return ActionExtractor._extract_eef_actions(
                dataset_info, episode_slice)
    
    @staticmethod
    def _extract_joint_actions(dataset_info: DatasetInfo, episode_slice: slice,
                              joint_source: str) -> ActionData:
        """Extract joint-based actions."""
        data = dataset_info.data
        
        # Validate joint source availability
        if joint_source not in data:
            available = [k for k in dataset_info.available_keys if 'joint' in k]
            raise KeyError(f"'{joint_source}' not found. Available joint keys: {available}")
        
        actions = data[joint_source][episode_slice]
        logger.info(f"Using {joint_source} - Shape: {actions.shape}")
        
        # Validate joint ranges for safety (UR5e joint limits approximately ±2π)
        if joint_source == 'robot_joint':
            joint_data_6d = actions[:, :6] if actions.shape[1] > 6 else actions
            min_vals = np.min(joint_data_6d, axis=0)
            max_vals = np.max(joint_data_6d, axis=0)
            
            # Check for reasonable joint ranges
            if np.any(np.abs(min_vals) > 3.5) or np.any(np.abs(max_vals) > 3.5):
                logger.warning(f"Joint positions may be outside normal range:")
                logger.warning(f"  Min values: {min_vals}")
                logger.warning(f"  Max values: {max_vals}")
                logger.warning("  Proceed with caution!")
        
        # Extract gripper data
        gripper_actions = ActionExtractor._extract_gripper_from_joints(
            data, episode_slice, joint_source, actions)
        
        # Handle joint dimensions
        if actions.shape[1] > 6:
            actions = actions[:, :6]
            logger.info(f"Extracted 6D joint positions - Shape: {actions.shape}")
        
        # Validate joint_action data quality and handle problematic data
        if joint_source == 'joint_action':
            if np.allclose(actions, 0):
                logger.error("joint_action data is all zeros - this would send robot to dangerous zero position!")
                logger.error("This dataset was likely collected with EEF control, not joint control.")
                logger.error("Automatically switching to robot_joint for safety.")
                logger.info("To avoid this, use: -j robot_joint or switch to: -m eef")
                
                # Automatically fall back to robot_joint for safety
                if 'robot_joint' in data:
                    actions = data['robot_joint'][episode_slice]
                    logger.info(f"Using robot_joint fallback - Shape: {actions.shape}")
                    
                    # Re-extract gripper data from robot_joint
                    gripper_actions = ActionExtractor._extract_gripper_from_joints(
                        data, episode_slice, 'robot_joint', actions)
                    
                    # Handle joint dimensions for robot_joint
                    if actions.shape[1] > 6:
                        actions = actions[:, :6]
                        logger.info(f"Extracted 6D joint positions - Shape: {actions.shape}")
                else:
                    raise ValueError("joint_action is all zeros and robot_joint not available. Use EEF mode instead.")
            else:
                # Check if joint_action contains reasonable values
                if np.abs(np.mean(actions)) < 1e-6 and np.std(actions) < 1e-6:
                    logger.warning("joint_action data has very small values - verify this is intended")
        
        return ActionData(actions=actions, gripper_actions=gripper_actions)
    
    @staticmethod
    def _extract_eef_actions(dataset_info: DatasetInfo, episode_slice: slice) -> ActionData:
        """Extract end-effector based actions."""
        data = dataset_info.data
        
        if 'robot_eef_pose' not in data:
            available = [k for k in dataset_info.available_keys if 'eef' in k or 'pose' in k]
            raise KeyError(f"'robot_eef_pose' not found. Available pose keys: {available}")
        
        actions = data['robot_eef_pose'][episode_slice]
        logger.info(f"Using EEF poses - Shape: {actions.shape}")
        logger.info(f"Rotation format: axis-angle, shape: {actions[:,3:6].shape}")
        
        # Extract gripper from EEF data
        gripper_actions = None
        if actions.shape[1] >= 7:
            gripper_actions = actions[:, 6:7]
            logger.info(f"EEF poses include gripper - Shape: {gripper_actions.shape}")
        
        return ActionData(actions=actions, gripper_actions=gripper_actions)
    
    @staticmethod
    def _extract_gripper_from_joints(data: Dict[str, np.ndarray], episode_slice: slice,
                                   joint_source: str, actions: np.ndarray) -> Optional[np.ndarray]:
        """Extract gripper data from joint information."""
        # Try to get gripper from current joint source
        if actions.shape[1] > 6:
            gripper = actions[:, 6:7]
            logger.info(f"Extracted gripper from {joint_source} - Shape: {gripper.shape}")
            return gripper
        
        # Fallback: try to get gripper from robot_joint if using joint_action
        if joint_source == 'joint_action' and 'robot_joint' in data:
            robot_joint_data = data['robot_joint'][episode_slice]
            if robot_joint_data.shape[1] > 6:
                gripper = robot_joint_data[:, 6:7]
                logger.info(f"Using gripper from robot_joint - Shape: {gripper.shape}")
                return gripper
        
        return None


class MovementFilter:
    """Handles filtering of non-moving segments from action sequences."""
    
    @staticmethod
    def filter_actions(dataset_info: DatasetInfo, episode_slice: slice,
                      action_data: ActionData, ctrl_mode: str,
                      movement_threshold: float = 0.001) -> ActionData:
        """Filter out non-moving segments for smoother replay."""
        
        if ctrl_mode == 'joint':
            return MovementFilter._filter_joint_movements(
                dataset_info, episode_slice, action_data)
        else:
            return MovementFilter._filter_pose_movements(
                action_data, movement_threshold)
    
    @staticmethod
    def _filter_joint_movements(dataset_info: DatasetInfo, episode_slice: slice,
                               action_data: ActionData) -> ActionData:
        """Filter joint movements based on joint position differences."""
        data = dataset_info.data
        
        try:
            obs_joints = data['robot_joint'][episode_slice]
            joint_actions = data['joint_action'][episode_slice]
            
            # Compare only first 6 dimensions
            obs_joints_6d = obs_joints[:, :6] if obs_joints.shape[1] > 6 else obs_joints
            joint_actions_6d = joint_actions[:, :6] if joint_actions.shape[1] > 6 else joint_actions
            
            # Inline are_joints_close: normalize to [0,2π], check euclidean dist
            norm1 = np.mod(obs_joints_6d, 2 * np.pi)
            norm2 = np.mod(joint_actions_6d, 2 * np.pi)
            distances = np.linalg.norm(norm1 - norm2, axis=1)
            not_moving_mask = distances <= np.pi / 180
            moving_mask = ~not_moving_mask
        except KeyError as e:
            logger.warning(f"Could not filter joint movements: {e}")
            moving_mask = np.ones(len(action_data.actions), dtype=bool)
        
        return ActionData(
            actions=action_data.actions[moving_mask],
            gripper_actions=action_data.gripper_actions[moving_mask] 
                           if action_data.gripper_actions is not None else None
        )
    
    @staticmethod
    def _filter_pose_movements(action_data: ActionData, 
                              movement_threshold: float) -> ActionData:
        """Filter pose movements based on position differences."""
        actions = action_data.actions
        pose_diff = np.diff(actions[:, :3], axis=0)
        pose_movement = np.linalg.norm(pose_diff, axis=1)
        
        moving_indices = np.where(pose_movement > movement_threshold)[0] + 1
        moving_mask = np.zeros(len(actions), dtype=bool)
        moving_mask[0] = True  # Always include first pose
        moving_mask[moving_indices] = True
        
        return ActionData(
            actions=actions[moving_mask],
            gripper_actions=action_data.gripper_actions[moving_mask] 
                           if action_data.gripper_actions is not None else None
        )


# Home poses (EEF) in WORLD frame: [x, y, z, rx, ry, rz]
HOME_POSE_WORLD = {
    'left': [0.20, -0.2, 0.2, -2.2192, 2.2148, 0.0091],
}


def _precise_wait(t_end: float, slack_time: float = 0.001):
    """Hybrid sleep + spin wait for precise timing."""
    t_wait = t_end - time.monotonic()
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.monotonic() < t_end:
            pass


class RobotReplayer:
    """Handles robot control and action execution via direct RTDE."""

    def __init__(self, robot_ip: str, speed_slider: float, arm: str):
        self.robot_ip = robot_ip
        self.speed_slider = speed_slider
        self.arm = arm

    def _go_home(self, rtde_c: RTDEControlInterface, arm: str):
        """Move robot to home pose before replay (world frame -> base frame)."""
        if arm not in HOME_POSE_WORLD:
            logger.warning(f"No home pose defined for arm '{arm}', skipping go-home.")
            return
        pose_world = HOME_POSE_WORLD[arm]
        T_bw = BASE_IN_WORLD[arm]
        pose_base = [
            pose_world[0] - T_bw[0, 3],
            pose_world[1] - T_bw[1, 3],
            pose_world[2] - T_bw[2, 3],
        ] + pose_world[3:]
        logger.info(f"Moving to home pose (world): {pose_world}")
        logger.info(f"Home pose (base): {pose_base}")
        rtde_c.moveL(pose_base, 0.25, 0.5)
        logger.info("Reached home pose.")

    def replay_episode(self, action_data: ActionData, ctrl_mode: str,
                      frequency: int, batch_size: int):
        """Execute action sequence on robot using direct RTDE."""

        logger.info(f"Connecting to robot at {self.robot_ip}...")
        rtde_c = RTDEControlInterface(self.robot_ip)
        rtde_r = RTDEReceiveInterface(self.robot_ip)
        rtde_io = RTDEIOInterface(self.robot_ip)

        rtde_io.setSpeedSlider(self.speed_slider)
        logger.info(f"Speed slider set to {self.speed_slider}")

        current_pose = rtde_r.getActualTCPPose()
        logger.info(f"Current TCP pose: {np.round(current_pose, 4).tolist()}")

        try:
            # Go to home pose first
            self._go_home(rtde_c, self.arm)

            actions = action_data.actions
            dt = 1.0 / frequency
            n_actions = len(actions)
            logger.info(f"Replaying {n_actions} actions at {frequency} Hz ({ctrl_mode} mode)...")
            logger.info("Press Ctrl+C to stop replay")

            t_start = time.monotonic()

            for i in range(n_actions):
                if ctrl_mode == 'joint':
                    pose = actions[i, :6].tolist()
                    rtde_c.servoJ(pose, 0, 0, dt, 0.1, 300)
                else:  # eef
                    pose = actions[i, :6].tolist()
                    rtde_c.servoL(pose, 0, 0, dt, 0.1, 300)

                if i % 50 == 0:
                    logger.info(f"Step {i}/{n_actions}")

                _precise_wait(t_start + (i + 1) * dt)

            if ctrl_mode == 'joint':
                rtde_c.servoStop()
            else:
                rtde_c.servoStop()

            logger.info("Replay completed successfully!")

        except KeyboardInterrupt:
            logger.info("Replay interrupted by user")
            rtde_c.servoStop()
        except Exception as e:
            logger.error(f"Error during replay: {e}")
            rtde_c.servoStop()
            raise
        finally:
            rtde_c.stopScript()


def get_episode_slice(episode_num: int, episode_ends: np.ndarray) -> slice:
    """Get slice for specific episode."""
    if episode_num >= len(episode_ends):
        raise ValueError(f"Episode {episode_num} not found. Available: 0-{len(episode_ends)-1}")
    
    episode_start = 0 if episode_num == 0 else episode_ends[episode_num-1]
    episode_end = episode_ends[episode_num]
    
    logger.info(f"Replaying episode {episode_num}: timesteps {episode_start} to {episode_end-1}")
    return slice(episode_start, episode_end)


def get_robot_ip(args) -> str:
    """Get robot IP based on selected arm."""
    return args.robot_left_ip if args.arm == 'left' else args.robot_right_ip


def world_to_base_actions(action_data: ActionData, arm: str) -> ActionData:
    """Convert EEF actions from world frame to robot base frame.
    Since R_bw = I, only position (xyz) is offset; rotation stays the same."""
    T_bw = BASE_IN_WORLD[arm]
    actions = action_data.actions.copy()
    # Subtract base-in-world translation from xyz
    actions[:, 0] -= T_bw[0, 3]
    actions[:, 1] -= T_bw[1, 3]
    actions[:, 2] -= T_bw[2, 3]
    logger.info(f"Converted EEF poses: world frame -> {arm} base frame (offset {T_bw[:3,3]})")
    return ActionData(actions=actions, gripper_actions=action_data.gripper_actions)


def replay_episode_pipeline(args):
    """Main pipeline for episode replay."""

    # Detect input type: session directory (raw) vs zarr
    if SessionLoader.is_session_dir(args.dataset):
        logger.info(f"Detected raw session directory: {args.dataset}")
        # Force EEF mode for raw session data
        if args.mode != 'eef':
            logger.warning("Raw session data only supports EEF mode, switching to eef.")
            args.mode = 'eef'
        action_data = SessionLoader.load(args.dataset)
    else:
        # Original zarr path
        dataset_info = DatasetLoader.load(args.dataset)
        episode_slice = get_episode_slice(args.episode, dataset_info.episode_ends)
        action_data = ActionExtractor.extract(
            dataset_info, episode_slice, args.mode, args.joint_source)

        if 'left_tactile' in dataset_info.data and 'right_tactile' in dataset_info.data:
            left_tactile = dataset_info.data['left_tactile'][episode_slice]
            right_tactile = dataset_info.data['right_tactile'][episode_slice]
            logger.info(f"Tactile data found - Left: {left_tactile.shape}, Right: {right_tactile.shape}")

    # Convert world frame -> robot base frame for EEF mode
    if args.mode == 'eef':
        action_data = world_to_base_actions(action_data, args.arm)

    # Create and run replayer
    robot_ip = get_robot_ip(args)
    replayer = RobotReplayer(robot_ip, args.speed, args.arm)
    replayer.replay_episode(action_data, args.mode, args.frequency, args.batch_size)


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Replay dataset on UR5e robot (supports raw session dirs and zarr)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # use DATA_PATH from top of file
  %(prog)s /path/to/session_20260314_172507          # raw session dir (auto SLAM processing)
  %(prog)s data/puzzle_expert_30.zarr                # zarr dataset
  %(prog)s data/puzzle_expert_30.zarr -m joint       # zarr with joint mode
        """
    )

    # Required arguments
    parser.add_argument('dataset', nargs='?', default=DATA_PATH,
                       help='Path to session directory or zarr dataset (default: DATA_PATH)')
    
    # Control arguments
    parser.add_argument('-e', '--episode', type=int, default=0,
                       help='Episode number to replay (default: %(default)s)')
    parser.add_argument('-m', '--mode', choices=['joint', 'eef'], default='eef',
                       help='Control mode (default: %(default)s)')
    parser.add_argument('-j', '--joint-source', choices=['robot_joint', 'joint_action'], 
                       default='robot_joint',
                       help='Joint data source (default: %(default)s)')
    parser.add_argument('-s', '--speed', type=float, default=0.1,
                       help='Speed scaling factor 0.0-1.0 (default: %(default)s)')
    
    # Execution parameters
    parser.add_argument('-f', '--frequency', type=int, default=3,
                       help='Control frequency in Hz (default: %(default)s)')
    parser.add_argument('-b', '--batch-size', type=int, default=3,
                       help='Actions per batch (default: %(default)s)')
    parser.add_argument('--no-filter', action='store_true', default=True,
                       help='Disable movement filtering')
    
    # Robot configuration
    parser.add_argument('--robot-left-ip', default='192.168.0.3',
                       help='Left robot IP (default: %(default)s)')
    parser.add_argument('--robot-right-ip', default='192.168.0.2',
                       help='Right robot IP (default: %(default)s)')
    parser.add_argument('--arm', choices=['left', 'right'], default='left',
                       help='Arm for single-arm control (default: %(default)s)')
    parser.add_argument('--no-gripper', action='store_true',
                       help='Disable gripper control')
    
    return parser


def print_configuration(args):
    """Print configuration summary."""
    print("=== Dataset Replay Configuration ===")
    print(f"Dataset: {args.dataset}")
    print(f"Episode: {args.episode}")
    print(f"Control mode: {args.mode}")
    if args.mode == 'joint':
        print(f"Joint source: {args.joint_source}")
    print(f"Speed: {args.speed}")
    print(f"Frequency: {args.frequency} Hz")
    print(f"Batch size: {args.batch_size}")
    print(f"Filter movement: {not args.no_filter}")
    print(f"Arm: {args.arm}")
    print(f"Gripper: {not args.no_gripper}")
    print("=" * 40)


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate dataset path
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset not found at {args.dataset}")
        sys.exit(1)
    
    print_configuration(args)
    
    # Run replay pipeline
    try:
        replay_episode_pipeline(args)
    except Exception as e:
        logger.error(f"Replay failed: {e}")
        raise


if __name__ == '__main__':
    main()