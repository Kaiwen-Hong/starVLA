# FastUMI 原始数据 → LeRobot/StarVLA v2.1 格式转换详解

本文档详细记录 `V3-convert_to_starvla_cropped-augment-mp.py` 的完整处理流程。
所有行号引用均来自该脚本（以下简称 V3-mp）及 `utils/pose_utils.py`。

> 源码位置: `fastumipro-collection/0srarvla-lerobo/V3-convert_to_starvla_cropped-augment-mp.py`

---

## 0. 全局常量配置

**V3-mp 第 54-65 行:**

```python
FPS = 20
IMAGE_SIZE_DEFAULT = (256, 256)
ROBOT_TYPE = "fastumi"

TRAJECTORY_FILE_BASE = "slam_raw_pose_worldframe_downsampled"
TIMESTAMPS_FILE = "timestamps.csv"
FRAMES_DIR = "Frames"

DEFAULT_THRESHOLD_MM = 2.0
N_OBS_STEPS = 2
HORIZON = 16
MIN_EPISODE_LENGTH = N_OBS_STEPS + HORIZON - 1  # = 17
```

- `FPS = 20`: 输出数据集的帧率
- `MIN_EPISODE_LENGTH = 17`: 裁剪后低于此长度的 episode 会被丢弃（因为训练需要 2 步观测 + 16 步 action horizon）

---

## 1. 加载轨迹文件

**V3-mp 第 148-177 行 `load_trajectory()`:**

```python
def load_trajectory(filepath):
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or len(line) == 0:  # 跳过注释和空行
                continue
            parts = line.split()
            if len(parts) < 8:  # 必须至少 8 列
                continue
            values = [float(p) for p in parts[:8]]  # 取前 8 列
            data.append(values)

    data = np.array(data)  # shape: (N, 8)
    return {
        'time':      data[:, 0],  # (N,) 时间戳 (秒)
        'x':         data[:, 1],  # (N,) 世界坐标 X (米)
        'y':         data[:, 2],  # (N,) 世界坐标 Y (米)
        'z':         data[:, 3],  # (N,) 世界坐标 Z (米)
        'roll_deg':  data[:, 4],  # (N,) 绕X轴旋转 (度)
        'pitch_deg': data[:, 5],  # (N,) 绕Y轴旋转 (度)
        'yaw_deg':   data[:, 6],  # (N,) 绕Z轴旋转 (度)
        'gripper':   data[:, 7],  # (N,) 夹爪状态 (0~1, 原始语义: 0=闭合, 1=张开)
    }
```

### 输入文件格式

- 路径: `{session_dir}/SLAM_Poses/slam_raw_pose_worldframe_downsampled-v{version}.txt`
- 文件名由 **V3-mp 第 68-71 行** `get_trajectory_filename(version)` 生成
- 每行: `timestamp x y z roll_deg pitch_deg yaw_deg gripper`，空格分隔
- `#` 开头的行被跳过

### 数据形状变化

```
原始 txt 文件 (每行 8 个 float)
    ↓ np.array(data)
(N, 8) float64
    ↓ 拆分为字典
time: (N,), x: (N,), y: (N,), z: (N,), roll_deg: (N,), pitch_deg: (N,), yaw_deg: (N,), gripper: (N,)
```

---

## 2. 轨迹转为 10D 状态表示

**V3-mp 第 215-232 行 `process_trajectory_10d()`:**

```python
def process_trajectory_10d(traj):
    n_frames = len(traj['time'])
    processed = np.zeros((n_frames, 10), dtype=np.float32)  # 输出: (N, 10)

    for i in range(n_frames):
        # [0:3] 位置，直接拷贝
        processed[i, 0] = traj['x'][i]
        processed[i, 1] = traj['y'][i]
        processed[i, 2] = traj['z'][i]

        # [3:9] 旋转，欧拉角(度) → 弧度 → rot6d(6维)
        roll_rad  = np.deg2rad(traj['roll_deg'][i])
        pitch_rad = np.deg2rad(traj['pitch_deg'][i])
        yaw_rad   = np.deg2rad(traj['yaw_deg'][i])
        rot6d = euler_to_rot6d(roll_rad, pitch_rad, yaw_rad)  # (6,)
        processed[i, 3:9] = rot6d

        # [9] 夹爪，翻转语义: 1.0 - gripper (原始0=闭→转换后1=闭)
        g = traj['gripper'][i]
        processed[i, 9] = 0.0 if np.isnan(g) else 1.0 - g

    return processed  # (N, 10) float32
```

### `euler_to_rot6d` 内部调用链

**pose_utils.py 第 166-177 行:**

```python
def euler_to_rot6d(roll, pitch, yaw):
    R = euler_to_rotation_matrix(roll, pitch, yaw)  # (3, 3)
    return mat_to_rot6d(R)                           # (6,)
```

**pose_utils.py 第 128-143 行** `euler_to_rotation_matrix`:

```python
def euler_to_rotation_matrix(roll, pitch, yaw):
    # 使用 scipy，约定: 'xyz' 内旋 (intrinsic), 即 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    return Rotation.from_euler('xyz', [roll, pitch, yaw]).as_matrix()  # (3, 3)
```

**pose_utils.py 第 50-77 行** `mat_to_rot6d`:

```python
def mat_to_rot6d(mat):
    # 取旋转矩阵的前两行，展平为 6 维向量
    # mat shape: (3, 3) → 取 mat[:2, :] → (2, 3) → flatten → (6,)
    # 结果: [r00, r01, r02, r10, r11, r12]
    return mat[:2, :].flatten()  # (6,)
```

### 数据形状变化

```
traj 字典 (各字段 shape (N,))
    ↓ process_trajectory_10d()
(N, 10) float32

每帧 10 维的含义:
  [0]   x        位置X (米)
  [1]   y        位置Y (米)
  [2]   z        位置Z (米)
  [3]   r00      旋转矩阵第1行第1列
  [4]   r01      旋转矩阵第1行第2列
  [5]   r02      旋转矩阵第1行第3列
  [6]   r10      旋转矩阵第2行第1列
  [7]   r11      旋转矩阵第2行第2列
  [8]   r12      旋转矩阵第2行第3列
  [9]   gripper  夹爪 (0=张开, 1=闭合, 已翻转)
```

### 为什么用 rot6d?

rot6d 是旋转矩阵前两行的展平，第三行可以通过叉积恢复。相比欧拉角:
- **无万向锁** (Gimbal Lock): 欧拉角在 pitch ≈ ±90° 时退化
- **连续表示**: 神经网络可以更稳定地学习旋转
- 参考论文: Zhou et al., 2019, "On the Continuity of Rotation Representations in Neural Networks"

---

## 3. 空闲帧裁剪 (Idle Frame Cropping)

**V3-mp 第 78-141 行 `detect_crop_range()`**

目的: 去掉轨迹头部（机器人还没开始运动）和尾部（任务已完成但还在录制）的无效帧。

### 3a. 计算逐帧位移

**V3-mp 第 78-80 行:**

```python
def compute_movement_per_frame(positions):
    pos_changes = np.diff(positions, axis=0)          # (N, 3) → (N-1, 3)
    return np.sqrt(np.sum(pos_changes**2, axis=1))    # (N-1,) 每帧的欧氏距离 (米)
```

### 3b. 裁剪起点 (crop_start)

**V3-mp 第 88-95 行:**

```python
pos_movement = compute_movement_per_frame(positions)  # (N-1,)
window = 5
crop_start = 0

for i in range(len(pos_movement) - window + 1):
    if np.all(pos_movement[i:i+window] > threshold_m):  # 连续 5 帧都在运动
        crop_start = i
        break
```

- 默认 `threshold_m = 2.0mm / 1000 = 0.002m` (V3-mp 第 513 行)
- 逻辑: 找到第一个**连续 5 帧位移都超过阈值**的位置

### 3c. 裁剪终点 (crop_end) — 有夹爪变化时

**V3-mp 第 97-115 行:**

```python
if gripper is not None and len(gripper) > 0:
    gripper_changes = np.abs(np.diff(gripper))           # (N-1,)
    significant_changes = np.where(gripper_changes > 0.1)[0]  # 夹爪变化 > 0.1 的帧

    if len(significant_changes) > 0:
        last_gripper_change = significant_changes[-1]  # 最后一次夹爪动作
        buffer = 15                                     # 往后多保留 15 帧
        search_start = min(last_gripper_change + buffer, n_frames - 1)

        stability_threshold = 0.0015  # 1.5mm，比开始的阈值更小
        stability_window = 5

        for i in range(search_start, len(pos_movement) - stability_window + 1):
            if np.all(pos_movement[i:i+stability_window] < stability_threshold):
                crop_end = i + 1  # 连续 5 帧位移 < 1.5mm → 机器人停了
                break
        else:
            crop_end = min(search_start + 20, n_frames)  # 没找到稳定点，最多再加 20 帧
```

### 3d. 裁剪终点 — 无夹爪变化时

**V3-mp 第 116-131 行:**

```python
    else:
        # 没有显著夹爪变化 或 无夹爪数据
        is_moving = pos_movement > threshold_m
        moving_indices = np.where(is_moving)[0]
        if len(moving_indices) > 0:
            crop_end = min(moving_indices[-1] + 2, n_frames)  # 最后一个运动帧 + 2
        else:
            crop_end = n_frames
```

### 3e. 最短长度检查

**V3-mp 第 518-523 行 (在 `process_single_episode` 中):**

```python
new_length = crop_end - crop_start
if new_length < MIN_EPISODE_LENGTH:  # < 17 帧
    crop_stats['skipped'] = True
    return {'success': False, 'crop_stats': crop_stats}  # 丢弃此 episode
```

### 数据形状变化

```
原始 positions: (N, 3)
    ↓ detect_crop_range()
crop_start, crop_end (两个 int)
    ↓
有效范围: positions[crop_start : crop_end]
有效长度: M = crop_end - crop_start  (M < N, 且 M >= 17)
```

---

## 4. 图像加载与处理

### 4a. 加载图像时间戳

**V3-mp 第 180-188 行 `load_image_timestamps()`:**

```python
def load_image_timestamps(timestamps_file):
    # 输入: RGB_Images/timestamps.csv (CSV格式, 含 frame_index 和 timestamp 列)
    frame_indices = []
    timestamps = []
    with open(timestamps_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame_indices.append(int(row['frame_index']))
            timestamps.append(float(row['timestamp']))
    return frame_indices, np.array(timestamps)
    # 返回: frame_indices (list[int]), image_timestamps (np.array shape (K,))
```

### 4b. 时间戳匹配: 轨迹帧 → 图像帧

**V3-mp 第 191-193 行 `find_nearest_frame()`:**

```python
def find_nearest_frame(traj_timestamp, image_timestamps, frame_indices):
    idx = np.argmin(np.abs(image_timestamps - traj_timestamp))  # 最近邻搜索
    return frame_indices[idx]  # 返回对应的图像帧号
```

- 轨迹和图像有各自的时间戳，采样率不一定相同
- 对每个轨迹时间点，找时间最近的图像帧

### 4c. 图像加载、裁剪、缩放

**V3-mp 第 196-208 行 `load_and_process_image()`:**

```python
def load_and_process_image(image_path, target_size=(256, 256), use_center_crop=True):
    img = cv2.imread(str(image_path))          # BGR, shape: (H_orig, W_orig, 3), 如 (1080, 1920, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # RGB, shape: (1080, 1920, 3)

    if use_center_crop:
        img = center_crop_and_resize(img, target_size)  # → (256, 256, 3)
    else:
        img = cv2.resize(img, (target_size[1], target_size[0]))  # → (256, 256, 3)

    return img.astype(np.uint8)  # (256, 256, 3) uint8
```

### `center_crop_and_resize` 内部实现

**pose_utils.py 第 496-524 行:**

```python
def center_crop_and_resize(img, target_size=(224, 224)):
    h, w = img.shape[:2]   # 例如 h=1080, w=1920

    crop_size = min(h, w)  # = 1080 (取短边为正方形边长)
    left = (w - crop_size) // 2  # = (1920 - 1080) // 2 = 420
    top  = (h - crop_size) // 2  # = (1080 - 1080) // 2 = 0

    cropped = img[top:top+crop_size, left:left+crop_size]  # (1080, 1080, 3)

    resized = cv2.resize(cropped, (target_size[1], target_size[0]))  # (256, 256, 3)

    return resized
```

### 数据形状变化

```
原始图像 (JPEG/PNG):  (1080, 1920, 3) BGR uint8
    ↓ cv2.cvtColor
                       (1080, 1920, 3) RGB uint8
    ↓ center_crop (取中心 1080x1080)
                       (1080, 1080, 3) RGB uint8
    ↓ cv2.resize
                       (256, 256, 3) RGB uint8
```

---

## 5. 计算相对动作 (Relative Action)

这是数据处理中最核心的数学部分。action 不是简单的差值，而是**在当前帧局部坐标系下的相对变换**。

### 5a. 状态 → 4x4 齐次变换矩阵

**V3-mp 第 235-242 行 `pose_to_mat()`:**

```python
def pose_to_mat(pose):
    pos = pose[:3]              # (3,) 位置 [x, y, z]
    rot6d = pose[3:9]           # (6,) rot6d
    R = rot6d_to_mat(rot6d)     # (3, 3) 旋转矩阵 (Gram-Schmidt 正交化)
    T = np.eye(4)               # (4, 4) 单位矩阵
    T[:3, :3] = R               # 填入旋转
    T[:3, 3] = pos              # 填入平移
    return T                    # (4, 4) 齐次变换矩阵
```

其中 `rot6d_to_mat` 的 Gram-Schmidt 过程 (**pose_utils.py 第 100-110 行**):

```python
a1 = d6[:3]   # rot6d 前 3 维 → 旋转矩阵第 1 行 (未归一化)
a2 = d6[3:]   # rot6d 后 3 维 → 旋转矩阵第 2 行 (未归一化)

b1 = normalize(a1)                      # 归一化第 1 行
b2 = a2 - np.sum(b1 * a2) * b1          # 减去 a2 在 b1 方向的投影
b2 = normalize(b2)                      # 归一化第 2 行
b3 = np.cross(b1, b2)                   # 叉积得到第 3 行

return np.stack([b1, b2, b3], axis=0)   # (3, 3) 正交旋转矩阵
```

### 5b. 计算相对变换

**V3-mp 第 252-258 行 `compute_relative_action_10d()`:**

```python
def compute_relative_action_10d(base_state, target_state):
    base_mat   = pose_to_mat(base_state[:9])    # T_base:   (4, 4) 当前帧的世界坐标变换
    target_mat = pose_to_mat(target_state[:9])   # T_target: (4, 4) 下一帧的世界坐标变换

    # 核心公式: T_relative = T_base^{-1} @ T_target
    # 这是在当前帧 (base) 的局部坐标系下，到达 target 需要的变换
    relative_mat = np.linalg.inv(base_mat) @ target_mat  # (4, 4)

    relative_pose = mat_to_pose10d(relative_mat)  # (9,) = [rel_x, rel_y, rel_z, rel_rot6d(6)]
    gripper = target_state[9]                      # 夹爪取目标帧的绝对值 (不是 delta)

    return np.concatenate([relative_pose, [gripper]]).astype(np.float32)  # (10,) float32
```

**V3-mp 第 245-249 行 `mat_to_pose10d()`:**

```python
def mat_to_pose10d(T):
    pos = T[:3, 3]            # (3,) 相对位移
    R = T[:3, :3]             # (3, 3) 相对旋转矩阵
    rot6d = mat_to_rot6d(R)   # (6,) 相对旋转的 rot6d 表示
    return np.concatenate([pos, rot6d])  # (9,)
```

### 数学含义

```
T_base   = 当前帧在世界坐标系中的位姿 (4x4)
T_target = 下一帧在世界坐标系中的位姿 (4x4)

T_relative = T_base^{-1} @ T_target

这意味着:
  T_target = T_base @ T_relative

即: 从当前帧出发，在当前帧的局部坐标系中施加 T_relative，就到达下一帧。
这与 "世界坐标系下的差值" (T_target - T_base) 不同！
旋转和平移是耦合在一起计算的。
```

### 输出 action 的 10 维含义

```
action (10,) float32:
  [0:3]  rel_pos    在当前帧坐标系下的相对平移 (米)
  [3:9]  rel_rot6d  在当前帧坐标系下的相对旋转 (rot6d)
  [9]    gripper    目标帧的夹爪绝对值 (0=张开, 1=闭合)
```

**重要**: 这里 action 的旋转不是 `R_delta = R_next @ R_current^T`（世界坐标系下的旋转差），而是完整 body-frame relative transform 的旋转部分。位置部分也不是 `pos_next - pos_current`，而是 `R_current^T @ (pos_next - pos_current)`，即在当前帧局部坐标系下的相对位移。

---

## 6. 逐帧数据组装

**V3-mp 第 541-569 行 (在 `process_single_episode` 中):**

```python
frames_data = []
for t in range(crop_start, min(crop_end - 1, n_frames - 1)):
    # 注意上界是 crop_end - 1，因为 action 需要 states[t+1]

    # 6a. 时间戳匹配 → 图像帧号
    frame_idx = find_nearest_frame(traj_times[t], image_timestamps, frame_indices)

    # 6b. 查找图像文件 (先 jpg 后 png)
    image_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
    if not image_path.exists():
        image_path = frames_dir / f"frame_{frame_idx:06d}.png"
        if not image_path.exists():
            continue  # 图像缺失，跳过

    # 6c. 加载并处理图像
    image = load_and_process_image(image_path, image_size, use_center_crop=True)
    # → (256, 256, 3) uint8

    # 6d. 取当前帧和下一帧的 10D 状态
    current_state = states[t]      # (10,) float32
    next_state    = states[t + 1]  # (10,) float32

    # 6e. 跳过含 NaN 的帧
    if np.any(np.isnan(current_state)) or np.any(np.isnan(next_state)):
        continue

    # 6f. 计算相对动作
    action = compute_relative_action_10d(current_state, next_state)  # (10,) float32

    # 6g. 存入列表
    frames_data.append({
        'state':  current_state,   # (10,) float32 — 绝对状态
        'image':  image,           # (256, 256, 3) uint8 — RGB 图像
        'action': action,          # (10,) float32 — 相对动作
    })
```

### 关键细节

- 遍历范围 `[crop_start, min(crop_end-1, n_frames-1))`，即最后一帧不会被处理（因为它没有 `t+1` 来计算 action）
- 最终每个 episode 产出 `M` 帧数据，`M ≤ crop_end - crop_start - 1`

---

## 7. 数据增强 (轨迹版本)

**V3-mp 第 681-694 行:**

```python
versions_to_process = augment_versions if augment_versions else [None]

# 构建任务列表: 每个 (session, version) 作为独立 episode
tasks = []
for session_dir in session_dirs:
    for version in versions_to_process:
        tasks.append((
            str(session_dir), version, task_description,
            True,  # use_center_crop
            threshold_mm, image_size_tuple,
        ))
```

- 命令行参数如 `--augment-versions 0 5 10 15 20`
- 对应不同的轨迹文件: `slam_raw_pose_worldframe_downsampled-v0.txt`, `-v5.txt`, ..., `-v20.txt`
- 这些是 SLAM 系统对同一段录制做不同帧偏移的重建结果，提供轨迹层面的数据增强
- 例如 100 个 session × 5 个版本 = 500 个 episodes

---

## 8. Phase 1: 多进程并行处理

**V3-mp 第 698-710 行:**

```python
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = {executor.submit(process_single_episode, task): task for task in tasks}

    with tqdm(total=len(tasks), desc="Processing") as pbar:
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
            pbar.update(1)
```

- 使用 `ProcessPoolExecutor` (多进程)，因为工作是 CPU 密集型（图像读取、矩阵运算）
- 每个 worker 独立执行 `process_single_episode()`: 加载轨迹 → 10D 转换 → 裁剪 → 匹配图像 → 读图 → 计算 action
- 各 worker 之间完全独立，无共享状态

---

## 9. Phase 2: 排序与索引分配

**V3-mp 第 712-725 行:**

```python
# 只保留成功的结果
successful = [r for r in results if r.get('success', False)]
# 按 (session_name, version) 排序，保证确定性
successful.sort(key=lambda x: (x['session_name'], x.get('version', 0) or 0))

# 顺序分配全局索引
global_index = 0
for i, result in enumerate(successful):
    result['episode_index'] = i                          # episode 编号: 0, 1, 2, ...
    result['global_index_start'] = global_index           # 全局帧偏移
    global_index += len(result['frames_data'])            # 累加

total_frames = global_index  # 所有 episode 的总帧数
```

---

## 10. Phase 3: 多线程并行写入

### 10a. 写 Parquet 文件

**V3-mp 第 588-612 行 `write_single_episode()`:**

```python
def write_single_episode(episode_index, global_index_start, frames_data,
                         output_dir, video_codec):
    n_frames = len(frames_data)
    chunk_idx = episode_index // 1000  # 每 1000 个 episode 一个 chunk

    # state 和 action 从 numpy 转为 python list (parquet 兼容性)
    states  = [f['state'].tolist() for f in frames_data]   # list of list[float], 每个 len=10
    actions = [f['action'].tolist() for f in frames_data]  # list of list[float], 每个 len=10

    df = pd.DataFrame({
        'observation.state': states,                                        # list[10 floats]
        'action':            actions,                                       # list[10 floats]
        'timestamp':         [i / FPS for i in range(n_frames)],            # 0.0, 0.05, 0.10, ...
        'frame_index':       list(range(n_frames)),                         # 0, 1, 2, ...
        'episode_index':     [episode_index] * n_frames,                    # 全部相同
        'index':             list(range(global_index_start,
                                        global_index_start + n_frames)),    # 全局连续
        'task_index':        [0] * n_frames,                                # 固定 0
    })

    parquet_path = output_dir / f"data/chunk-{chunk_idx:03d}" / f"episode_{episode_index:06d}.parquet"
    df.to_parquet(parquet_path, index=False)
```

#### Parquet 每行的数据结构

| 列名 | dtype | shape/含义 |
|------|-------|-----------|
| `observation.state` | list\[float32\] (len=10) | `[x, y, z, rot6d(6), gripper]` 绝对状态 |
| `action` | list\[float32\] (len=10) | `[rel_xyz(3), rel_rot6d(6), gripper]` body-frame 相对动作 |
| `timestamp` | float32 | `frame_index / 20` 秒 |
| `frame_index` | int64 | episode 内帧号 (从 0 开始) |
| `episode_index` | int64 | 全局 episode 编号 |
| `index` | int64 | 全局帧编号 (跨 episode 连续) |
| `task_index` | int64 | 固定 0 (单任务) |

### 10b. 编码视频

**V3-mp 第 265-312 行 `encode_video()`:**

```python
def encode_video(images_hwc, output_path, fps=20, codec="h264"):
    h, w = images_hwc[0].shape[:2]  # 256, 256

    # h264 编码参数
    codec_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo",           # 输入格式: 原始像素
        "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24",        # 输入像素格式: RGB 每通道 8bit
        "-s", f"{w}x{h}",           # 输入分辨率: 256x256
        "-r", str(fps),             # 输入帧率: 20
        "-i", "-",                  # 从 stdin 读取
        *codec_args,                # 编码器参数
        "-pix_fmt", "yuv420p",      # 输出像素格式
        "-an",                      # 无音频
        str(output_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for img in images_hwc:
        proc.stdin.write(img.tobytes())  # 每帧 256*256*3 = 196608 bytes
    proc.stdin.close()
    proc.wait()
```

- 视频路径: `videos/chunk-{chunk_idx:03d}/observation.images.wrist/episode_{ep:06d}.mp4`
- 输入: 内存中的 RGB 图像列表，每张 `(256, 256, 3) uint8`
- 通过 pipe 喂给 ffmpeg，避免写临时文件
- 使用 `ThreadPoolExecutor` 而非 `ProcessPoolExecutor`，因为:
  1. 线程共享内存，不需要序列化大量图像 numpy 数组
  2. ffmpeg 是子进程，不受 Python GIL 限制

---

## 11. 写 Meta 文件

**V3-mp 第 438-478 行 `write_meta_files()`:**

### 11a. `meta/info.json`

**V3-mp 第 367-435 行 `generate_info_json()`:**

```json
{
  "codebase_version": "v2.1",
  "robot_type": "fastumi",
  "total_episodes": 500,
  "total_frames": 45000,
  "total_tasks": 1,
  "total_videos": 500,
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 20,
  "splits": { "train": "0:500" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    "observation.state": { "dtype": "float32", "shape": [10] },
    "observation.images.wrist": {
      "dtype": "video",
      "shape": [256, 256, 3],
      "info": {
        "video.height": 256, "video.width": 256, "video.channels": 3,
        "video.fps": 20.0, "video.codec": "h264", "video.pix_fmt": "yuv420p"
      }
    },
    "action": { "dtype": "float32", "shape": [10] },
    "timestamp": { "dtype": "float32", "shape": [1] },
    "frame_index": { "dtype": "int64", "shape": [1] },
    "episode_index": { "dtype": "int64", "shape": [1] },
    "index": { "dtype": "int64", "shape": [1] },
    "task_index": { "dtype": "int64", "shape": [1] }
  }
}
```

### 11b. `meta/modality.json`

**V3-mp 第 319-364 行 `generate_modality_json()`:**

描述 state/action 向量中各部分的位置和语义:

```json
{
  "action": {
    "eef_pos":   { "start": 0, "end": 3, "original_key": "action", "absolute": false },
    "eef_rot6d": { "start": 3, "end": 9, "original_key": "action", "rotation_type": "rotation_6d", "absolute": false },
    "gripper":   { "start": 9, "end": 10, "original_key": "action", "absolute": false }
  },
  "state": {
    "eef_pos":   { "start": 0, "end": 3, "original_key": "observation.state" },
    "eef_rot6d": { "start": 3, "end": 9, "original_key": "observation.state", "rotation_type": "rotation_6d" },
    "gripper":   { "start": 9, "end": 10, "original_key": "observation.state" }
  },
  "video": {
    "wrist": { "original_key": "observation.images.wrist" }
  },
  "annotation": {
    "human.action.task_description": { "original_key": "task_index" }
  }
}
```

### 11c. `meta/tasks.jsonl`

```json
{"task_index": 0, "task": "pickandplace-314"}
```

### 11d. `meta/episodes.jsonl`

每个 episode 一行:
```json
{"episode_index": 0, "length": 89, "tasks": ["pickandplace-314"]}
{"episode_index": 1, "length": 92, "tasks": ["pickandplace-314"]}
...
```

### 11e. `episode_session_mapping.json` (根目录)

**V3-mp 第 459-478 行:**

记录 episode ↔ session 的映射关系，用于 session-level train/val 分割:

```json
{
  "description": "Episode to session mapping for session-level train/val split",
  "augmented": true,
  "augment_factor": 5,
  "augment_versions": [0, 5, 10, 15, 20],
  "n_sessions": 100,
  "n_episodes": 500,
  "episode_mapping": [
    {"episode_index": 0, "session_name": "session_001", "version": 0},
    {"episode_index": 1, "session_name": "session_001", "version": 5},
    ...
  ],
  "session_to_episodes": {
    "session_001": [0, 1, 2, 3, 4],
    ...
  }
}
```

---

## 12. Train/Val 分割

**V3-mp 第 669-677 行:**

```python
if train_only:
    n_val = max(1, int(len(session_dirs) * val_ratio))  # 至少 1 个 val session
    n_train = len(session_dirs) - n_val
    val_sessions = [s.name for s in session_dirs[n_train:]]  # 最后面的 session 做验证
    session_dirs = session_dirs[:n_train]                     # 只保留前面的 session
```

- 按 **session 级别** 切分（不是 episode 级别），防止同一 session 的增强版本泄漏到验证集
- 按目录排序的顺序（通常是时间顺序），最后 20% 的 session 做验证

---

## 13. 完整数据形状变化总览

```
输入 (每个 session):
  轨迹文件:   txt, 每行 8 float
  图像文件夹: frame_000000.jpg ... (1080×1920 BGR)
  时间戳:     timestamps.csv

    ↓ load_trajectory()          第 148-177 行
  traj dict: 各字段 (N,)

    ↓ process_trajectory_10d()   第 215-232 行
  states: (N, 10) float32
    [xyz(3) + rot6d(6) + gripper(1)]
    欧拉角(度) → 弧度 → Rotation.as_matrix() → mat[:2,:].flatten()
    夹爪: 1.0 - g (翻转)

    ↓ detect_crop_range()        第 83-141 行
  crop_start, crop_end → 有效范围 [crop_start, crop_end)

    ↓ 逐帧处理                   第 541-569 行
  对 t in [crop_start, min(crop_end-1, n_frames-1)):
    图像: (1080, 1920, 3) → center_crop → (1080, 1080, 3) → resize → (256, 256, 3) uint8
    state: states[t]     → (10,) float32   绝对值
    action: inv(T_t) @ T_{t+1} → (10,) float32   body-frame 相对值

    ↓ write_single_episode()     第 588-620 行
  Parquet: episode_{i:06d}.parquet, M 行 × 7 列
  Video:   episode_{i:06d}.mp4, M 帧 256×256 @ 20fps h264

    ↓ write_meta_files()         第 438-478 行
  meta/info.json, modality.json, tasks.jsonl, episodes.jsonl
  episode_session_mapping.json
```

---

## 14. 输出目录结构

```
{output_dir}/{category_name}/
├── meta/
│   ├── info.json                  # 数据集全局元数据
│   ├── modality.json              # state/action 维度映射
│   ├── tasks.jsonl                # 任务描述
│   └── episodes.jsonl             # 每个 episode 的长度和任务
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet # 每个 episode 的帧数据
│       ├── episode_000001.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.wrist/
│           ├── episode_000000.mp4 # 每个 episode 的视频
│           ├── episode_000001.mp4
│           └── ...
└── episode_session_mapping.json   # episode ↔ session 映射
```
