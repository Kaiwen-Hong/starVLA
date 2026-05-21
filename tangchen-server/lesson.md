# Tangchen 服务器连接经验

## 拓扑

```
本机 Linux (kaiwen@kaiwen)
   │   Tailscale + SSH 公钥
   ▼
Windows 跳板机 (outsider, 100.86.61.58, 用户 wangp)
   │   Windows 上预装的 SSH 公钥（root 管理员配置）
   ▼
Tangchen GPU 服务器 (tams02, 76.53.68.2:23, 用户 wangpc)
```

**关键点**：两段都是 SSH，但**端口 23**（不是默认 22），用户名两端不同（`wangp` vs `wangpc`）。

---

## 当前已经配置好的（免密）

- **本机 → Windows**：免密，本机公钥已放进 Windows 的 `C:\ProgramData\ssh\administrators_authorized_keys`（因为 `wangp` 是管理员，公钥**必须**放这个路径，不是 `C:\Users\wangp\.ssh\authorized_keys`，这是 Windows OpenSSH 的硬规则）。
- **Windows → Tangchen**：免密，Windows 上已经有 root 管理员配置好的密钥。

本机 `~/.ssh/config` 里加了：

```sshconfig
Host win-outsider
  HostName 100.86.61.58
  User wangp
  IdentityFile ~/.ssh/id_ed25519

Host tangchen           # 注：这条只能配合 ProxyJump，但 Tangchen 没收我们的 key，所以直接 `ssh tangchen` 不通
  HostName 76.53.68.2
  Port 23
  User wangpc
  IdentityFile ~/.ssh/id_ed25519
  ProxyJump win-outsider
```

---

## 重要限制：不能在 Tangchen 上加自己的 SSH 公钥

Tangchen 的 `/home/wangpc/.ssh/` 目录和 `authorized_keys` 文件**属于 root**：

```
drwxr-xr-x  root:root  /home/wangpc/.ssh
-rw-r--r--  root:root  /home/wangpc/.ssh/authorized_keys
```

`wangpc` 用户**只读不写**。所以：

- ❌ `ssh-copy-id tangchen` —— 直接被服务器 `Permission denied (publickey)` 拒，因为它不接受密码登录
- ❌ 手动 `echo key >> ~/.ssh/authorized_keys` —— 权限不够
- ❌ `chmod ~/.ssh` —— "Operation not permitted"

**结果**：从本机直接 `ssh tangchen` / `scp` / `rsync` / VSCode Remote-SSH 都用不了。必须走 Windows 中转。

如果想用 VSCode Remote-SSH 或直接 scp，**唯一办法是请管理员（root）帮忙**把本机公钥追加到 Tangchen 的 `authorized_keys`。本机公钥位置：`~/.ssh/id_ed25519.pub`。

---

## 日常使用方法（经过验证）

### 方法 1：手动两步进入交互式 shell（最稳，tmux/python/conda 都正常）

```bash
ssh win-outsider               # 进 Windows，免密
ssh -p 23 wangpc@76.53.68.2    # 进 Tangchen，免密
```

进去之后是正常 bash，可以：
- `tmux new -s star` / `tmux a -t star`
- 跑 conda 环境、跑 Python、debug
- `cd /scratch/wangpc/starVLA && git pull`

### 方法 2：一条命令直达（推荐加 alias）

加到 `~/.bashrc`：

```bash
alias tc='ssh -t win-outsider ssh -p 23 wangpc@76.53.68.2'
```

然后：

```bash
tc                                                # 直接进 Tangchen
tc "cd /scratch/wangpc/starVLA && git pull"       # 远程跑一条命令
```

### 方法 3：在本机脚本里远程跑命令（非交互式）

```bash
ssh -o BatchMode=yes win-outsider \
    'ssh -o BatchMode=yes -p 23 wangpc@76.53.68.2 "cd /scratch/wangpc/starVLA && git pull"'
```

**实测可用** —— git pull 跑通了，fast-forward 没问题。

⚠️ 三层 shell 的引号嵌套很容易出错（特别是 Windows cmd.exe 当中间层）。复杂命令优先用方法 1/2 交互式跑。

---

## 文件传输

因为不能直接 `scp` 到 Tangchen，要走 Windows 中转。两种方式：

### 方式 A：经 Windows 中转（rsync 多次跳）

```bash
# 本机 → Windows
scp local-file win-outsider:C:/Users/wangp/tmp/

# 登进 Windows，再传到 Tangchen
ssh win-outsider
scp -P 23 C:/Users/wangp/tmp/local-file wangpc@76.53.68.2:/scratch/wangpc/
```

### 方式 B：直接 pipe（适合小文件 / 文本）

```bash
cat local-file | ssh -t win-outsider 'ssh -p 23 wangpc@76.53.68.2 "cat > /scratch/wangpc/remote-file"'
```

如果传输频繁，建议**找管理员加公钥**，然后直接 `scp file tangchen:~/` 完事。

---

## tmux 工作流（已验证 Tangchen 上 tmux 正常）

```bash
tc                          # 进 Tangchen
tmux ls                     # 看现有 session（实测有 "star" 和 "1"）
tmux a -t star              # 接已有 session
# Ctrl+b d 脱离 session（代码继续后台跑）
exit                        # 退出 ssh（Windows 那层也会自动断）
```

**重要**：tmux session 在 Tangchen 上常驻，即使你断了 SSH 也不会停。下次回来用 `tmux a -t star` 重新接上。

---

## 故障排查速查

| 症状 | 原因 | 解决 |
|---|---|---|
| `ssh win-outsider` 还要密码 | Windows 的 `administrators_authorized_keys` 没写对 / ACL 没改对 | 检查 `C:\ProgramData\ssh\administrators_authorized_keys`，跑 `icacls` 改权限 |
| `ssh tangchen` (用 ProxyJump 的) → `Permission denied (publickey)` | Tangchen 上没我们的公钥，且 root 锁死了不让加 | 走 Windows 中转 (方法 1/2/3)；或请管理员加 key |
| 三层引号报 `unexpected EOF` / `'head' is not recognized` | Windows cmd.exe 把内层引号吃掉 | 改成方法 1（交互式）跑，或把命令写到 Tangchen 上的脚本里再调用 |
| `Tailscale` 连不上 Windows | Tailscale 没运行 / 不同 tailnet | `tailscale status` 检查 |

---

## 一次性命令速查

```bash
# 验证两跳免密
ssh win-outsider "hostname"
ssh -t win-outsider 'ssh -p 23 wangpc@76.53.68.2 "hostname && whoami"'

# 远程 git pull
ssh win-outsider 'ssh -p 23 wangpc@76.53.68.2 "cd /scratch/wangpc/starVLA && git pull"'

# 远程看 tmux 状态
ssh win-outsider 'ssh -p 23 wangpc@76.53.68.2 "tmux ls"'
```
