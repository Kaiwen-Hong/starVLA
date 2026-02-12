# Fix: Two Conflicting Conda Installations Breaking `starvla` Alias

**Date:** 2025-02-07
**Symptom:** `conda activate starVLA` shows `(starVLA)` in prompt, but `which python` resolves to `~/miniforge3/bin/python` (Python 3.12, base env) instead of the lab env's Python 3.10. Results in `ModuleNotFoundError: No module named 'rich'` when running the smoke test.

---

## Context

This is a shared lab account (`haonan`) on an HPC cluster. Kaiwen's personal shell config lives in `~/.bashrc-kaiwen`, sourced via the `starvla` alias defined in `~/.bashrc`.

There are **two separate miniforge3 installations**:

| | Home miniforge | Lab miniforge |
|---|---|---|
| **Path** | `~/miniforge3/` (`/n/home01/haonan/miniforge3/`) | `/net/.../kaiwen/miniforge3/` |
| **Initialized by** | `~/.bashrc` line 44 | `~/.bashrc-kaiwen` line 33 |
| **Base Python** | 3.12 | 3.10 |
| **Has `starVLA` env?** | **No** (removed 2025-02-07, was incomplete) | Yes (fully installed, `pip install -e .` done here) |

The `starvla` alias in `~/.bashrc` (line 84):
```bash
alias starvla='source ~/.bashrc-kaiwen && cd /net/.../kaiwen/starVLA && conda activate starVLA'
```

---

## Problem: Three Layers of PATH Corruption

### Layer 1: Home miniforge `bin/` stuck in PATH

`~/.bashrc` line 13 unconditionally prepends `~/miniforge3/bin` to PATH at shell startup:
```bash
for p in "$HOME/miniforge3/bin" "$HOME/miniforge3/condabin" "$HOME/.local/bin"; do
  PATH="$p:$PATH"
done
```

When `~/.bashrc-kaiwen` later reinitializes conda from the lab miniforge, it replaces the `conda` shell function but **does not remove `~/miniforge3/bin` from PATH**.

### Layer 2: Stale `CONDA_PREFIX` from home miniforge

After `~/.bashrc` runs, these environment variables are set by the home miniforge's conda init:
```
CONDA_PREFIX=/n/home01/haonan/miniforge3    # points to HOME base
CONDA_SHLVL=1
CONDA_DEFAULT_ENV=base
```

`~/.bashrc-kaiwen` reinitializes the conda function (so `conda` now calls the lab binary) but **does not clear these variables**. When `conda activate starVLA` runs, it sees `CONDA_PREFIX=~/miniforge3` as the "current env" and incorporates its paths into the new PATH stack.

### Layer 3: `00-base-node-path.sh` activate hook (the final nail)

`~/.bashrc` lines 93-107 define a `PROMPT_COMMAND` hook called `__ensure_conda_node_hook`. On every prompt, it symlinks this file into every conda env's `activate.d/`:

```
~/miniforge3/etc/conda/activate.d/00-base-node-path.sh
```

That script contains:
```bash
_BASE_NODE_BIN="$HOME/miniforge3/bin"
case ":$PATH:" in
  *":${_BASE_NODE_BIN}:"*) ;;
  *) export PATH="${_BASE_NODE_BIN}:$PATH" ;;
esac
```

This **unconditionally prepends `~/miniforge3/bin` to the front of PATH** every time any `conda activate` runs. Even if layers 1 and 2 were fixed, this hook would re-inject the home Python 3.12 at the top of PATH, guaranteeing it always wins over the env's Python 3.10.

### Net result

After `starvla` alias completes, PATH looks like:
```
1. ~/miniforge3/bin              <- HOME base Python 3.12 (WINS, no 'rich')
2. .../cmake/...
3. ~/.local/bin
4. .../gcc/...
5. /net/.../kaiwen/miniforge3/envs/starVLA/bin  <- correct Python 3.10 (LOSES)
```

`CONDA_PREFIX` correctly points to the lab env, and the prompt shows `(starVLA)`, but `which python` resolves to position 1.

---

## Solution: Changes to `~/.bashrc-kaiwen`

Added three blocks near the top of `~/.bashrc-kaiwen`, **before** the conda init block:

### Change 1: Strip home miniforge from PATH (fixes Layer 1)

```bash
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/miniforge3" | tr '\n' ':')
PATH=${PATH%:}
```

Removes all `~/miniforge3/bin` and `~/miniforge3/condabin` entries that `~/.bashrc` put there.

### Change 2: Unset stale conda variables (fixes Layer 2)

```bash
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
```

Clears the home miniforge's conda state so the lab miniforge starts clean.

### Change 3: Disable the PROMPT_COMMAND hook (fixes Layer 3)

```bash
PROMPT_COMMAND=$(echo "$PROMPT_COMMAND" | sed 's/__ensure_conda_node_hook[; ]*//')
```

Removes the `__ensure_conda_node_hook` from `PROMPT_COMMAND` so it stops symlinking `00-base-node-path.sh` into conda envs.

Also manually removed the existing symlink:
```bash
rm /net/.../kaiwen/miniforge3/envs/starVLA/etc/conda/activate.d/00-base-node-path.sh
```

### Why Changes 1–3 alone were NOT enough

The `~/.bashrc-kaiwen` changes correctly clean PATH, unset stale variables, and disable the PROMPT_COMMAND hook **before** the lab conda init runs. However, the `activate.d/00-base-node-path.sh` symlink already existed inside the lab starVLA env (created by the PROMPT_COMMAND hook in a previous session). Conda's `activate` mechanism sources all scripts in `$CONDA_PREFIX/etc/conda/activate.d/` **during** `conda activate`, which happens **after** `~/.bashrc-kaiwen` has finished. So the symlink re-injected `~/miniforge3/bin` into PATH at activate time, undoing the cleanup.

The `~/.bashrc-kaiwen` fix prevents the hook from creating **new** symlinks, but does not remove **existing** ones. Furthermore, the `__ensure_conda_node_hook` still runs in any terminal session that hasn't sourced `~/.bashrc-kaiwen` (e.g. a plain terminal), so it would **re-create** symlinks in any lab env that gets activated from such a session.

### Change 4: Delete existing activate.d symlinks

```bash
# Delete all symlinks across all lab miniforge envs
find /net/.../kaiwen/miniforge3 -name "00-base-node-path.sh" -type l -delete
```

This removes symlinks from starVLA, RoboTwin, and the lab miniforge base env.

However, this alone is **not permanent** — the `__ensure_conda_node_hook` in `~/.bashrc` would re-create them in any non-`starvla` session.

### Change 5: Disable the source script (permanent fix for Layer 3)

Replaced the contents of the source file with a no-op comment:

```bash
echo '# disabled — was injecting ~/miniforge3/bin into PATH, breaking lab conda envs' \
  > ~/miniforge3/etc/conda/activate.d/00-base-node-path.sh
```

This is the **permanent fix**. Even if `__ensure_conda_node_hook` creates new symlinks, they all point to this source file, which now does nothing. This is safe for collaborators because `~/.bashrc` line 13 already adds `~/miniforge3/bin` to PATH at shell startup — the hook script was redundant.

### Change 6: Remove Home miniforge's starVLA env (cleanup)

The Home miniforge (`~/miniforge3/`) had its own `starVLA` env with incomplete packages. Since all conda envs should live under lab storage (`/net/.../kaiwen/miniforge3/`), the Home copy was removed:

```bash
~/miniforge3/bin/conda env remove -n starVLA -y
```

This eliminates any risk of accidentally activating the wrong starVLA env and avoids confusion between the two installations.

---

## Full diff of `~/.bashrc-kaiwen`

```diff
 # Kaiwen's personal config

 # Working directory
 cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan

+# Strip home miniforge from PATH and reset stale conda state
+# so lab miniforge takes full precedence
+PATH=$(echo "$PATH" | tr ':' '\n' | grep -v "$HOME/miniforge3" | tr '\n' ':')
+PATH=${PATH%:}
+unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE _CE_CONDA _CE_M
+
+# Disable the PROMPT_COMMAND hook from .bashrc that symlinks
+# ~/miniforge3/bin into every conda env's activate.d/
+PROMPT_COMMAND=$(echo "$PROMPT_COMMAND" | sed 's/__ensure_conda_node_hook[; ]*//')
+
 # Add .local/bin to PATH
 export PATH="$HOME/.local/bin:$PATH"

 # MuJoCo setup
 ...
 (rest of file unchanged)
```

`~/.bashrc` was **not modified** (it is shared with other users on this account).

---

## Verification

After the fix, the `starvla` alias correctly resolves:

```
=== BEFORE (simulating post-.bashrc state) ===
which python: /n/home01/haonan/miniforge3/bin/python       # Python 3.12

=== AFTER starvla alias ===
which python: /net/.../kaiwen/miniforge3/envs/starVLA/bin/python  # Python 3.10
CONDA_PREFIX: /net/.../kaiwen/miniforge3/envs/starVLA
rich: imports successfully
home miniforge in PATH: (none)
```

The smoke test should now work:
```bash
source ~/.bashrc
starvla
conda activate starVLA
python starVLA/model/framework/QwenGR00T.py   # should print model and exit
```

---

## Summary of All Changes

| Change | What | Fixes |
|--------|------|-------|
| 1. Strip PATH | `~/.bashrc-kaiwen`: remove `~/miniforge3` from PATH | Layer 1 |
| 2. Unset vars | `~/.bashrc-kaiwen`: unset stale `CONDA_PREFIX` etc. | Layer 2 |
| 3. Disable hook | `~/.bashrc-kaiwen`: remove `__ensure_conda_node_hook` from `PROMPT_COMMAND` | Layer 3 (prevents new symlinks) |
| 4. Delete symlinks | `find ... -name "00-base-node-path.sh" -type l -delete` | Layer 3 (removes existing symlinks) |
| 5. Disable source | Replace `~/miniforge3/etc/conda/activate.d/00-base-node-path.sh` with no-op | Layer 3 (permanent fix, all envs) |
| 6. Remove Home env | `~/miniforge3/bin/conda env remove -n starVLA` | Eliminates wrong env entirely |

Changes 1–3 alone were **insufficient** because existing `activate.d` symlinks still ran during `conda activate`. Change 4 removed them but wasn't permanent (the hook would re-create them). **Change 5 is the permanent fix** — disabling the source file makes all symlinks (existing and future) no-ops.
