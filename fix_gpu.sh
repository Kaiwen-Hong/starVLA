#!/bin/bash
# Fix: H200 GPUs require -open kernel module variant + -server stream
# (fabric manager / nscq only ship for nvidia-*-server-*).
#
# Single apt transaction: install server-open AND remove desktop/closed in one
# go, so apt resolves all dependencies consistently (avoids the libnvidia-
# compute-535 fallback that two-step purge/install would trigger).
#
# NOTE: this script does NOT use apt autoremove, apt upgrade, apt clean, or any
# command that would touch packages beyond the explicit lists below.

set -e

STAGE="${1:-dryrun}"

# Packages to install (server-stream, open variant)
INSTALL=(
  nvidia-driver-595-server-open
  nvidia-fabricmanager-595
  libnvidia-nscq-595
)

# Packages to remove (desktop-stream + closed-variant + 590 leftovers).
# Only packages currently installed will actually be passed to apt.
PURGE_CANDIDATES=(
  nvidia-dkms-590
  nvidia-dkms-595
  nvidia-kernel-source-595
  nvidia-driver-590-open
  nvidia-utils-590
  nvidia-utils-595
  nvidia-compute-utils-590
  nvidia-compute-utils-595
  nvidia-kernel-common-590
  nvidia-kernel-common-595
  nvidia-firmware-590-590.48.01
  nvidia-firmware-595-595.71.05
  libnvidia-compute-590
  libnvidia-compute-595
  libnvidia-cfg1-590
  libnvidia-cfg1-595
  libnvidia-common-590
  libnvidia-common-595
  libnvidia-decode-590
  libnvidia-decode-595
  libnvidia-encode-590
  libnvidia-encode-595
  libnvidia-extra-590
  libnvidia-extra-595
  libnvidia-fbc1-590
  libnvidia-fbc1-595
  libnvidia-gl-590
  libnvidia-gl-595
  xserver-xorg-video-nvidia-590
  xserver-xorg-video-nvidia-595
  linux-modules-nvidia-590-open-generic
  linux-modules-nvidia-590-open-6.8.0-106-generic
  linux-modules-nvidia-590-open-6.8.0-110-generic
  linux-modules-nvidia-595-open-generic
  linux-modules-nvidia-595-open-6.8.0-117-generic
)

# Filter purge list to only installed packages, and append "-" suffix for apt
build_apt_args() {
  APT_ARGS=("${INSTALL[@]}")
  for p in "${PURGE_CANDIDATES[@]}"; do
    if dpkg -l "$p" 2>/dev/null | grep -q "^ii"; then
      APT_ARGS+=("${p}-")
    fi
  done
}

if [ "$STAGE" = "dryrun" ]; then
  echo "===== STAGE 1: DRY RUN — nothing will be changed ====="
  echo
  build_apt_args
  echo "--- Combined install+remove transaction (single apt-get install call) ---"
  echo "Args: ${APT_ARGS[*]}"
  echo
  sudo apt-get install --simulate "${APT_ARGS[@]}"
  echo
  echo "===== KEY THINGS TO CHECK IN THE OUTPUT ====="
  echo "1. NO 'libnvidia-compute-535' (or any old version) being newly installed."
  echo "2. NO 'cuda', 'nsight', 'libcudart' packages being removed."
  echo "3. NO 'upgraded' count > 0 (we only want install/remove, no upgrades)."
  echo "4. Should install: nvidia-driver-595-server-open + nvidia-fabricmanager-595"
  echo "                 + nvidia-dkms-595-server-open + all -595-server libs"
  echo "                 + nvidia-firmware-595-server-595.71.05"
  echo "                 + nvidia-kernel-common-595-server"
  echo "===== If all checks pass, run:  bash $0 apply  ====="
  exit 0
fi

if [ "$STAGE" = "apply" ]; then
  echo "===== STAGE 2: APPLYING FIX ====="
  echo "WARNING: this will modify drivers on a shared server. CTRL-C now if not OK."
  sleep 5

  echo "--- Stopping nvidia-persistenced ---"
  sudo systemctl stop nvidia-persistenced || true

  echo "--- Unloading current (wrong) nvidia modules ---"
  sudo rmmod nvidia_uvm     2>/dev/null || true
  sudo rmmod nvidia_drm     2>/dev/null || true
  sudo rmmod nvidia_modeset 2>/dev/null || true
  sudo rmmod nvidia         2>/dev/null || true

  build_apt_args
  echo "--- Single apt transaction: install + remove ---"
  sudo apt-get install -y "${APT_ARGS[@]}"

  echo "--- Loading new modules ---"
  sudo modprobe nvidia
  sudo modprobe nvidia_uvm
  sudo modprobe nvidia_modeset

  echo "--- Starting fabric manager + persistenced ---"
  sudo systemctl enable --now nvidia-fabricmanager
  sudo systemctl start nvidia-persistenced || true

  echo
  echo "===== Verifying ====="
  sleep 5
  nvidia-smi
  echo
  systemctl status nvidia-fabricmanager --no-pager | head -10
  exit 0
fi

echo "Usage: $0 dryrun   (or)   $0 apply"
exit 1
