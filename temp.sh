#!/bin/bash
sudo bash -c '{
echo "===== dmesg NVRM/GPU ====="
dmesg | grep -iE "nvidia|nvrm|gpu|fabric|nvlink" | tail -100
echo
echo "===== nvidia-firmware packages ====="
dpkg -l | grep -iE "nvidia-firmware|fabricmanager"
echo
echo "===== fabric manager service ====="
systemctl status nvidia-fabricmanager --no-pager 2>&1 | head -30
echo
echo "===== persistenced service ====="
systemctl status nvidia-persistenced --no-pager 2>&1 | head -15
echo
echo "===== loaded module file path ====="
modinfo nvidia | head -10
echo
echo "===== alternatives ====="
update-alternatives --display nvidia 2>&1 | head -20
echo
echo "===== /var/log nvidia ====="
ls -la /var/log/ | grep -i nvidia
} > /tmp/gpu_diag.txt 2>&1 && chmod 644 /tmp/gpu_diag.txt && echo DONE'
