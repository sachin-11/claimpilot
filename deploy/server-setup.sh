#!/usr/bin/env bash
# One-time setup of a fresh Ubuntu 24.04 EC2 instance: Docker + Compose plugin, and a 4 GB swap file
# (the reranker model and the app together can use more RAM than a 4 GB instance has).
#
#   curl -fsSL <raw url of this file> | sudo bash        or:  sudo bash deploy/server-setup.sh
set -euo pipefail

if [[ $EUID -ne 0 ]]; then
  echo "run as root (sudo)" >&2
  exit 1
fi

if ! command -v docker >/dev/null; then
  apt-get update
  apt-get install -y ca-certificates curl git make
  curl -fsSL https://get.docker.com | sh
fi
# the default login user of the Ubuntu AMI
usermod -aG docker "${SUDO_USER:-ubuntu}" || true

if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  grep -q "^/swapfile" /etc/fstab || echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi

echo "done. Log out and back in so the docker group applies, then follow deploy/README.md"
