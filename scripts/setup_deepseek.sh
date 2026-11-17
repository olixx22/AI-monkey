#!/bin/sh
# Explicit opt-in installation. The application's normal startup never runs this.
set -eu
umask 077
project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
service_dir="$project_dir/.services/opencode-deepseek"
revision=2e3c81d7ea47a5fabb5aa7fa2a4f45cf681981c9
python_bin=${CTF_BRIDGE_PYTHON:-python3.13}
command -v "$python_bin" >/dev/null
mkdir -p "$project_dir/.services"
if [ ! -d "$service_dir/.git" ]; then
    git clone https://github.com/Tsuev/opencode-deepseek.git "$service_dir"
fi
if [ -n "$(git -C "$service_dir" status --porcelain --untracked-files=no)" ]; then
    echo 'Bridge checkout has local source changes; refusing to overwrite.' >&2
    exit 1
fi
git -C "$service_dir" checkout --detach "$revision"
if [ ! -x "$service_dir/.venv/bin/python" ]; then
    "$python_bin" -m venv "$service_dir/.venv"
fi
"$service_dir/.venv/bin/python" -m pip install -r "$service_dir/requirements.txt"
# Upstream auth uses installed Google Chrome. No browser/profile is installed or modified here.
echo 'Bridge installed. Next: python3 -m ctf_agent bridge login'
