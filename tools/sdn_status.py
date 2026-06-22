#!/usr/bin/env python3
"""Emit a compact JSON status of the SDN training run.

Used by the automated iteration loop to decide whether training is still
running, how far it got, and how the loss is trending. Run on the server:

    python tools/sdn_status.py                       # auto-detect latest log
    python tools/sdn_status.py logs/train_sdn_x.log
"""
import glob
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def proc_running() -> bool:
    try:
        out = subprocess.check_output(['pgrep', '-af', 'run.py'], text=True)
    except subprocess.CalledProcessError:
        return False
    return any('train_sdn' in line for line in out.splitlines())


def checkpoint_info():
    ckpts = glob.glob(os.path.join(REPO, 'experiments/sdn/checkpoint_*.pkl'))
    epochs = []
    for c in ckpts:
        m = re.search(r'checkpoint_(\d+)\.pkl$', c)
        if m:
            epochs.append(int(m.group(1)))
    epochs.sort()
    return {
        'num_checkpoints': len(epochs),
        'max_epoch': epochs[-1] if epochs else None,
        'latest_ckpt': (f'experiments/sdn/checkpoint_{epochs[-1]}.pkl' if epochs else None),
    }


def latest_log():
    link = os.path.join(REPO, 'logs/train_sdn_latest.log')
    if os.path.islink(link) or os.path.exists(link):
        return os.path.realpath(link) if os.path.islink(link) else link
    logs = sorted(glob.glob(os.path.join(REPO, 'logs/train_sdn_*.log')),
                  key=os.path.getmtime)
    return logs[-1] if logs else None


def parse_log(path, tail_bytes=200_000):
    if not path or not os.path.exists(path):
        return {}
    with open(path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - tail_bytes))
        text = f.read().decode('utf-8', errors='ignore')
    epoch_losses = re.findall(r'epoch_loss[\'"]?\s*[:=]\s*([0-9.eE+-]+)', text)
    last_losses = re.findall(r'Loss:\s*([0-9.eE+-]+)', text)
    errors = [l for l in text.splitlines()
              if re.search(r'Traceback|Error|Segmentation|RuntimeError', l)]
    return {
        'log_path': path,
        'recent_epoch_losses': [float(x) for x in epoch_losses[-10:]],
        'recent_step_losses': [float(x) for x in last_losses[-5:]],
        'last_error_lines': errors[-5:],
        'tail': '\n'.join(text.splitlines()[-8:]),
    }


def main():
    log = sys.argv[1] if len(sys.argv) > 1 else latest_log()
    status = {
        'running': proc_running(),
        **checkpoint_info(),
        **parse_log(log),
    }
    print(json.dumps(status, indent=2))


if __name__ == '__main__':
    main()
