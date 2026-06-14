import sys
sys.path.insert(0, '.')
import subprocess
import os

os.chdir('d:/Worksolo/s0003')
cmd = [
    'python', 'train_1024expert.py',
    '--num_experts', '64',
    '--model_dim', '128',
    '--num_layers', '1',
    '--vocab_size', '512',
    '--batch_size', '4',
    '--seq_len', '32',
    '--max_steps', '30',
    '--report_interval', '10',
    '--cluster_interval', '100',
    '--warmup_steps', '10',
    '--cpu',
    '--no_inject_imbalance',
]
print("Running:", " ".join(cmd))
result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)
