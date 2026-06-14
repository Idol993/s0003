import sys
sys.path.insert(0, '.')
import subprocess
import os

os.chdir('d:/Worksolo/s0003')
cmd = [
    'python', 'train_1024expert.py',
    '--num_experts', '128',
    '--model_dim', '128',
    '--num_layers', '1',
    '--vocab_size', '512',
    '--batch_size', '4',
    '--seq_len', '64',
    '--max_steps', '200',
    '--report_interval', '20',
    '--cluster_interval', '50',
    '--warmup_steps', '30',
    '--cpu',
    '--inject_imbalance',
    '--inject_ratio', '0.03',
]
print("Running hell-difficulty imbalance test with 128 experts...")
result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print(f"\nReturn code: {result.returncode}")
