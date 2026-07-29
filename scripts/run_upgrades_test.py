"""
Tangent EM Upgrades Sweep Runner
=================================
Runs executions of oracle_em_tangent.py to evaluate:
  1. Upgrade B: Decoupled Optimization (EM with gamma=0, followed by a final Viterbi pass)
  2. Upgrade A + C: Centroid update damping (lr=0.3) + Binarized Potts threshold (tau=0.9)
Queries the resulting database and reports the results.
"""

import sys
import subprocess
import sqlite3
from pathlib import Path

db_path = "experiments/upgrades_test.db"
block_size = 16
clusters = 4
q_step = 24
max_iters = 10
sample_rate = 0.2
gamma_warmup = 3

# Clear old DB
db_file = Path(db_path)
if db_file.exists():
    db_file.unlink()

print(f"Starting Tangent Upgrades Test Sweeps on longdress...")
print(f"Database path: {db_file.absolute()}\n")

# Run 1: Upgrade B (Decoupled Viterbi)
print("\n============================================================")
print("RUNNING RUN 1: UPGRADE B (DECOUPLED OPTIMIZATION, GAMMA0 = 1000.0)")
print("============================================================")
cmd_b = [
    "python3", "-u", "scripts/oracle_em_tangent.py",
    "--block_size", str(block_size),
    "--clusters", str(clusters),
    "--q_step", str(q_step),
    "--gamma", "1000.0",
    "--max_iters", str(max_iters),
    "--sample_rate", str(sample_rate),
    "--adaptive_normals",
    "--decoupled",
    "--db_path", db_path
]
process = subprocess.Popen(cmd_b, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
process.wait()

# Run 2: Upgrade A + C (Momentum + Binarized Potts, Gamma0 = 1000.0)
print("\n============================================================")
print("RUNNING RUN 2: UPGRADE A + C (MOMENTUM LR = 0.3, THRESHOLD TAU = 0.9, GAMMA0 = 1000.0)")
print("============================================================")
cmd_ac = [
    "python3", "-u", "scripts/oracle_em_tangent.py",
    "--block_size", str(block_size),
    "--clusters", str(clusters),
    "--q_step", str(q_step),
    "--gamma", "1000.0",
    "--max_iters", str(max_iters),
    "--gamma_warmup", str(gamma_warmup),
    "--sample_rate", str(sample_rate),
    "--adaptive_normals",
    "--centroid_lr", "0.3",
    "--normal_threshold", "0.90",
    "--db_path", db_path
]
process = subprocess.Popen(cmd_ac, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in process.stdout:
    sys.stdout.write(line)
    sys.stdout.flush()
process.wait()

# Query DB
print("\n============================================================")
print("UPGRADES TEST COMPLETED. GENERATING REPORT...")
print("============================================================")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all runs in database
cursor.execute("SELECT run_id, gamma FROM runs ORDER BY run_id;")
runs = cursor.fetchall()

report_lines = []
report_lines.append("# Tangent EM Upgrades Results Report\n")
report_lines.append("This report summarizes the performance metrics of the **Tangent-Space EM loop upgrades** at baseline $\gamma_0 = 1000.0$:\n")
report_lines.append("| Run ID | Method Description | Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches | Avg L1 Norm |")
report_lines.append("|---:|:---|---:|---:|---:|---:|---:|---:|")

descriptions = {
    1: "Upgrade B (Decoupled Viterbi, gamma0 = 1000.0)",
    2: "Upgrade A+C (LR=0.3, Threshold=0.90, gamma0 = 1000.0)"
}

for run_id, gamma in runs:
    cursor.execute("""
        SELECT rate_reduction_pct, entropy_markov, overhead_bpv_markov, net_bpv_markov, num_switches, avg_l1_norm
        FROM iterations
        WHERE run_id = ? AND iteration = ?;
    """, (run_id, max_iters))
    row = cursor.fetchone()
    if row:
        savings, entropy, overhead, net, switches, avg_l1 = row
        desc = descriptions.get(run_id, f"Run {run_id}")
        report_lines.append(f"| {run_id} | {desc} | {savings:.3f}% | {entropy:.4f} | {overhead:.5f} | {net:+.5f} | {switches} | {avg_l1:.2f} |")

conn.close()

# Write to markdown artifact file
artifact_dir = Path("/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb")
artifact_dir.mkdir(parents=True, exist_ok=True)
report_file = artifact_dir / "tangent_upgrades_results.md"

with open(report_file, "w") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"\nReport written to: {report_file.absolute()}")
