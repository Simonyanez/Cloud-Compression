"""
Decoupled Viterbi Sweep
========================
Runs Upgrade B (decoupled mode) for gamma_final = 2000.0 and 4000.0.
Prints out the resulting GFT savings, sequence entropy, and Net BPV.
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

# We will append to the existing DB, no unlinking.
print(f"Starting Decoupled Viterbi Sweep for gamma0 = 2000.0 and 4000.0...")
print(f"Database path: {Path(db_path).absolute()}\n")

gammas = [2000.0, 4000.0]

for gamma in gammas:
    print(f"\n============================================================")
    print(f"RUNNING DECOUPLED VITERBI WITH GAMMA_FINAL = {gamma}")
    print(f"============================================================")
    
    cmd = [
        "python3", "-u", "scripts/oracle_em_tangent.py",
        "--block_size", str(block_size),
        "--clusters", str(clusters),
        "--q_step", str(q_step),
        "--gamma", str(gamma),
        "--max_iters", str(max_iters),
        "--sample_rate", str(sample_rate),
        "--adaptive_normals",
        "--decoupled",
        "--db_path", db_path
    ]
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()

print("\n============================================================")
print("DECOUPLED SWEEP COMPLETED. RESULTS REPORT:")
print("============================================================")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all runs in database
cursor.execute("SELECT run_id, gamma FROM runs ORDER BY run_id;")
runs = cursor.fetchall()

print("| Run ID | Gamma0 | Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches |")
print("|---:|---:|---:|---:|---:|---:|---:|")

for run_id, gamma in runs:
    cursor.execute("""
        SELECT rate_reduction_pct, entropy_markov, overhead_bpv_markov, net_bpv_markov, num_switches
        FROM iterations
        WHERE run_id = ? AND iteration = ?;
    """, (run_id, max_iters))
    row = cursor.fetchone()
    if row:
        savings, entropy, overhead, net, switches = row
        print(f"| {run_id} | {gamma:.1f} | {savings:.3f}% | {entropy:.4f} | {overhead:.5f} | {net:+.5f} | {switches} |")

conn.close()
