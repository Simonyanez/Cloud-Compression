"""
Tangent Geometric Scan Order Sweep Runner
==========================================
Runs a sequence of oracle_em_tangent.py executions with --geometric_scan enabled
across progressive gamma values: gamma in [0.0, 100.0, 1000.0, 2000.0]
Queries the resulting SQLite database to compare results with Morton order.
"""

import sys
import subprocess
import sqlite3
from pathlib import Path

# Config
gammas = [0.0, 100.0, 1000.0, 2000.0]
db_path = "experiments/tangent_geom_sweep.db"
block_size = 16
clusters = 4
q_step = 24
max_iters = 10
sample_rate = 0.2
gamma_warmup = 3

# Clear old sweep DB if exists
db_file = Path(db_path)
if db_file.exists():
    db_file.unlink()

print(f"Starting Tangent Geometric Scan Sweep on longdress...")
print(f"Database path: {db_file.absolute()}\n")

for gamma in gammas:
    print(f"\n============================================================")
    print(f"RUNNING GEOMETRIC SCAN EM WITH GAMMA = {gamma}")
    print(f"============================================================")
    
    cmd = [
        "python3", "-u", "scripts/oracle_em_tangent.py",
        "--block_size", str(block_size),
        "--clusters", str(clusters),
        "--q_step", str(q_step),
        "--gamma", str(gamma),
        "--max_iters", str(max_iters),
        "--gamma_warmup", str(gamma_warmup),
        "--sample_rate", str(sample_rate),
        "--geometric_scan",
        "--db_path", db_path
    ]
    
    # Run process and stream stdout
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    process.wait()
    
    if process.returncode != 0:
        print(f"Error: Run with gamma={gamma} failed with exit code {process.returncode}")
        sys.exit(1)

# sweep completed, query DB and build summary markdown table
print("\n============================================================")
print("GEOMETRIC SCAN SWEEP COMPLETED. GENERATING REPORT...")
print("============================================================")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all runs in database
cursor.execute("SELECT run_id, gamma FROM runs ORDER BY run_id;")
runs = cursor.fetchall()

report_lines = []
report_lines.append("# Tangent Geometric Scan Sweep Results Report\n")
report_lines.append("This report summarizes the performance metrics of the **Tangent-Space EM loop with Geometric Scan Order (Normal Azimuth Sorting)** across progressive continuity penalty values ($\gamma$):\n")
report_lines.append("| Gamma ($\gamma$) | Savings % | $H(X\\|X-1)$ | Overhead BPV | Net Markov BPV | Switches | Avg L1 Norm |")
report_lines.append("|---:|---:|---:|---:|---:|---:|---:|")

for run_id, gamma in runs:
    cursor.execute("""
        SELECT rate_reduction_pct, entropy_markov, overhead_bpv_markov, net_bpv_markov, num_switches, avg_l1_norm
        FROM iterations
        WHERE run_id = ? AND iteration = ?;
    """, (run_id, max_iters))
    row = cursor.fetchone()
    if row:
        savings, entropy, overhead, net, switches, avg_l1 = row
        report_lines.append(f"| {gamma:.1f} | {savings:.3f}% | {entropy:.4f} | {overhead:.5f} | {net:+.5f} | {switches} | {avg_l1:.2f} |")

conn.close()

# Write to markdown artifact file
artifact_dir = Path("/home/simao/.gemini/antigravity-cli/brain/bea95583-c565-4fdd-b212-641f0ae23deb")
artifact_dir.mkdir(parents=True, exist_ok=True)
report_file = artifact_dir / "tangent_geom_sweep_results.md"

with open(report_file, "w") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"\nReport written to: {report_file.absolute()}")
