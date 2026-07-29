"""
Overnight Sweep Runner
======================
Executes a multi-parameter sweep from cheapest (B=32, K=2) to heaviest (B=8, K=6)
specifically targeting mid-quantizations (Q=24, Q=36) to find the net gain sweet spots.

Records results to a SQLite database. The sweep is resumable: if a configuration
already exists in the database, it is skipped.

Usage:
  python3 scripts/run_overnight_sweep.py --db_path experiments/overnight_sweep.db
"""

import sys
import subprocess
import sqlite3
from pathlib import Path
from argparse import ArgumentParser

# Path setup
project_root = Path(__file__).resolve().parent.parent

def run_already_exists(db_path: Path, bsize: int, clusters: int, q_step: int, gamma: float, sa: int) -> bool:
    if not db_path.exists():
        return False
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT r.run_id FROM runs r
            JOIN iterations i ON r.run_id = i.run_id
            WHERE r.block_size = ? AND r.clusters = ? AND r.q_step = ? 
              AND abs(r.gamma - ?) < 1e-4 AND r.spatially_adaptive = ?
            LIMIT 1
        """, (bsize, clusters, q_step, gamma, sa))
        res = cursor.fetchone()
        conn.close()
        return res is not None
    except sqlite3.OperationalError:
        return False

def main():
    parser = ArgumentParser(description="Resumable overnight sweep runner")
    parser.add_argument("--db_path", type=str, default="experiments/overnight_sweep.db")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Define Sweep Matrix
    # Ordered strictly from fastest/cheapest (B=32, K=2) to slowest (B=8, K=6)
    # and focusing on mid-quantizations Q=24 and Q=36.
    sweep_list = []

    block_sizes = [32, 16, 8]
    clusters_list = [2, 3, 4, 6]
    q_steps = [24, 36]
    sa_options = [1, 0]

    for bsize in block_sizes:
        for clusters in clusters_list:
            for q in q_steps:
                # Gamma needs to scale with block size due to total block cost scaling
                if bsize == 32:
                    gammas = [1000.0, 3000.0, 6000.0]
                elif bsize == 16:
                    gammas = [300.0, 800.0, 1500.0]
                else:  # bsize == 8
                    gammas = [100.0, 300.0, 600.0]
                
                for gamma in gammas:
                    for sa in sa_options:
                        sweep_list.append({
                            "block_size": bsize,
                            "clusters": clusters,
                            "q_step": q,
                            "gamma": gamma,
                            "spatially_adaptive": sa
                        })

    print(f"=== Overnight Sweep Initialization ===")
    print(f"Database: {db_path.absolute()}")
    print(f"Total configurations in matrix: {len(sweep_list)}")

    completed = 0
    skipped = 0

    for i, cfg in enumerate(sweep_list):
        bs = cfg["block_size"]
        c = cfg["clusters"]
        q = cfg["q_step"]
        g = cfg["gamma"]
        sa = cfg["spatially_adaptive"]

        # Check resumability
        if run_already_exists(db_path, bs, c, q, g, sa):
            print(f"[{i+1}/{len(sweep_list)}] Skipping: B={bs}, C={c}, Q={q}, G={g}, SA={sa} (Already completed)")
            skipped += 1
            continue

        print(f"\n[{i+1}/{len(sweep_list)}] RUNNING: B={bs}, C={c}, Q={q}, G={g}, SA={sa}...")
        
        cmd = [
            "python3", "scripts/oracle_em_overnight.py",
            "--block_size", str(bs),
            "--clusters", str(c),
            "--q_step", str(q),
            "--gamma", str(g),
            "--sample_rate", "0.20", 
            "--max_iters", "6",      
            "--min_iters", "4",
            "--gamma_warmup", "3",   
            "--freeze_slopes",       
            "--db_path", str(db_path)
        ]
        
        if sa == 1:
            cmd.append("--spatially_adaptive")

        # Execute subprocess
        try:
            subprocess.run(cmd, check=True)
            completed += 1
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Run failed for configuration: {cfg}")
            print(e)
            continue

    print(f"\n=== Sweep Finished ===")
    print(f"Total checked: {len(sweep_list)}")
    print(f"Newly completed: {completed}")
    print(f"Skipped (resumed): {skipped}")

if __name__ == "__main__":
    main()
