"""
Sweep Database Analyzer
=======================
Queries the overnight sweep database to find:
  1. All configurations that achieved a net compression gain (net_bpv_markov > 0).
  2. The best performing configurations for each block size (B) and Q-step (Q).
  3. A comparison between spatially-adaptive Potts and standard Potts.
  4. GFT representation changes (L1 norm, energy compaction) in winning configurations.

Usage:
  python3 scripts/analyze_sweep.py --db_path experiments/overnight_sweep.db
"""

import sqlite3
from pathlib import Path
from argparse import ArgumentParser

def main():
    parser = ArgumentParser(description="Analyze sweep database results")
    parser.add_argument("--db_path", type=str, default="experiments/overnight_sweep.db")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"Error: Database file not found at {db_path.absolute()}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Query 1: Total runs and iterations logged
    cursor.execute("SELECT COUNT(*) FROM runs")
    total_runs = cursor.fetchone()[0]
    print(f"============================================================")
    print(f"DATABASE SUMMARY: {db_path.name}")
    print(f"Total runs logged: {total_runs}")
    print(f"============================================================\n")

    # Query 2: Winning configurations (net_bpv_markov > 0) in final iterations
    # We look at the final iteration of each run (which has the highest gamma)
    query_winners = """
    SELECT 
        r.run_id, r.block_size, r.clusters, r.q_step, r.gamma, r.spatially_adaptive,
        i.iteration, i.rate_reduction_pct, i.entropy_markov, i.overhead_bpv_markov, i.net_bpv_markov,
        i.avg_l1_norm, i.structural_l1_norm
    FROM runs r
    JOIN iterations i ON r.run_id = i.run_id
    WHERE i.iteration = (SELECT MAX(iteration) FROM iterations WHERE run_id = r.run_id)
      AND i.net_bpv_markov > 0.0
    ORDER BY i.net_bpv_markov DESC
    """
    
    cursor.execute(query_winners)
    winners = cursor.fetchall()
    
    print(f"--- WINNING CONFIGURATIONS (Net BPV Markov > 0) ---")
    if not winners:
        print("  No configurations achieved a positive net compression gain.")
    else:
        print(f"  Found {len(winners)} winning configurations:\n")
        print(f"  {'RunID':>5} | {'B':>3} | {'K':>2} | {'Q':>2} | {'Gamma':>6} | {'SA':>2} | {'Savings%':>8} | {'H(X|X-1)':>8} | {'Overhead':>8} | {'Net Gain':>9}")
        print(f"  " + "-"*85)
        for w in winners:
            run_id, b, k, q, gamma, sa, it, rd, ent, oh, net, l1, l1_s = w
            sa_str = "Y" if sa == 1 else "N"
            print(f"  {run_id:>5d} | {b:>3d} | {k:>2d} | {q:>2d} | {gamma:>6.1f} | {sa_str:>2} | {rd:>7.2f}% | {ent:>8.4f} | {oh:>8.5f} | {net:>+9.5f}")
            
    # Query 3: Best configuration per Block Size and Q-Step
    print(f"\n--- BEST CONFIGURATION PER BLOCK SIZE & Q-STEP ---")
    for b in [32, 16, 8]:
        for q in [24, 36]:
            cursor.execute("""
            SELECT 
                r.run_id, r.clusters, r.gamma, r.spatially_adaptive,
                i.rate_reduction_pct, i.entropy_markov, i.overhead_bpv_markov, i.net_bpv_markov
            FROM runs r
            JOIN iterations i ON r.run_id = i.run_id
            WHERE r.block_size = ? AND r.q_step = ?
              AND i.iteration = (SELECT MAX(iteration) FROM iterations WHERE run_id = r.run_id)
            ORDER BY i.net_bpv_markov DESC
            LIMIT 1
            """, (b, q))
            res = cursor.fetchone()
            if res:
                run_id, k, gamma, sa, rd, ent, oh, net = res
                sa_str = "Yes" if sa == 1 else "No"
                result_str = f"WIN ({net:+.5f} bpv)" if net > 0 else f"LOSS ({net:+.5f} bpv)"
                print(f"  B={b:<2} Q={q:<2} -> Run {run_id:<4} (K={k}, G={gamma:.0f}, SA={sa_str}) | Savings={rd:.2f}% | Overhead={oh:.5f} bpv | Result: {result_str}")
            else:
                print(f"  B={b:<2} Q={q:<2} -> No runs logged")

    # Query 4: Spatially-Adaptive Potts vs Standard Potts head-to-head comparison
    print(f"\n--- SPATIALLY-ADAPTIVE VS STANDARD POTTS COMPARISON (Average Net BPV) ---")
    cursor.execute("""
    SELECT 
        r.spatially_adaptive, 
        AVG(i.net_bpv_markov), 
        AVG(i.rate_reduction_pct), 
        AVG(i.entropy_markov)
    FROM runs r
    JOIN iterations i ON r.run_id = i.run_id
    WHERE i.iteration = (SELECT MAX(iteration) FROM iterations WHERE run_id = r.run_id)
    GROUP BY r.spatially_adaptive
    """)
    comparison = cursor.fetchall()
    for row in comparison:
        sa, avg_net, avg_savings, avg_ent = row
        sa_str = "Spatially-Adaptive Potts" if sa == 1 else "Standard Potts"
        print(f"  {sa_str:<25} | Avg Net BPV: {avg_net:+.5f} | Avg GFT Savings: {avg_savings:.2f}% | Avg Entropy: {avg_ent:.3f} bits")

    conn.close()

if __name__ == "__main__":
    main()
