import sqlite3
import pandas as pd
import collections
import heapq
from pathlib import Path

def build_huffman_tree(freq_map):
    heap = [[weight, [symbol, ""]] for symbol, weight in freq_map.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def add_huffman_analysis_to_db(experiment_code: str):
    db_path = Path(f"results/PCADC_TEST/{experiment_code}/{experiment_code}.db")
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Create new tables if they don't exist
    cur.execute("""CREATE TABLE IF NOT EXISTS cluster_codes(
        experiment_code TEXT,
        block_size INT,
        cluster_id INT,
        probability REAL,
        huffman_code TEXT,
        code_length INT
    )""")

    cur.execute("""CREATE TABLE IF NOT EXISTS code_estimation(
        experiment_code TEXT,
        block_size INT,
        total_blocks INT,
        cluster_entropy REAL,
        estimated_total_bits REAL
    )""")

    # Clear existing data in these tables to avoid duplicates if script is re-run
    cur.execute(f"DELETE FROM cluster_codes WHERE experiment_code = '{experiment_code}'")
    cur.execute(f"DELETE FROM code_estimation WHERE experiment_code = '{experiment_code}'")

    # Get final labels from the 'cluster' table
    cluster_df = pd.read_sql_query("SELECT label FROM cluster", conn)
    
    if cluster_df.empty:
        print(f"No cluster labels found for {experiment_code}. Skipping Huffman analysis.")
        conn.close()
        return

    labels = cluster_df['label'].values
    num_blocks = len(labels)
    
    # Calculate frequencies and probabilities
    label_counts = collections.Counter(labels)
    probabilities = {label: count / num_blocks for label, count in label_counts.items()}
    
    # Build Huffman tree and get codes
    huffman_codes_list = build_huffman_tree(probabilities)
    huffman_codes = dict(huffman_codes_list)

    for symbol, code in huffman_codes_list:
        prob = probabilities[symbol]
        cur.execute(
            "INSERT INTO cluster_codes VALUES (?, ?, ?, ?, ?, ?)",
            (
                experiment_code,
                # Get block_size from the experiment_code or config, for now hardcode or get from clustering_history
                # For simplicity, let's assume block_size is part of experiment_code for now, or fetch from config.json
                # For now, we'll fetch it from the clustering_history table's first entry
                pd.read_sql_query("SELECT block_size FROM clustering_history LIMIT 1", conn).iloc[0]['block_size'],
                symbol,
                prob,
                code,
                len(code)
            )
        )

    # Estimate total bits
    estimated_total_bits = sum(label_counts[label] * len(huffman_codes[label]) for label in label_counts)
    
    # Get cluster_entropy from the last state in clustering_history
    history_df = pd.read_sql_query("SELECT cluster_entropy FROM clustering_history ORDER BY iteration DESC LIMIT 1", conn)
    cluster_entropy = history_df.iloc[0]['cluster_entropy'] if not history_df.empty else 0.0

    cur.execute(
        "INSERT INTO code_estimation VALUES (?, ?, ?, ?, ?)",
        (
            experiment_code,
            pd.read_sql_query("SELECT block_size FROM clustering_history LIMIT 1", conn).iloc[0]['block_size'],
            num_blocks,
            cluster_entropy,
            estimated_total_bits
        )
    )

    conn.commit()
    conn.close()

if __name__ == "__main__":
    experiments = ["RD-Fast-B16", "RD-Fast-B8", "RD-Fast-B4"]
    for exp in experiments:
        print(f"Adding Huffman analysis for experiment: {exp}")
        add_huffman_analysis_to_db(exp)
        print(f"Finished Huffman analysis for experiment: {exp}")
