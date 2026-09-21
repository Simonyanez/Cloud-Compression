#!/usr/bin/env python3
"""
Seeds a fresh E10 ablation DB with the sequences/cluster_centroids/mode_adapters tables
copied verbatim from the main results/exp_e10_multisequence.db, so exp_e10_multisequence_benchmark.py
--ablation-no-selfloop can reuse the already-trained codebook (same seq_id, same centroids, same
calibrated s_lp/s_lw) instead of retraining from scratch -- the ablation isolates the effect of
forcing s_lp=0 at encode time, not a fresh (and randomly different) codebook.

Only copies the tables that init_db()/has_codebook()/has_adapters()/load_codebook()/
load_adapters() actually read before deciding to skip training (see exp_e10_multisequence_benchmark.py):
sequences, cluster_centroids, mode_adapters. Does NOT copy frame_results/aggregate_results/etc --
those are exactly what this run is meant to produce fresh.
"""
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from exp_e10_multisequence_benchmark import init_db  # reuses the exact same schema

SRC = str(REPO_ROOT / "results" / "exp_e10_multisequence.db")
DST = str(REPO_ROOT / "results" / "exp_e10_ablation_noselfloop.db")

TABLES_TO_COPY = ["sequences", "cluster_centroids", "mode_adapters"]


def main():
    init_db(DST, overwrite=False)  # creates all tables fresh (empty) with the real schema

    src = sqlite3.connect(f"file:{SRC}?mode=ro", uri=True)
    dst = sqlite3.connect(DST)

    for table in TABLES_TO_COPY:
        cols = [r[1] for r in src.execute(f"PRAGMA table_info({table})")]
        rows = src.execute(f"SELECT {','.join(cols)} FROM {table}").fetchall()
        placeholders = ",".join(["?"] * len(cols))
        dst.executemany(
            f"INSERT INTO {table} ({','.join(cols)}) VALUES ({placeholders})", rows
        )
        print(f"  [+] {table}: {len(rows)} rows copied")

    dst.commit()

    # Sanity check: same seq_id values present in both DBs, same codebook row counts per seq/K.
    src_seq = dict(src.execute("SELECT name, seq_id FROM sequences").fetchall())
    dst_seq = dict(dst.execute("SELECT name, seq_id FROM sequences").fetchall())
    assert src_seq == dst_seq, f"seq_id mismatch: {src_seq} vs {dst_seq}"
    for name, sid in src_seq.items():
        src_n = src.execute("SELECT COUNT(*) FROM cluster_centroids WHERE seq_id=?", (sid,)).fetchone()[0]
        dst_n = dst.execute("SELECT COUNT(*) FROM cluster_centroids WHERE seq_id=?", (sid,)).fetchone()[0]
        assert src_n == dst_n, f"{name}: cluster_centroids row count mismatch ({src_n} vs {dst_n})"

    print(f"Done. Seeded {DST} from {SRC}: {len(src_seq)} sequences, seq_id-consistent, "
          f"codebook row counts verified equal.")
    src.close(); dst.close()


if __name__ == "__main__":
    raise SystemExit(main())
