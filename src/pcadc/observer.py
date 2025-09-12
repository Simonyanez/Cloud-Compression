from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
from dataclasses import dataclass
import sqlite3
import numpy as np

from .blocks import Block
from .graph import StructuralGraph, AttributeGraph
from .color import FitResult
from .decider import RDO_Decision
from .encoder import EncodeResult
from .clusterer import Codebook
from .transforms import CoeffsContainer


# ------------------------
# Event types
# ------------------------

@dataclass
class FitEvent:
    block: Block
    result: FitResult

@dataclass
class CodebookEvent:
    block: Block
    codebook: Codebook

@dataclass
class RDOEvent:
    q_step: int
    decision: RDO_Decision
    coeffs_container: CoeffsContainer

@dataclass
class EncodeEvent:
    experiment_code: str
    result: EncodeResult

@dataclass
class CoeffsEvent:
    block: Block
    graph: StructuralGraph | AttributeGraph
    coeffs: np.ndarray
    GFT_mat: np.ndarray


# Union of all events
ExperimentEvent = Union[FitEvent, CodebookEvent, RDOEvent, EncodeEvent, CoeffsEvent]


# ------------------------
# Observer interface
# ------------------------

class ExperimentObserver(ABC):
    @abstractmethod
    def update(self, event: ExperimentEvent) -> None:
        """React to an event with its associated data."""
        pass


# TODO: MOVE THIS TO THE INFORMATION THEORY METRICS MODULE 
def coeff_entropy(coeffs: np.ndarray, bins: int = 64) -> float:
    """Shannon entropy of quantized coefficients."""
    hist, _ = np.histogram(coeffs, bins=bins, density=True)
    hist = hist[hist > 0]  # avoid log(0)
    return float(-np.sum(hist * np.log2(hist)))

def energy_compaction(coeffs: np.ndarray, k: int | None = None) -> float:
    """Fraction of energy in top-k coefficients by magnitude."""
    energy_total = float(np.sum(coeffs**2))
    if energy_total == 0:
        return 0.0
    if k is None:
        coeffs_sorted = np.sort(coeffs**2)[::-1]
        k = coeffs_sorted.shape[0] // 4  # e.g., top 25%
        return float(np.sum(coeffs_sorted[:k]) / energy_total)
    return float(np.sum(coeffs[:k]**2) / energy_total)

# ------------------------
# SQLite Sink
# ------------------------

class SQLiteSink(ExperimentObserver):
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(db_path)
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self.conn.cursor()

        cur.execute("""CREATE TABLE IF NOT EXISTS fit(
            block_idx INT,
            block_id TEXT,
            rmse REAL,
            coeffs TEXT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS coeffs(
            block_idx INT,
            block_id TEXT,
            graph_id TEXT,
            graph_descriptor TEXT,
            energy REAL
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS decision(
            block_idx INT,
            block_id TEXT,
            graph_id TEXT,
            graph_descriptor TEXT,
            rate_diff REAL,
            dist_diff REAL,
            entropy_diff REAL,
            cost REAL
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS encoding(
            experiment_code TEXT,
            q_step INT,
            psnr REAL,
            bpv REAL,
            bitstream_size INT,
            overhead_bpv REAL,
            overhead_size INT
        )""")

        cur.execute("""CREATE TABLE IF NOT EXISTS cluster(
            block_idx INT,
            block_id TEXT,
            label INT,
            centroid TEXT
        )""")

        self.conn.commit()

    def update(self, event: ExperimentEvent) -> None:
        cur = self.conn.cursor()

        if isinstance(event, FitEvent):
            cur.execute(
                "INSERT INTO fit VALUES (?, ?, ?, ?)",
                (event.block.block_idx, event.block.block_id, event.result.rmse, str(event.result.coeffs))
            )

        elif isinstance(event, CodebookEvent):
            label = event.codebook.labels[event.block.block_idx]
            centroid = event.codebook.get_assigned_centroid(event.block.block_idx)
            cur.execute(
                "INSERT INTO cluster VALUES (?, ?, ?, ?)",
                (event.block.block_idx, event.block.block_id, int(label), str(centroid.tolist()))
            )

        elif isinstance(event, CoeffsEvent):
            # Skip entropy here; it will be calculated in RDO
            coeff = event.coeffs
            graph_descriptor = event.graph.metadata.graph_descriptor
            graph_id = event.graph.metadata.graph_id
            eng = energy_compaction(coeff, k=10)
            cur.execute(
                "INSERT INTO coeffs(block_idx, block_id, graph_id, graph_descriptor, energy) VALUES (?, ?, ?, ?, ?)",
                (event.block.block_idx, event.block.block_id, graph_id, graph_descriptor, eng)
            )

        elif isinstance(event, RDOEvent):
            # Quantize coefficients
            coeffs_raw = event.coeffs_container.coeffs
            q_step = event.q_step
            entropy_diffs = []
            for i, graph_coeffs in enumerate(coeffs_raw):
                quantized = np.round(graph_coeffs / q_step) * q_step
                entropy_val = coeff_entropy(quantized)
                entropy_diffs.append(entropy_val)
            rates = event.decision.rates
            dists = event.decision.distorsions
            rate_diff = 0
            dist_diff = 0
            entropy_diff = 0
            if len(rates) > 1:
                rate_diff = float(rates[1] - rates[0])
                dist_diff = float(dists[1] - dists[0])
                entropy_diff = float(entropy_diffs[1] - entropy_diffs[0])
            cur.execute(
                "INSERT INTO decision(block_idx, block_id, graph_id, graph_descriptor, rate_diff, dist_diff, entropy_diff, cost) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (event.coeffs_container.block.block_idx,
                 event.coeffs_container.block.block_id,
                 event.decision.selected_graph_metadata.graph_id,
                 event.decision.selected_graph_metadata.graph_descriptor,
                 rate_diff,
                 dist_diff,
                 entropy_diff,
                 float(event.decision.cost))
            )

        elif isinstance(event, EncodeEvent):
            r = event.result
            cur.execute(
                "INSERT INTO encoding(experiment_code, q_step, psnr, bpv, bitstream_size, overhead_bpv, overhead_size) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (event.experiment_code,
                 event.result.q_step,
                 float(r.PSNR),
                 float(r.bpv),
                 int(r.bitstream_size),
                 float(r.overhead_bpv),
                 int(r.overhead_bitstream_size))
            )

        self.conn.commit()

class DiagnosticVisualizer(ExperimentObserver):
    def __init__(self):
        pass
        
    def update(self, event: ExperimentEvent) -> None:
        match event:
            case FitEvent(): self._visualize_fit(event)
            case CodebookEvent(): self._visualize_codebook(event)
            case CoeffsEvent(): self._visualize_coeffs(event)
            case RDOEvent(): self._visualize_rdo(event)
            case EncodeEvent(): self._visualize_encoding(event)

    def _visualize_fit(self, event: FitEvent):
        #TODO: Plot original block
        #TODO: Plot Y channel colormap
        #TODO: Plot Y approximation by fit with its RMSE
        pass

    def _visualize_codebook(self, event: CodebookEvent):
        #TODO: Plot Y channel colormap
        #TODO: Plot Y approximation using cluster centroid
        #TODO: Plot Y approximation using assigned cluster centroid
        #TODO: Plot labels and assignations distributions
        pass

    def _visualize_coeffs(self, event: CoeffsEvent):
        #TODO: Plot energy compaction 10 first coefficients 3 channels
        #TODO: Plot graph with its connections
        #TODO: Plot first GFT matrix as colormap
        #TODO: Plot first basis projection
        pass

    def _visualize_rdo(self, event:RDOEvent):
        #TODO: Plot distorsions and rates with graph ids
        #TODO: Plot coeffs energy compaction compared showing selected with its RDO Cost
        pass

    def _visualize_encoding(self, event: EncodeEvent):
        #TODO: Plot results table
        pass

