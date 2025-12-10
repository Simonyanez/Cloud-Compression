from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path
from dataclasses import dataclass
import matplotlib.pyplot as plt
import time
import sqlite3
import numpy as np

from .blocks import Block
from .graph import StructuralGraph, AttributeGraph
from .color import FitResult, Approximator
from .decider import RDO_Decision
from .encoder import EncodeResult
from .clusterer import Codebook
from .transforms import CoeffsContainer
from .visualization import Visualizer
from .managers import VisualizationManager, VisualizationType
from .rd_cluster.states import ClusteringHistory
import collections
import heapq


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

@dataclass
class ClusteringHistoryEvent:
    experiment_code: str
    block_size: int
    history: ClusteringHistory


# Union of all events
ExperimentEvent = Union[FitEvent, CodebookEvent, RDOEvent, EncodeEvent, CoeffsEvent, ClusteringHistoryEvent]


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

        cur.execute("""CREATE TABLE IF NOT EXISTS clustering_history(
            experiment_code TEXT,
            block_size INT,
            iteration INT,
            q_step INT,
            total_cost REAL,
            cluster_entropy REAL,
            avg_rate REAL,
            avg_distortion REAL,
            hamming_distance_from_previous INT
        )""")

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
        
        elif isinstance(event, ClusteringHistoryEvent):
            previous_labels = None
            for state in event.history.states:
                hamming_dist = 0
                if previous_labels is not None:
                    hamming_dist = np.sum(previous_labels != state.labels)
                
                cur.execute(
                    "INSERT INTO clustering_history VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.experiment_code,
                        event.block_size,
                        state.iteration,
                        state.qstep_value,
                        state.total_cost,
                        state.cluster_entropy,
                        state.avg_rate,
                        state.avg_distortion,
                        int(hamming_dist)
                    )
                )
                previous_labels = state.labels

            # --- New code for Hamming Code Estimation ---
            final_state = event.history.states[-1]
            labels = final_state.labels
            num_blocks = len(labels)
            
            # Calculate frequencies and probabilities
            label_counts = collections.Counter(labels)
            probabilities = {label: count / num_blocks for label, count in label_counts.items()}
            
            # Build Huffman tree and get codes
            huffman_codes_list = build_huffman_tree(probabilities)
            
            for symbol, code in huffman_codes_list:
                prob = probabilities[symbol]
                cur.execute(
                    "INSERT INTO cluster_codes VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        event.experiment_code,
                        event.block_size,
                        symbol,
                        prob,
                        code,
                        len(code)
                    )
                )

            # Estimate total bits
            huffman_codes = dict(huffman_codes_list)
            estimated_total_bits = sum(label_counts[label] * len(huffman_codes[label]) for label in label_counts)
            
            cur.execute(
                "INSERT INTO code_estimation VALUES (?, ?, ?, ?, ?)",
                (
                    event.experiment_code,
                    event.block_size,
                    num_blocks,
                    final_state.cluster_entropy,
                    estimated_total_bits
                )
            )

        self.conn.commit()

class DiagnosticVisualizer(ExperimentObserver):
    def __init__(self):
        self.enabled = True
        self.visualization_manager = VisualizationManager()
        self.visualizer = Visualizer()
        pass
        
    def update(self, event: ExperimentEvent) -> None:
        if not self.enabled:
            return

        match event:
            case FitEvent(): self._visualize_fit(event)
            case CodebookEvent(): self._visualize_codebook(event)
            case CoeffsEvent(): self._visualize_coeffs(event)
            case RDOEvent(): self._visualize_rdo(event)
            case EncodeEvent(): self._visualize_encoding(event)

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def _visualize_fit(self, event: FitEvent):
        Vblock, Ablock = event.block.get_data()
        fig = self.visualization_manager.create_visualization(VisualizationType.FIT_ANALYSIS,
                                                        name=f"Fit Visualization for Block {event.block.block_idx}",
                                                        vertices=Vblock,
                                                        attributes=Ablock,
                                                        fit_result=event.result
                                                        )
        # NOTE: Maybe show on demand
        plt.show()
        #TODO: Plot original block
        #TODO: Plot Y channel colormap
        #TODO: Plot Y approximation by fit with its RMSE
        pass

    def _visualize_codebook(self, event: CodebookEvent):
        Vblock, Ablock = event.block.get_data()
        assignation = event.codebook.assignation
        labels = event.codebook.labels
        Vblock_rotated = Approximator()._spatial_norm(Vblock)

        assigned_centroid = event.codebook.get_assigned_centroid(event.block.block_idx)
        Y_by_assignation = Vblock_rotated @ assigned_centroid.T

        labeled_centroid = event.codebook.get_labeled_centroid(event.block.block_idx)
        Y_by_label = Vblock_rotated @ labeled_centroid.T
        
        #TODO: Plot Y channel colormap
        #TODO: Plot Y approximation using cluster centroid
        #TODO: Plot Y approximation using assigned cluster centroid
        #TODO: Plot labels and assignations distributions
        pass

    def _visualize_coeffs(self, event: CoeffsEvent):
        coeffs = event.coeffs
        #TODO: Plot energy compaction 10 first coefficients 3 channels
        Vblock, _ = event.block.get_data()
        weights = event.graph.weights
        GFT_mat = event.GFT_mat
        first_basis = GFT_mat[:, 0]
        #TODO: Plot graph with its connections
        #TODO: Plot first GFT matrix as colormap
        #TODO: Plot first basis projection
        pass

    def _visualize_rdo(self, event:RDOEvent):
        rates = event.decision.rates
        distorsions = event.decision.distorsions
        graph_ids = [graph.graph_id for graph in event.coeffs_container.graphs]
        coeffs = event.coeffs_container.coeffs
        #TODO: Plot distorsions and rates with graph ids
        #TODO: Plot coeffs energy compaction compared showing selected with its RDO Cost
        pass

    def _visualize_encoding(self, event: EncodeEvent):
        #TODO: Plot experiment summary
        #TODO: Plot results table
        pass

