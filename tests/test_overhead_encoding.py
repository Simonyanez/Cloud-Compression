import numpy as np
import pytest
import constriction
from pcadc.encoder import Encoder

def test_arithmetic_encoding_overhead_basic():
    """Test the arithmetic encoding logic with a simple biased distribution."""
    # Create a biased assignation array (mostly 0s)
    assignation = np.array([0, 0, 0, 0, 1, 0, 0, 0, 2, 0] * 100) # 1000 elements
    
    # 1) Build empirical probability model
    unique_labels, counts = np.unique(assignation, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()

    # Map labels to contiguous indices [0, ..., K-1]
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indexed_assignation = np.array(
        [label_to_index[x] for x in assignation],
        dtype=np.int32
    )

    # Create categorical entropy model
    model = constriction.stream.model.Categorical(
        probabilities.astype(np.float32),
        perfect=False
    )

    # Encode using RangeEncoder
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(indexed_assignation, model)
    compressed = encoder.get_compressed()

    # Each word is uint32 -> 32 bits
    bs_size = len(compressed) * 32
    
    # Theoretical entropy: -sum(p * log2(p))
    entropy = -np.sum(probabilities * np.log2(probabilities))
    theoretical_bits = entropy * len(assignation)
    
    print(f"\nResults for Biased Pattern:")
    print(f"Unique Labels: {unique_labels}")
    print(f"Probabilities: {probabilities}")
    print(f"Entropy: {entropy:.4f} bits/symbol")
    print(f"Theoretical total bits: {theoretical_bits:.2f}")
    print(f"Arithmetic coded bits: {bs_size}")
    
    # Arithmetic coding should be close to theoretical entropy for large N
    # Allow some overhead for the range coder (usually very small)
    assert bs_size >= theoretical_bits
    # Efficiency check: should not be vastly larger than entropy + 32 bits (1 word)
    assert bs_size <= theoretical_bits + 64 

def test_arithmetic_encoding_uniform():
    """Test with uniform distribution (maximum entropy)."""
    assignation = np.array([0, 1, 2, 3] * 250) # 1000 elements, uniform
    
    unique_labels, counts = np.unique(assignation, return_counts=True)
    probabilities = counts.astype(np.float64) / counts.sum()
    
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indexed_assignation = np.array([label_to_index[x] for x in assignation], dtype=np.int32)

    model = constriction.stream.model.Categorical(probabilities.astype(np.float32), perfect=False)
    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(indexed_assignation, model)
    compressed = encoder.get_compressed()
    bs_size = len(compressed) * 32
    
    entropy = -np.sum(probabilities * np.log2(probabilities)) # log2(4) = 2.0
    theoretical_bits = entropy * len(assignation)
    
    print(f"\nResults for Uniform Pattern:")
    print(f"Entropy: {entropy:.4f} bits/symbol")
    print(f"Theoretical total bits: {theoretical_bits:.2f}")
    print(f"Arithmetic coded bits: {bs_size}")
    
    assert entropy == pytest.approx(2.0)
    assert bs_size >= theoretical_bits
    assert bs_size <= theoretical_bits + 64

def test_encoder_get_overhead_bpv_integration():
    """Test the integration within the Encoder class."""
    # We need to mock or provide a minimal Coeffs array since _as_bpv uses its shape
    encoder = Encoder()
    encoder.Coeffs = np.zeros((10000, 3)) # 10000 voxels
    
    # 1000 blocks (each block has one label in assignation)
    assignation = np.random.choice([0, 1, 2], size=1000, p=[0.7, 0.2, 0.1])
    
    bs_size, bpv = encoder.get_overhead_bpv(assignation)
    
    assert bs_size > 0
    assert bpv > 0
    assert bpv == bs_size / 10000
    print(f"\nEncoder Integration:")
    print(f"Total Overhead Bits: {bs_size}")
    print(f"Overhead BPV (per voxel): {bpv:.6f}")

if __name__ == "__main__":
    # If run directly, run these tests
    test_arithmetic_encoding_overhead_basic()
    test_arithmetic_encoding_uniform()
    test_encoder_get_overhead_bpv_integration()
