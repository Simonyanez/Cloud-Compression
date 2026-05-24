#!/bin/bash

# Script to run vanishing cluster tests consecutively from 4 to 8 clusters
# Usage: ./bin/run_vanishing_tests.sh

# Get the project root directory (parent of bin)
ROOT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$ROOT_DIR"

echo "=== Starting Vanishing Cluster Tests (4 to 8 clusters) ==="

for C in {1..8}
do
    echo ""
    echo "--------------------------------------------------------"
    echo "[*] Running test for C = $C clusters"
    echo "--------------------------------------------------------"
    
    python3 scripts/test_vanishing_clusters.py --clusters "$C" --block_size 8 --sample_frac 0.005 --max_iters 20
    
    if [ $? -ne 0 ]; then
        echo "[!] Error occurred during test with $C clusters. Continuing..."
    fi
done

echo ""
echo "=== All tests completed! Check the 'results' folder for output. ==="
