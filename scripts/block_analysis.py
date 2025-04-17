import sys
from pathlib import Path

# Get the absolute path to the 'src' directory relative to the project root
project_root = Path(__file__).resolve().parent.parent  # Two levels up from the script directory
src_path = project_root / 'src'

# Add the 'src' directory to sys.path
sys.path.insert(0, str(src_path))

# Print the updated sys.path for debugging purposes
print(sys.path)
import os
print(os.listdir())
# Now proceed with your imports
import argparse
from typing import Optional, TYPE_CHECKING
from transforms import *
from visualization import *
from main import * 
if TYPE_CHECKING:
    from objects import *

def main():
    parser = argparse.ArgumentParser(description="Checking certain block")

    # Add arguments
    parser.add_argument('--b', '--block-idx', type=int, help='Block index')
    parser.add_argument('--v', '--verbose', type=bool, default=False, help='Enable verbose ouput')
    args = parser.parse_args()

    if args.v:
        print("Verbose output enabled")

    inspect_block(args.b)

def inspect_block(idx):
    pc_path = Path("res/longdress_vox10_1051.ply")
    visualizer = Visualizer()
    researcher = Researcher()
    researcher.point_cloud(pc_path)
    researcher.point_cloud.do_block_partitioning(8 )
    V = researcher.point_cloud.V
    A = researcher.point_cloud.A
    block = researcher.point_cloud.get_block(idx)
    researcher._process_block(V, A, block, sl_weight=0.8, sl_percentage=0.15)
    selected_coeff, selected_graph = researcher._block_decider(block, q_step=24)
    visualizer.visualize_coeffs(selected_coeff)

if __name__ == "__main__":
    main()