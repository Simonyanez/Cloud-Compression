import argparse
from typing import Optional,TYPE_CHECKING
from src.transforms import *
if TYPE_CHECKING:
    from src.objects import * 


def main():
    parser = argparse.ArgumentParser(description="Checking certain block")

    # Add arguments
    parser.add_argument('--b', '--block-idx', type=int, help='Block index')
    parser.add_argument('--v', '--verbose', type=bool, default=False, help='Enable verbose ouput')
    parser.parse_args()

def inspect_block(idx,):
    bloc
