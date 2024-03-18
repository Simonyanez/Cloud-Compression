from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any

class Block:
    def __init__(self,block,graph,property):
        self.block = block
        self.graph = graph              
        self.property = property

    # For printing object
    def __str__(self):
        pass

# Builder interface
class Block_Builder(ABC):
    """
    The Block Builder interface specifies methods for creating different parts or 
    characteristics of Blocks.
    """
    @property               # No need for using parenthesis
    @abstractmethod
    def product(self):
        pass

    @abstractmethod
    def produce_block(self,block):
        pass

    @abstractmethod         # Use parenthesis to call
    def produce_graph(self,graph):
        pass

    @abstractmethod
    def produce_property(self,property):
        pass

# Concrete builder for direction experiment
class DirectionBlockBuilder(Block_Builder):

    def produce_block(self,block):
        self.block = block

    def produce_graph(self,graph):
        # TODO: Create Graph class for more refactoring
        self.graph = graph
    
    def produce_property(self,property):
        # TODO: Create Property class for more refactoring
        self.property = property

    def build(self):
        return Block(self.block, self.graph, self.property)
    
class DifferenceBlockBuilder(Block_Builder):
    pass

class BlockDirector:
    def __init__(self,builder):
        self.builder = builder
    
    def construct(self,block,graph,property):
        self.builder.produce_block(block)
        self.builder.produce_graph(graph)
        self.builder.produce_property(property)
        return self.builder.build()