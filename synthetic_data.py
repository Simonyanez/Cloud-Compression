# Factory Method
from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
import utils.visualization as vis
import utils.linear as li
import graph.create as cr 
import graph.transforms as tf 

# This is the common generator Factory
class Block_Generator(ABC):
    @abstractmethod
    def block_method(self,size):
        self.size = size
        pass

# These are the method for constructing different types of blocks
class Block_1D_Creator(Block_Generator):
    def block_method(self,size) -> Block:
        return Block_1D(size)

class Block_2D_Creator(Block_Generator):
    def block_method(self,size) -> Block:
        return Block_2D(size)
    
class Block_3D_Creator(Block_Generator):
    def block_method(self,size) -> Block:
        return Block_3D(size)

# These are the products
class Block(ABC):
    def __init__(self,size):
        self.size = size

    @abstractmethod
    def operation(self) -> str:
        pass

# TODO: Implement 1D Block and do initial experiments in 1D data
class Block_1D(Block):
    """
    Every block product will have its own characteristics, depending on the
    operations possibles for them.
    """

    def __init__(self, size: int):
        super().__init__(size)
        self.V = None
        self.A = None
        self._direction = None
        self._sign = None

    def operation(self):
        return f"Block_1D operation with size {self.size}"

    @property
    def direction(self):
        return self._direction

    @direction.setter
    def direction(self, base_vector=None):
        if base_vector is not None:
            self._direction = base_vector
        else:
            self._direction = np.random.randint(2, size=(1, 3))  # From 0 to 1 possible values

    @property
    def sign(self):
        return self._sign

    @sign.setter
    def sign(self, sign=None):
        if sign is not None:
            self._sign = sign
        else:
            self._sign = np.random.randint(-1, 2, size=(1, 3))  # -1, 0, or 1

    def make_linear_gradient(self, num: int):
        assert num < self.size + 1, "The vector is not size compatible"
        positions = list(range(num))
        self.V = np.array([self.direction * position for position in positions])
        gradient = self.sign * self.direction
        zero_array = np.zeros((1, 3),dtype=np.int32)
        relu = np.maximum(zero_array, gradient)
        Y_component = np.int32(np.sum([relu * 255 - gradient * np.round(255 / (num-1)) * self.V[position, :] for position in positions],axis=2)/np.sum(self.direction))
        U_component = np.random.randint(0, 256, (num, 1))
        V_component = np.random.randint(0, 256, (num, 1))

        print(f"These are the sizes {Y_component.shape,U_component.shape,V_component.shape}")
        self.A = np.array([Y_component, U_component, V_component]).T.reshape(num,3)
        self.V = np.array([self.V[:,0,0],self.V[:,0,1],self.V[:,0,2]]).T

class Block_2D(Block):
    def operation(self):
        pass

    def direction(self,base_vectors = None):
        target_vectors = [np.array([1, 1, 1]),
                          np.array([1, 1, 0]),
                          np.array([1, 0, 1]),
                          np.array([0, 1, 1])
                          ]

        assert (base_vectors[0] + base_vectors[1]) in target_vectors, "Not l.i. or bad magnitude" 
        if base_vectors is not None:
            self._direction = base_vectors
        else:
        # In 2D is 2 ortogonal vectors a plane
            # Randomly select a target vector
            target_vector = target_vectors[np.random.randint(0, len(target_vectors))]
            while True:
                base_vectors_0 = np.random.randint(0, target_vector + 1)
                if np.any(base_vectors_0 != 0):
                    break

            # Calculate base_vectors[1] ensuring non-negativity
            base_vectors_1 = target_vector - base_vectors_0

            # Ensure base_vectors[0] and base_vectors[1] are linearly independent
            while not li.is_linearly_independent(base_vectors_0, base_vectors_1):
                base_vectors_0 = np.random.randint(0, target_vector + 1)
                base_vectors_1 = target_vector - base_vectors_0
            base_vectors = [base_vectors_0,base_vectors_1]
            
            self._direction = base_vectors

    # This direction could be filled in any way for experimental purpose
    # should be a fully dense version

class Block_3D(Block):
    def operation(self):
        pass

    def direction(self,base_vectors = None):
        # In 3D is 3 ortogonal vectors a hyperplane.
        pass

    # This direction could be filled in any way for experimental purpose
    # should be a fully dense version

# Extended client code
def client_code(factory: Block_Generator, size: int) -> None:
    block = factory.block_method(size)
    print(block.operation())

    # Set direction and sign
    if isinstance(block, Block_1D):
        block.direction = np.array([[1, 1, 1]])  # Example direction
        block.sign = np.array([[1, -1, 1]])      # Example sign

        # Create the linear gradient
        block.make_linear_gradient(num=size)

        # Visualize
        _,fig=vis.Yvisualization(block.V,block.A)

        # Find borders
        W, _ = cr.compute_graph_unit(block.V)
        DegreeVector = np.sum(W, axis=1)
        sorted_indices = np.argsort(DegreeVector)
        first_threshold = int(0.2 * len(sorted_indices))
        borders_idx = sorted_indices[:first_threshold]
        _,_ = vis.border_visualization(block.V, block.A, borders_idx)
        
        # Selected borders self-looped
        choosed_positions = [0,7]
        choosed_weights = [0,0]
        idx_map = dict(zip(choosed_positions,choosed_weights))
        GFT, Gfreq, Ablockhat = tf.compute_GFT_noQ(W,block.A,idx_closest=idx_map)
        _,_ = vis.border_visualization(block.V, block.A, choosed_positions)
        
        # First base plot
        _= vis.component_visualization(block.V, GFT[:,0], "self-looped",'cg')
        print(f"This is the vector size and minmax {GFT[:,0].shape, np.min(GFT[:,0]),np.max(GFT[:,0])}")
        base_fig = vis.base_plot(GFT[:,0],choosed_weights)

if __name__ == "__main__":
    print("App: Launched with the Block_1D_Creator.")
    client_code(Block_1D_Creator(), 8)
    print("\n")
    plt.show()