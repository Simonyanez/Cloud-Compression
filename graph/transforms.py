
# Importing Parent Folder to Path
import os
main_folder = os.getcwd()
import sys
sys.path.insert(0, main_folder)

# Other imports
import matplotlib.pyplot as plt
import numpy as np
from graph.create import *
from graph.graph import *
from sklearn.preprocessing import normalize
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power, eigh
#from scipy.linalg import eigh

class GFT():
    def __init__(self):
        # TODO: Give parameters
        pass

    def __call__(self, graph: Graph, block: Block):
        self.graph = graph
        self.block = block
        self._exec()

    def _exec(self):
        n_components, labels = self._check_connected()
        if n_components > 1:
            Coeffs = self._process_disconnected(n_components,labels)
        else:
            Coeffs = self._process_connected()
        return Coeffs

    def _check_connected(self) -> tuple[np.ndarray, np.ndarray]:
        Adj = self.graph.weights
        Adj_sparse = csr_matrix(Adj)
        num_components, labels = connected_components(Adj_sparse, directed=False, return_labels=True)
        return num_components, labels
    
    def _process_disconnected(self, num_components, labels):
        GFT_processor = GFT()
        Q_norm = np.zeros((num_components, num_components))
        for pos, component in enumerate(range(num_components)):
            subgraph_indexes = np.where(labels == component)[0]
            Q_norm[pos, pos] = len(subgraph_indexes)
            subgraph, subblock = self._create_subobjects(subgraph_indexes)
            GFT_processor(subgraph, subblock)            
        pass

    def _create_subobjects(self, subgraph_indexes) -> tuple[Graph, Block]:
        Ablock = self.block.Ablock
        Vblock = self.block.Vblock
        W = self.graph.weights
        W_sub = W[subgraph_indexes, :][:, subgraph_indexes]
        subgraph = Graph(W_sub) # Creates a subgraph without connections
        subblock = Block(Vblock, Ablock, subgraph_indexes)
        return subgraph, subblock

    def _process_connected(self, Q: Optional[np.ndarray] = None):
        if Q is None:
            n = self.block.Ablock.shape[0]
            Q = np.identity(n)

        Qm = fractional_matrix_power(Q, -0.5)
        L = self._get_laplacian(Qm)
        GFT_matrix, _ = self._compute_GFT(L)
        A = self.block.Ablock
        Coeffs = GFT_matrix.T @ A
        return Coeffs

    def _get_laplacian(self, Qm: np.ndarray) -> np.ndarray:
        A = self.graph.weights         # Adjacency matrix
        D = np.diag(np.sum(W, axis=0)) # Degree matrix
        C = np.diag(np.diag(W))        # Self-loops matrix
        L = D - A + C
        L_q = Qm @ L @ Qm
        return L_q


    def _compute_GFT(self, L: np.ndarray) -> np.ndarray:
        GFT_matrix = np.array([1.0])
        Gfreq = np.array([0.0])
        if L.shape[0] > 1:
            eigvals, eigvecs = np.linalg.eigh(L)
            eigvals_idxsorted = np.argsort(np.abs(eigvals))
            GFT_matrix = eigvecs[:, eigvals_idxsorted]
            # TODO: No indent implementation
            for i in range(GFT_matrix.shape[0]):
                if GFT_matrix[i,0] < 0:
                    GFT_matrix[i,:] = GFT_matrix[i,:]*(-1)
            Gfreq = eigvals[eigvals_idxsorted]
        return GFT_matrix, Gfreq


class Transformer():
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwds):
        pass
        
    def compute_GFT(self, A, Adj, idx_closest):
        if Adj.shape[0] > 1:
            if idx_closest is not None:
                L = w2l(Adj, idx_closest, iter = iter)
            else:
                L = w2l(Adj, iter = iter)
            if debug:
                print(f"L: {L}")
            # L is normalized by the way it's build
            D, GFT = np.linalg.eigh(L) # D eigen values and GFT eigenvectors
            idxSorted = np.argsort(np.abs(D))      # Order of the eigenvalues. # np.abs(D) 
            GFT = GFT[:,idxSorted]         # GFT ordered by eigenvalues order first less

            for i in range(GFT.shape[0]):
                if GFT[i,0] < 0:
                    GFT[i,:] =  GFT[i,:]*(-1) 
            # GFT[:,0] = np.abs(GFT[:,0])
            # GFT = GFT.T         # Because the matrix that do the transform is this one
            Gfreq = np.sort(D)
    
            Gfreq[0] = np.abs(Gfreq[0])
            
            Ahat = np.matmul(GFT.T, A)      # @ is a shortcut for matmul, yet i dont like it
            if np.iscomplexobj(Ahat) and iter is not None:
                print(f"This is the block that has complex values {iter}")

        else:  # 1D-case, DC only
            GFT = np.array([1.0])
            Gfreq = np.array([0.0])
            Ahat = np.matmul(GFT.T, A)

        return GFT, Gfreq, Ahat

def w2l(W, idx_closest_map=None, iter=None):
    """
    Convert weight matrix to Laplacian matrix.

    Args:
        W (numpy.ndarray): Weight matrix.
        idx_closest (numpy.ndarray): Indices of the closest points (optional).

    Returns:
        L (numpy.ndarray): Laplacian matrix.
    """
    sz_W = W.shape
    C = np.zeros(sz_W)

    if np.any(W < 0):
        if iter is not None:
            print(f"This is the block that has complex values {iter}")
        # Handle negative weights differently if needed

    if idx_closest_map is not None:
        # TODO: Make this from structure and not hardcoded
        for idx in idx_closest_map.keys():
            C[idx, idx] = idx_closest_map[idx]
            #W[idx,idx] = idx_closest_map[idx]

    D = np.diag(np.sum(W, axis=0))

    # Be careful that C = np.diag(np.diag(W)) if the self-loops are originally at the structure of the graph

    L = D - W + np.diag(np.diag(W)) + C
    return L

def check_connected(W):
    """
    Check if Graph is connected so it can be splitted in its results
    """
    # Convert the adjacency matrix to a sparse matrix (for efficiency)
    W_sparse = csr_matrix(W)

    # Use connected_components to find the number of connected components and labels for each node
    num_components, labels = connected_components(W_sparse, directed=False, return_labels=True)
    return num_components, labels

def iterative_GFT(W, A, V, idx_map=None, debug=False):
    """
    Compute the Graph Fourier Transform (GFT) iteratively for each disconnected component of the graph.
    """
    num_components, labels = check_connected(W)
    if num_components == 1:
        GFT, Gfreq, Coeff = compute_GFT_noQ(W, A, idx_closest=idx_map,debug=debug)
        return GFT, Gfreq, Coeff
    
    
    GFT = []
    Gfreq = []
    Ahat = []
    DC_pos = []  # To store the indices of nodes for each component
    U = []
    isDC = []
    V_new = np.zeros((num_components, 3))  # Assuming V has shape (n, 3) for 3D coordinates
    Q_norm = np.zeros((num_components,num_components))
    for pos,component in enumerate(range(num_components)):  # Loop through each component (0, 1, ..., num_components-1)
        # Get the indices of nodes belonging to the current component
        component_indices = np.where(labels == component)[0]
        Q_norm[pos,pos] = len(component_indices)
        # Create the subgraph (W_curr and A_curr) for the current component
        W_curr = W[component_indices, :][:, component_indices]  # W_curr is subgraph for the component
        A_curr = A[component_indices, :]  # A_curr is the signal matrix for the component
        
        DC_pos.append(component_indices[0])
        
        # Compute GFT for this subgraph
        
        GFT_curr, Gfreq_curr, Ahat_curr = compute_GFT_noQ(W_curr, A_curr,idx_closest=None,debug=debug)  # Assume this function is implemented
        
        Utmp = np.zeros((W.shape[0], len(component_indices)))
        Utmp[component_indices, :] = GFT_curr
        U.append(Utmp)  # Add the subgraph GFT to the U list
        
        # Create isDC array, which marks the first node as DC
        isDCtmp = np.zeros(len(component_indices), dtype=bool)
        isDCtmp[0] = 1  # First node in the component is DC
        isDC.append(isDCtmp)  # Append to the isDC list
        
        # Append results
        GFT.append(GFT_curr)
        Gfreq.append(Gfreq_curr)
        Ahat.append(Ahat_curr)

        # Average position of connected points per connection
        V_new[component, :] = np.mean(V[component_indices, :], axis=0)
        
        
    # Convert lists to numpy arrays
    U = np.concatenate(U, axis=1)  # Concatenate along axis 1 to form the full U matrix
    isDC = np.concatenate(isDC, axis=0)  # Concatenate isDC for all components
    Ahat_1 = U.T @ A
    Ahat_low = Ahat_1[isDC, :]
    Ahat_high = Ahat_1[np.logical_not(isDC), :]

    # Complete graph creation
    Wnew = complete_graph(V_new)
    
    # Assuming compute_GFT_noQ works and returns the appropriate GFT for Wnew
    # A[:Wnew.shape[0]]
    
    GFT_new, Gfreq_new = compute_GFT(Wnew,Q_norm, debug=False)  # Use Wnew's shape for A

    Gfreq = np.hstack(Gfreq)
    Coeff = np.concatenate([GFT_new.T @ Ahat_low, Ahat_high], axis=0)
    Gfreq = np.concatenate([Gfreq_new, Gfreq[np.logical_not(isDC)]], axis=0)

    return GFT_new, Gfreq, Coeff

def compute_GFT_noQ(Adj, A, idx_closest=None, iter=None, debug=False):
    """
    Compute the Graph Fourier Transform (GFT) without using the quality matrix.

    Parameters:
        Adj (numpy.ndarray): Adjacency matrix of the graph.
        A (numpy.ndarray): Attribute matrix.
        idx_closest (numpy.ndarray or None): Index of the closest points (optional)
    if Adj.shape[0] > 1:
        if idx_closest is not None:
            L = w2l(Adj, idx_closest, iter = iter)
        else:
            L = w2l(Adj, iter = iter)
        if debug:
            print(f"L: {L}")
        # L is normalized by the way it's build
        D, GFT = np.linalg.eigh(L) # D eigen values and GFT eigenvectors
        idxSorted = np.argsort(np.abs(D))      # Order of the eigenvalues. # np.abs(D) 
        GFT = GFT[:,idxSorted]         # GFT ordered by eigenvalues order first less

        for i in range(GFT.shape[0]):
            if GFT[i,0] < 0:
                GFT[i,:] =  GFT[i,:]*(-1) 
        # GFT[:,0] = np.abs(GFT[:,0])
        # GFT = GFT.T         # Because the matrix that do the transform is this one
        Gfreq = np.sort(D)
 
        Gfreq[0] = np.abs(Gfreq[0])
        
        Ahat = np.matmul(GFT.T, A)      # @ is a shortcut for matmul, yet i dont like it
        if np.iscomplexobj(Ahat) and iter is not None:
            print(f"This is the block that has complex values {iter}")

    else:  # 1D-case, DC only
        GFT = np.array([1.0])
        Gfreq = np.array([0.0])
        Ahat = np.matmul(GFT.T, A)

    return GFT, Gfreq, Ahat.

    Returns:
        numpy.ndarray: Graph Fourier Transform.
        numpy.ndarray: Eigenvalues of the Laplacian matrix (sorted in ascending order).
        numpy.ndarray: Transformed attribute matrix.
    """

    if Adj.shape[0] > 1:
        if idx_closest is not None:
            L = w2l(Adj, idx_closest, iter = iter)
        else:
            L = w2l(Adj, iter = iter)
        if debug:
            print(f"L: {L}")
        # L is normalized by the way it's build
        D, GFT = np.linalg.eigh(L) # D eigen values and GFT eigenvectors
        idxSorted = np.argsort(np.abs(D))      # Order of the eigenvalues. # np.abs(D) 
        GFT = GFT[:,idxSorted]         # GFT ordered by eigenvalues order first less

        for i in range(GFT.shape[0]):
            if GFT[i,0] < 0:
                GFT[i,:] =  GFT[i,:]*(-1) 
        # GFT[:,0] = np.abs(GFT[:,0])
        # GFT = GFT.T         # Because the matrix that do the transform is this one
        Gfreq = np.sort(D)
 
        Gfreq[0] = np.abs(Gfreq[0])
        
        Ahat = np.matmul(GFT.T, A)      # @ is a shortcut for matmul, yet i dont like it
        if np.iscomplexobj(Ahat) and iter is not None:
            print(f"This is the block that has complex values {iter}")

    else:  # 1D-case, DC only
        GFT = np.array([1.0])
        Gfreq = np.array([0.0])
        Ahat = np.matmul(GFT.T, A)

    return GFT, Gfreq, Ahat

def compute_iGFT_noQ(Adj, Ahat_val, idx_closest=None):
    if Adj.shape[0] > 1:
        if idx_closest is not None:
            L = w2l(Adj, idx_closest, iter = iter)
        else:
            L = w2l(Adj, iter = iter)
        # L is normalized by the way it's build
        D, GFT = np.linalg.eigh(L) # D eigen values and GFT eigenvectors
        idxSorted = np.argsort(np.abs(D))      # Order of the eigenvalues. # np.abs(D) 
        GFT = GFT[:,idxSorted]         # GFT ordered by eigenvalues order first less

        for i in range(GFT.shape[0]):
            if GFT[i,0] < 0:
                GFT[i,:] =  GFT[i,:]*(-1) 
    GFT_inv = np.linalg.inv(GFT.T)

    Arec = np.matmul(GFT_inv, Ahat_val)
    return GFT_inv, Arec

def compute_GFT(Adj, Q, debug=False):
    """
    Compute the Graph Fourier Transform (GFT) using the adjacency matrix and quality matrix Q.

    Parameters:
        Adj (numpy.ndarray): Adjacency matrix of the graph.
        Q (numpy.ndarray): Quality matrix (assumed to be a vector).

    Returns:
        numpy.ndarray: Graph Fourier Transform (GFT).
        numpy.ndarray: Eigenvalues of the Laplacian matrix (sorted in ascending order).
    """
    # Qm is the diagonal matrix with 1/sqrt(Q) on the diagonal
    # Q = np.maximum(Q, 1e-8)  # Avoid division by zero and negative square roots
    Qm = fractional_matrix_power(Q,-0.5)  # Q is assumed to be a vector, so Q^(-1/2) gives the inverse square root of Q
    # Compute the Laplacian matrix using Qm
    L = w2l(Adj)
    L_q = Qm @ L @ Qm  # Matrix multiplication
    if debug:
        print(f"This is the Adjacency {Adj} \n Q original matrix {Q} \n Q normalization matriz {Qm} \n Normalized Laplacian {L_q}")
        print(f'Inverse check {fractional_matrix_power(Qm,-2)}')
    # Eigenvalue decomposition of the Laplacian matrix L
    if Adj.shape[0] > 1:
        # Compute the eigenvalues and eigenvectors
        D, GFT = np.linalg.eigh(L_q)  # D is eigenvalues, GFT is eigenvectors
        idxSorted = np.argsort(np.abs(D))  # Sort eigenvalues in ascending order
        GFT = GFT[:, idxSorted]  # Sort eigenvectors accordingly
        
        # Ensure positive eigenvectors (if necessary)
        for i in range(GFT.shape[0]):
            if GFT[i,0] < 0:
                GFT[i,:] =  GFT[i,:]*(-1) 
        

        # Gfreq corresponds to the eigenvalues
        Gfreq = np.sort(D)
    else:  # Handle the case where the graph is just a single point
        GFT = np.array([1.0])
        Gfreq = np.array([0.0])
    
    if debug:
        print(f"This is the Adjacency {Adj} \n Q normalization matriz {Qm} \n Normalized Laplacian {L_q} \n GFT values {GFT}")

    return GFT, Gfreq

def eig_vector_rotation(repeated_eig_pos, repeated_eig, GFT):
    random_matrix = np.random.randn(repeated_eig.shape[1], repeated_eig.shape[1])
    Q, _ = np.linalg.qr(random_matrix)
    
    # Apply rotation to the eigenvectors
    rotated_eigenvectors = repeated_eig @ Q
    
    # Update the GFT matrix with rotated eigenvectors
    GFT_new = GFT.copy()  # Make a copy to avoid modifying the original GFT
    GFT_new[:, repeated_eig_pos] = rotated_eigenvectors
    
    # Compute new coefficients
    Coeff_new = GFT_new.T @ Ablock
    return GFT_new, Coeff_new

def optimize_rotation(Gfreq, GFT, Ablock, max_iter=100, tol=1e-6):
    Coeff = GFT.T @ Ablock
    Coeff_new = Coeff.copy()  # Start with the original Coeffs
    
    uniques = np.unique(Gfreq)
    
    for unique in uniques: 
        condition = Gfreq == unique
        if np.sum(condition) > 1:  # Only proceed if there are repeated eigenvectors
            repeated_eig_pos = np.argwhere(condition).flatten()
            repeated_eig = GFT[:, repeated_eig_pos]
            
            # Initialize the iteration counter
            iter_count = 0
            while iter_count < max_iter:
                # Update the eigenvectors by rotating them
                GFT_new, Coeff_new = eig_vector_rotation(repeated_eig_pos, repeated_eig, GFT)
                
                # Check if the new coefficients are better
                if np.sum(np.abs(Coeff_new[repeated_eig_pos])) < np.sum(np.abs(Coeff[repeated_eig_pos])):
                    # Accept the new GFT and coefficients
                    GFT = GFT_new
                    Coeff = Coeff_new
                
                # If not improved, rotate again
                iter_count += 1
                
                # If no improvement after max iterations, stop
                if iter_count >= max_iter:
                    print(f"Reached max iterations for eigenvalue {unique}.")
                    break
        else:
            continue  # If only one eigenvector for this eigenvalue, skip
    
    return GFT, Coeff_new


if __name__ == "__main__":
    from create import *
    from utils import visualization as vis
    V = np.load('V_longdress.npy')
    C_rgb = np.load('C_longdress.npy')
    indexes = get_block_indexes(V,8)
    Vblock = V[indexes[1906][0]: indexes[1906][1]]
    Ablock = C_rgb[indexes[1906][0]: indexes[1906][1]]
    print(Vblock.shape)
    #vis.visualization(Vblock, Ablock, 'None', 'None', 'None')
    W, edge = compute_graph_MSR(Vblock)
    GFT, Gfreq, Ahat = iterative_GFT(W, Ablock, Vblock, 1906)
    plt.matshow(GFT)
    plt.title('GFT transform for normalized complete graph')
    # plt.plot(Ahat)
    plt.show()