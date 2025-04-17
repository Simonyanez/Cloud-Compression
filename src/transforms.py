

# Other imports
import matplotlib.pyplot as plt
import numpy as np
# from graph.create import *
from graph import *
from objects import *
from visualization import *
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from scipy.linalg import fractional_matrix_power, eigh
#from scipy.linalg import eigh
import logging
logging.basicConfig(filename="logs/graph.log", filemode="w", level=logging.DEBUG)
logger = logging.getLogger(__name__)

from typing import Optional

class GFT():
    # FIXME: Disconnected components not working correctly
    def __init__(self):
        self.visualizer = Visualizer()
        # TODO: Give parameters
        pass

    def __call__(self, graph: Graph, block: Block, Q: Optional[np.ndarray] = None) -> tuple[np.ndarray, np.ndarray]:
        self.graph = graph
        self.block = block
        GFT_matrix, Coeffs = self._exec(Q)
        return GFT_matrix, Coeffs
    
    def _exec(self, Q):
        n_components, labels = self._check_connected()
        if n_components > 1:
            logger.debug("Found disconnected graph")
            GFT_matrix, Coeffs = self._process_disconnected(n_components,labels)
        else:
            print("Processing as connected now")
            GFT_matrix, Coeffs = self._process_connected(Q)
        return GFT_matrix, Coeffs

    def _check_connected(self) -> tuple[np.ndarray, np.ndarray]:
        Adj = self.graph.weights
        Adj_sparse = csr_matrix(Adj)
        num_components, labels = connected_components(Adj_sparse, directed=False, return_labels=True)
        return num_components, labels
    
    def _process_disconnected(self, num_components, labels):
        GFT_processor = GFT()
        Q_norm = np.zeros((num_components, num_components))
        N = self.graph.weights.shape[0]
        U = None
        Vmean = np.zeros((num_components, 3))
        isDC = np.zeros(N, dtype=bool)
        i = 0
        # FIXME isn't pos and component the same?
        for pos, component in enumerate(range(num_components)):
            subgraph_indexes = np.where(labels == component)[0]
            Q_norm[pos, pos] = len(subgraph_indexes)
            subgraph, subblock = self._create_subobjects(subgraph_indexes)
            GFT_sub, _ = GFT_processor(subgraph, subblock)  
            U = self._fill_disconnected_transform(subgraph_indexes, GFT_sub, U)
            isDC[i] = 1
            print(len(subgraph_indexes), subgraph_indexes)
            i += len(subgraph_indexes)
            Vmean[component, :] = np.mean(self.block.Vblock[subgraph_indexes, :], axis = 0)
        Coeffs = U.T @ self.block.Ablock
        self.visualizer.visualize_gft(U, title="Disconnected GFT reordered")
        self.visualizer.visualize_coeffs(Coeffs, title="Disconnected coeffs")
        Coeffs_low, Coeffs_high = Coeffs[isDC,:], Coeffs[np.logical_not(isDC),:]
        self.visualizer.visualize_coeffs(Coeffs_low, title = "Low disconnected")
        self.visualizer.visualize_coeffs(Coeffs_high,title = "High disconnected" )
        meangraph, meanblock = self._create_meanobjects(Vmean)
        self.visualizer(meangraph, meanblock)
        self.visualizer.visualize_block()
        self.visualizer.visualize_graph()
        self.visualizer.visualize_gft(Q_norm, title="Q_norm")
        GFT_mean, _ = GFT_processor(meangraph, meanblock, Q_norm)
        print(f"This is coeffs low {Coeffs_low}")
        # freq_order = np.argsort(Gfreq_mean)
        # print(f"This is reordered Gfreq {freq_order}")
        # GFT_mean_sorted = GFT_mean[:, freq_order]
        Coeffs_low_fixed = GFT_mean.T @ Coeffs_low
        # Apply reordered transform
        
        print(f"This is coeffs low fixed {Coeffs_low_fixed}")
        Coeffs_fix = np.concatenate([Coeffs_low_fixed, Coeffs_high])
        self.visualizer.visualize_gft(GFT_mean, title="mean GFT")
        # Coeffs_fix = np.concatenate([GFT_mean.T @ Coeffs_low, Coeffs_high]) # Mean coeff transform with its GFT mean structure
        self.visualizer.visualize_coeffs(GFT_mean.T @ Coeffs_low)
        self.visualizer.visualize_coeffs(Coeffs_fix, title="Coeffs Fixed")
        return U, Coeffs_fix

    def _create_subobjects(self, subgraph_indexes) -> tuple[Graph, Block]:
        Asubblock = self.block.Ablock[subgraph_indexes,:]
        Vsubblock = self.block.Vblock[subgraph_indexes,:]
        W = self.graph.weights
        W_sub = W[subgraph_indexes, :][:, subgraph_indexes]
        aux_tuple = (-1,-1)
        subblock = Block((-1,-1), block_num=-1)
        subblock._init_auxiliary(Vblock=Vsubblock, Ablock =Asubblock, subidxs=subgraph_indexes)
        subgraph = Graph(subblock.id)
        subgraph._init_data(weights=W_sub,edges=[]) # Creates a subgraph without connections
        return subgraph, subblock

    def _create_meanobjects(self, Vmean: np.ndarray):
        print(f"Vmean: {Vmean}")
        meanblock = Block(idxs=(-1,-1), block_num=-2)
        meanblock._init_auxiliary(Vblock=Vmean, Ablock=Vmean, subidxs=None) # Use Vmean auxiliary for attributes only for calling. Coeffs will be useless
        meangraph = StructuralGraph(meanblock.id)
        meangraph._init_data(Vmean, threshold= np.inf)
        return meangraph, meanblock
    
    def _fill_disconnected_transform(self, subgraph_indexes: np.ndarray, GFT_matrix: np.ndarray, U: np.ndarray):
        """
        Fill an auxiliary
        """
        num_nodes = self.graph.weights.shape[0]
        Utmp    = np.zeros((num_nodes, len(subgraph_indexes)))
        Utmp[subgraph_indexes, :] = GFT_matrix
        if U is None:
            return Utmp
        return np.concatenate([U, Utmp], axis = 1)

    def reorder_coeffs_by_vmean(self, Vmean: np.ndarray, Coeffs_low_fixed: np.ndarray) -> np.ndarray:
        """
        Reorder the rows of Coeffs_low_fixed to match the spatial order in Vmean.
        This fixes random flips/swaps from spectral decomposition.
        """
        from sklearn.preprocessing import normalize
        from scipy.optimize import linear_sum_assignment

        Vmean_norm = normalize(Vmean)
        coeffs_norm = normalize(Coeffs_low_fixed)

        # Compute cosine similarity
        similarity = Vmean_norm @ coeffs_norm.T  # shape: (num_components, num_components)
        cost = -np.abs(similarity)
        row_ind, col_ind = linear_sum_assignment(cost)

        # Reorder rows
        Coeffs_low_sorted = Coeffs_low_fixed[col_ind]
        return Coeffs_low_sorted

    def _process_connected(self, Q: Optional[np.ndarray] = None):
            """
            Process a connected block to compute the GFT matrix and coefficients.

            Args:
                Q (Optional[np.ndarray]): The weighting matrix. Defaults to the identity matrix.

            Returns:
                tuple[np.ndarray, np.ndarray]: The GFT matrix and the coefficients.
            """
            if Q is None:
                n = self.block.Ablock.shape[0]
                Q = np.identity(n)

            # Handle 1-point blocks
            if Q.shape[0] == 1:
                GFT_matrix = np.array([[1.0]])
                Coeffs = self.block.Ablock
                return GFT_matrix, Coeffs
            try:
                Qm = fractional_matrix_power(Q, -0.5)
                print(f"This is Qm {Qm}")
                L = self._get_laplacian(Qm)
                print(f"This is L {L}")
                GFT_matrix, _ = self._compute_GFT(L)
                A = self.block.Ablock
                Coeffs = GFT_matrix.T @ A
                return GFT_matrix, Coeffs
            except:
                logger.debug(f"Q is {Q.shape}  --> {Q}")

    def _get_laplacian(self, Qm: np.ndarray) -> np.ndarray:
        """
        Compute the normalized Laplacian matrix.

        Args:
            Qm (np.ndarray): The square root of the inverse weighting matrix.

        Returns:
            np.ndarray: The normalized Laplacian matrix.
        """
        A = self.graph.weights  # Adjacency matrix
        D = np.diag(np.sum(A, axis=0))  # Degree matrix
        C = np.diag(np.diag(A))  # Self-loops matrix
        L = D - A + C
        L_q = Qm @ L @ Qm
        return L_q

    def _compute_GFT(self, L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the Graph Fourier Transform (GFT) matrix and frequencies.

        Args:
            L (np.ndarray): The Laplacian matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: The GFT matrix and the frequencies.
        """
        if L.shape[0] == 1:
            # Handle 1-point blocks
            GFT_matrix = np.array([[1.0]])
            Gfreq = np.array([0.0])
        else:
            # Compute eigenvalues and eigenvectors for larger blocks
            eigvals, eigvecs = np.linalg.eigh(L)
            print(f"This are the eig vals {eigvals}")
            eigvals_idxsorted = np.argsort(eigvals)  # Changed from abs value
            print(f"Eig vals sorted {eigvals_idxsorted}")
            GFT_matrix = eigvecs[:, eigvals_idxsorted]
            print(f"This is pre fix GFT {GFT_matrix}")
            # Ensure the first eigenvector is positive
            for i in range(GFT_matrix.shape[0]):
                if eigvals_idxsorted[i] < 0:
                    print(eigvals_idxsorted[i])

                if GFT_matrix[i, 0] < 0:
                    GFT_matrix[i, :] = GFT_matrix[i, :] * (-1)
            print(f"This is post-fix GFT {GFT_matrix}")
            Gfreq = eigvals[eigvals_idxsorted]
            print(f"Final gfreqs {Gfreq}")
        return GFT_matrix, Gfreq


        

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


if __name__ == "__main__":
    from create import *
    from . import visualization as vis
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