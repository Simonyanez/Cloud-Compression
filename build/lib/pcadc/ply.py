import numpy as np

def ply_read8i(filename):
    """
    Read vertices and their colors from a ply file.

    Args:
    filename (str): Name of ply file in current directory.

    Returns:
    tuple: Tuple containing vertices, colors, and voxel depth.
    """
    with open(filename, 'r', encoding='UTF-8') as fid:
        fid.readline()  # Read 'ply\n'
        fid.readline()  # Read 'format ascii 1.0\n'
        fid.readline()  # Read 'comment Version 2, Copyright 2017, 8i Labs, Inc.\n'
        fid.readline()  # Read 'comment frame_to_world_scale %g\n'
        fid.readline()  # Read 'comment frame_to_world_translation %g %g %g\n'
        w = int(fid.readline().split()[2])  # Read 'comment width %d\n'
        N = int(fid.readline().split()[2])  # Read 'element vertex %d\n'

        fid.readline()  # Read 'property float x\n'
        fid.readline()  # Read 'property float y\n'
        fid.readline()  # Read 'property float z\n'
        fid.readline()  # Read 'property uchar red\n'
        fid.readline()  # Read 'property uchar green\n'
        fid.readline()  # Read 'property uchar blue\n'
        fid.readline()  # Read 'end_header'

        lines = fid.readlines()
        A = [list(map(float, line.strip().split())) for line in lines]

    V = np.array([[row[0], row[1], row[2]] for row in A])           # As numpy array
    C = np.array([[row[3], row[4], row[5]] for row in A])           # As numpy array
    J = int(np.log2(w + 1))

    return V, C, J

def ply_readMVUB(filename):
    """
    Read vertices and their colors from a ply file.

    Args:
    filename (str): Name of ply file in current directory.

    Returns:
    tuple: Tuple containing vertices and colors.
    """
    with open(filename, 'r', encoding='UTF-8') as fid:
        fid.readline()  # Read 'ply\n'
        fid.readline()  # Read 'format ascii 1.0\n'
        N = int(fid.readline().split()[2])  # Read 'element vertex %d\n'

        fid.readline()  # Read 'property float x\n'
        fid.readline()  # Read 'property float y\n'
        fid.readline()  # Read 'property float z\n'
        fid.readline()  # Read 'property uchar red\n'
        fid.readline()  # Read 'property uchar green\n'
        fid.readline()  # Read 'property uchar blue\n'

        lines = fid.readlines()
        A = [list(map(float, line.strip().split())) for line in lines]

    V = [[row[0], row[1], row[2]] for row in A]
    C = [[row[3], row[4], row[5]] for row in A]

    return V, C

def ply_write(filename, V, C, F=None):
    """
    Write the mesh to a ply file with integer vertex coordinates and colors.

    Args:
    filename (str): Name of ply file in current directory.
    V (ndarray): Nx3 matrix of 3D coordinates of total N vertices (integers).
    C (ndarray): Nx3 matrix of R,G,B colors on the N vertices (integers).
    F (ndarray, optional): Mx3 matrix of vertex indices of M triangles (integers).

    Returns:
    None
    """
    N = V.shape[0]
    M = F.shape[0] if F is not None else 0

    with open(filename, 'w', encoding='UTF-8') as fid:
        fid.write('ply\n')
        fid.write('format ascii 1.0\n')
        fid.write(f'element vertex {N}\n')
        fid.write('property int x\n')
        fid.write('property int y\n')
        fid.write('property int z\n')
        fid.write('property uchar red\n')
        fid.write('property uchar green\n')
        fid.write('property uchar blue\n')
        if M > 0:
            fid.write(f'element face {M}\n')
            fid.write('property list uchar int vertex_index\n')
        fid.write('end_header\n')
        
        for i in range(N):
            fid.write(f'{V[i, 0]} {V[i, 1]} {V[i, 2]} {C[i, 0]} {C[i, 1]} {C[i, 2]}\n')

        if F is not None:
            for i in range(M):
                fid.write(f'3 {F[i, 0]} {F[i, 1]} {F[i, 2]}\n')