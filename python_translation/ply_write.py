def ply_write(filename, V, C, F=None):
    """
    Write the mesh to a ply file.

    Args:
    filename (str): Name of ply file in current directory.
    V (ndarray): Nx3 matrix of 3D coordinates of total N vertices.
    C (ndarray): Nx3 matrix of R,G,B colors on the N vertices.
    F (ndarray, optional): Mx3 matrix of vertex indices of M triangles.

    Returns:
    None
    """
    N = V.shape[0]
    M = F.shape[0] if F is not None else 0

    with open(filename, 'w', encoding='UTF-8') as fid:
        fid.write('ply\n')
        fid.write('format ascii 1.0\n')
        fid.write(f'element vertex {N}\n')
        fid.write('property float x\n')
        fid.write('property float y\n')
        fid.write('property float z\n')
        fid.write('property uchar red\n')
        fid.write('property uchar green\n')
        fid.write('property uchar blue\n')
        if M > 0:
            fid.write(f'element face {M}\n')
            fid.write('property list uchar int vertex_index\n')
        fid.write('end_header\n')
        
        if F is not None:
            for i in range(N):
                fid.write(f'{V[i, 0]:.6f} {V[i, 1]:.6f} {V[i, 2]:.6f} {C[i, 0]} {C[i, 1]} {C[i, 2]}\n')
        else:
            for i in range(N):
                fid.write(f'{V[i, 0]:.6f} {V[i, 1]:.6f} {V[i, 2]:.6f} {C[i, 0]} {C[i, 1]} {C[i, 2]}\n')

        if F is not None:
            for i in range(M):
                fid.write(f'3 {F[i, 0]} {F[i, 1]} {F[i, 2]}\n')
