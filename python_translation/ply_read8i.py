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
        w = int(fid.readline().split()[1])  # Read 'comment width %d\n'
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
    J = int(np.log2(w + 1))

    return V, C, J
