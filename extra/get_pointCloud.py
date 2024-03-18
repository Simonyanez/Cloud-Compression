# Authors: Eduardo Pavez <eduardo.pavez.carvelli@gmail.com, pavezcar@usc.edu>
# Copyright Eduardo Pavez, University of Southern California, Los Angeles, USA, 05/30/2020
# E. Pavez, B. Girault, A. Ortega, and P. A. Chou.
# "Region adaptive graph Fourier transform for 3D point clouds".
# IEEE International Conference on Image Processing (ICIP), 2020
# https://arxiv.org/abs/2003.01866

def get_pointCloud(dataset, sequence, frame):
    """
    Loads the point cloud data for a specific dataset, sequence, and frame.

    Parameters:
    - dataset (str): Name of the dataset.
    - sequence (str): Name of the sequence.
    - frame (int): Frame number.

    Returns:
    - V (ndarray): Array of 3D points.
    - C (ndarray): Array of RGB colors.
    - J (int): Resolution of voxel (only for 'MVUB' dataset).
    """
    # Define start and end frames based on the dataset and sequence
    if dataset == '8iVFBv2':
        if sequence == 'redandblack':
            startFrame, endFrame = 1450, 1749
        elif sequence == 'soldier':
            startFrame, endFrame = 536, 835
        elif sequence == 'longdress':
            startFrame, endFrame = 1051, 1350
        elif sequence == 'loot':
            startFrame, endFrame = 1000, 1299
        else:
            # Display a warning if the sequence doesn't belong to the dataset
            warning_msg = 'The provided sequence {} does not belong to dataset {}'
            print(warning_msg.format(sequence, dataset))
    elif dataset == 'MVUB':
        if sequence == 'andrew9':
            startFrame, endFrame = 0, 317
        elif sequence == 'david9':
            startFrame, endFrame = 0, 215
        elif sequence == 'phil9':
            startFrame, endFrame = 0, 244
        elif sequence == 'ricardo9':
            startFrame, endFrame = 0, 215
        elif sequence == 'sarah9':
            startFrame, endFrame = 0, 206
        else:
            # Display a warning if the sequence doesn't belong to the dataset
            warning_msg = 'The provided sequence {} does not belong to dataset {}'
            print(warning_msg.format(sequence, dataset))
        J = 9
    else:
        # Display a warning if the dataset is not recognized
        warning_msg = '{} is not a proper dataset'
        print(warning_msg.format(dataset))

    # Calculate the frame number to be read
    getframe = startFrame - 1 + frame
    if getframe > endFrame:
        # Display a warning if the frame number doesn't exist
        warning_msg = 'The frame number {} does not exist'
        print(warning_msg.format(frame))
        return None, None, None

    # Load point cloud data based on the dataset and sequence
    if dataset == '8iVFBv2':
        filename = f'8iVFBv2/{sequence}/Ply/{sequence}_vox10_{getframe:04d}.ply'
        V, C, J = ply_read8i(filename)
    elif dataset == 'MVUB':
        filename = f'MVUB/{sequence}/ply/frame{getframe:04d}.ply'
        V, C = ply_readMVUB(filename)

    return V, C, J if dataset == 'MVUB' else None
