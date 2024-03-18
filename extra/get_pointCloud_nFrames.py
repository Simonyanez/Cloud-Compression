# Authors: Eduardo Pavez <eduardo.pavez.carvelli@gmail.com, pavezcar@usc.edu>
# Copyright Eduardo Pavez, University of Southern California, Los Angeles, USA, 05/30/2020
# E. Pavez, B. Girault, A. Ortega, and P. A. Chou.
# "Region adaptive graph Fourier transform for 3D point clouds".
# IEEE International Conference on Image Processing (ICIP), 2020
# https://arxiv.org/abs/2003.01866

def get_pointCloud_nFrames(dataset, sequence):
    """
    Computes the number of frames in a specific dataset and sequence.

    Parameters:
    - dataset (str): Name of the dataset.
    - sequence (str): Name of the sequence.

    Returns:
    - nFrames (int): Number of frames in the specified dataset and sequence.
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

    # Compute the number of frames
    nFrames = endFrame - startFrame + 1
    return nFrames
