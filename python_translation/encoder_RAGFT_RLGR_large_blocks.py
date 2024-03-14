import numpy as np
from scipy.linalg import norm
from scipy.io import savemat
from time import time
from raht import raht_forward
from rlgr import RLGR_encode
from utils import get_point_cloud, rgb_to_yuv

def encoder_RAGFT_RLGR_large_blocks(dataset, sequence, b, color_step, experiment):
    T = len(sequence)
    n_steps = len(color_step)
    bytes_array = np.zeros((T, n_steps))
    mse_array = np.zeros((T, n_steps))
    n_voxels = np.zeros(T)
    time_array = np.zeros(T)

    param = {'bsize': b}

    for frame in range(T):
        start_time = time()
        V, Crgb, J = get_point_cloud(dataset, sequence[frame])
        N = V.shape[0]
        n_voxels[frame] = N
        C = rgb_to_yuv(Crgb)

        if experiment.startswith('exp_zhang'):
            param['isMultiLevel'] = False
        else:
            param['isMultiLevel'] = True

        Coeff, Gfreq, weights = raht_forward(C, param)
        Y = Coeff[:, 0]

        for i, step in enumerate(color_step):
            # Quantize coefficients
            Coeff_enc = np.round(Coeff / step)
            Y_hat = Coeff_enc[:, 0] * step

            # Compute squared error
            mse_array[frame, i] = norm(Y - Y_hat)**2 / (N * 255**2)

            # Encode coefficients using RLGR
            nbytes_Y, _ = RLGR_encode(Coeff_enc[:, 0])
            nbytes_U, _ = RLGR_encode(Coeff_enc[:, 1])
            nbytes_V, _ = RLGR_encode(Coeff_enc[:, 2])
            bytes_array[frame, i] = nbytes_Y + nbytes_U + nbytes_V

        end_time = time()
        time_array[frame] = end_time - start_time

        print(f"{dataset}/{sequence[frame]}/\t {experiment}\t {time_array[frame]}\t {frame}\t {T}")

    folder = f"RA-GFT/results/{dataset}/{sequence}/"
    filename = f"{folder}RA-GFT_{experiment}.mat"
    savemat(filename, {'MSE': mse_array, 'bytes': bytes_array, 'Nvox': n_voxels,
                       'b': b, 'colorStep': color_step, 'time': time_array})

# Example usage:
dataset = 'example_dataset'
sequence = ['sequence1', 'sequence2', 'sequence3']  # Example sequence names
b = 8  # Block size
color_step = [0.1, 0.2, 0.5]  # Example color step values
experiment = 'exp_zhang_example'  # Example experiment name

encoder_RAGFT_RLGR_large_blocks(dataset, sequence, b, color_step, experiment)
