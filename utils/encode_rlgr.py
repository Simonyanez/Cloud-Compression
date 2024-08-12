import rlgr
import os
import numpy as np

def encode_rlgr(data,filename="test.bin",is_signed=1):
    if os.path.isfile(filename):
        os.remove(filename)
    data = data.astype(np.uint8).tolist()
    do_write = 1
    enc = rlgr.file(filename, do_write)

    # Write data
    enc.rlgrWrite(data, is_signed)
    enc.close()
    numbits = os.path.getsize(filename) * 8
    return numbits