"""
@author: fournifl
"""
import pdb

import numpy as np
import cv2

def optional_print(to_print, end='\n'):
    bool_print = False
    if bool_print:
        print(to_print, end=end)


def assemble_frames_paths_into_3d_array(ls, start, stop, m, n, n_frames_in_video):
    if stop <= n_frames_in_video -1:
        block = stop - start
    else:
        block = n_frames_in_video - start
    frames_array = np.zeros((m, n, block))

    ii = 0
    for i in range(start, stop):
        frame = cv2.imread(str(ls[i]), cv2.IMREAD_GRAYSCALE)
        frames_array[:, :, ii] = frame
        ii += 1
    return frames_array, block
