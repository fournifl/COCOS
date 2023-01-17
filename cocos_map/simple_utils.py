"""
@author: fournifl
"""


import numpy as np
import cv2

def optional_print(to_print, end='\n'):
    bool_print = False
    if bool_print:
        print(to_print, end=end)


def assemble_frames_paths_into_3d_array(ls, start, stop, m, n, n_frames_in_video):
    if stop <= n_frames_in_video -1:
        l = stop
    else:
        l = n_frames_in_video - 1
    frames_array = np.zeros((m, n, l))
    for i in range(start, stop):
        frame = cv2.imread(str(ls[i]), cv2.IMREAD_GRAYSCALE)
        frames_array[:, :, i] = frame
    return frames_array, l



