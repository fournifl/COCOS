"""
@author: fournifl
"""
import pdb

import numpy as np
from PIL import Image
import cv2
import time

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
    # for i, frame_img in enumerate(ls):
    #     frame = cv2.imread(str(frame_img), cv2.IMREAD_GRAYSCALE)
    #     frames_array[:, :, i] = frame
    return frames_array, l



