"""
@author: fournifl
"""
import pdb
from typing import Callable
import numpy as np
import cv2


class FrameParser:
    """parse frames and read them only once, even in overlappoing case"""

    def __init__(self, frame_bin_size):
        # Initialize variables
        self.frames = []
        self.frames_last_indice = 0
        self.frame_bin_size = frame_bin_size

    def assemble_frames_paths_into_3d_array(self, ls, start, stop, m, n, n_frames_in_video):
        """get frames data between start and stop, avoiding to read twice the same data"""

        # get current bin size
        if stop <= n_frames_in_video - 1:
            block = self.frame_bin_size
        else:
            block = (n_frames_in_video - 1) - start
            stop = n_frames_in_video - 1

        # store frames
        if start == 0:
            self.frames = [cv2.imread(str(ls[i]), cv2.IMREAD_GRAYSCALE) for i in range(start, stop)]
            self.frames_last_indice += block
        else:
            ii = self.frames_last_indice - start
            self.frames = self.frames[ii:]
            for i in range(self.frames_last_indice, stop):
                self.frames.append(cv2.imread(str(ls[i]), cv2.IMREAD_GRAYSCALE))
            self.frames_last_indice += int(block / 2)

        return np.dstack(self.frames), block

