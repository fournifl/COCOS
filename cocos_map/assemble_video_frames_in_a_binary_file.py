# -*- coding: utf-8 -*-
"""
Created on Tue Mar  15 17:32:08 2022

@author: fournifl
"""
import pdb
import json
import cv2
import pickle
from glob import glob
import numpy as np
import pickle as pk
from pathlib import Path


# sitename
sitename = 'palavas_cristal1'

# data_dir
data_dir = Path('/home/florent/dev/COCOS/data/raw/palavas/cristal_1/')

# input projected frames
# dir_frames = '/home/florent/dev/COCOS/data/raw/palavas/cristal_1/frames_projected/'
dir_frames = data_dir.joinpath('frames_projected/')

# ls = sorted(glob(dir_frames + 'P*.png'))
ls = sorted(dir_frames.rglob("P*.png"))

# number of used frames
n = 25
if n is not None:
    ls = ls[0:n]
else:
    n = len(ls)

# height, width, of frames
img_0 = cv2.imread(str(ls[0]))
height, width = img_0.shape[0:2]

# frame duration
dt = 0.5

# grid projected coordinates
f_coords = data_dir.joinpath('coords_projected_grid.pkl')
grid_coords = pickle.load(open(f_coords, 'rb'), encoding='latin-1')
# grid_coords = pickle.load(open('/home/florent/dev/COCOS/data/raw/palavas/cristal_1/coords_projected_grid.pkl', 'rb'), encoding='latin-1')
dx = round(grid_coords['utmx_projected_grid'][0, 1] - grid_coords['utmx_projected_grid'][0, 0], 2)

# georef
georef = json.load(open('/home/florent/ownCloud/Projets/SuiviVideo/Palavas/cameras/CAM16/info/georef/georef.json'))

# update grid projection coordinates with camera position
camera_position =  georef['Grid_Coordinate_System_Offset']
grid_coords['utmx_projected_grid'] += camera_position[0]
grid_coords['utmy_projected_grid'] += camera_position[1]

# creation of output dictionnary
output_dict = {}
output_dict['X'] = grid_coords['utmx_projected_grid']
output_dict['Y'] = grid_coords['utmy_projected_grid']
output_dict['dx'] = dx
output_dict['dt'] = dt

output_dict['RectMov_gray'] = np.zeros((height, width, n))

for i, f in enumerate(ls):
    # conversion of frame to gray
    img = cv2.imread(str(f))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    output_dict['RectMov_gray'][:, :, i] = img_gray

# save output dictionnary
pickle.dump(output_dict, open(data_dir.joinpath('Video_' + sitename + '.pkl'), 'wb'))
# np.savez(data_dir.joinpath('Video_' + sitename), output_dict)

    