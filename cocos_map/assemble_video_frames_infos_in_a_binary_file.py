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
import matplotlib.pyplot as plt
from pathlib import Path

# sitename
site = 'palavas'
cam_name = 'cristal_2'
# site = 'chicama'
# cam_name = 'cam'
sitename = f'{site}_{cam_name}'

# date, hour of video
date = '20220314'
hour = '07h'

# data_dir
data_dir = Path(f'/home/florent/dev/COCOS/data/raw/{site}/{cam_name}/')

# input projected frames
dir_frames = data_dir.joinpath(f'{date}/{hour}/frames_projected_grey/')

ls = sorted(dir_frames.rglob("P*.png"))

# number of used frames
# n = 900
n = 200
# n = None
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

dx = round(grid_coords['utmx_projected_grid'][0, 1] - grid_coords['utmx_projected_grid'][0, 0], 2)

# georef
georef = json.load(open('/home/florent/ownCloud/Projets/SuiviVideo/Palavas/cameras/CAM17/info/georef/georef.json'))
# georef = json.load(open('/home/florent/ownCloud/Projets/SuiviVideo/Palavas/cameras/CAM21/info/georef/georef.json'))
# georef = None

# update grid projection coordinates with camera position
if georef is not None:
    camera_position = georef['Grid_Coordinate_System_Offset']
else:
    camera_position = [0, 0]
grid_coords['utmx_projected_grid'] = grid_coords['utmx_projected_grid'].astype(float)
grid_coords['utmy_projected_grid'] = grid_coords['utmy_projected_grid'].astype(float)
grid_coords['utmx_projected_grid'] += camera_position[0]
grid_coords['utmy_projected_grid'] += camera_position[1]

# creation of output dictionnary
output_dict = {}
output_dict['X'] = grid_coords['utmx_projected_grid']
output_dict['Y'] = grid_coords['utmy_projected_grid']
output_dict['dx'] = dx
output_dict['dt'] = dt
output_dict['m'] = height
output_dict['n'] = width
output_dict['n_frames'] = n

output_dict['ls_frames'] = ls

# save
pk.dump(output_dict, open(data_dir.joinpath(f'{date}/{hour}/', f'Video_infos_{sitename}_res_{dx:.1f}.pk'), 'wb'))


