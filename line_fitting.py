#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import display_images
from binary_map_pipeline import get_warp_matrix, binary_map_pipeline
from Line_class import Line
import pickle
import glob

# ============================== MAIN PROGRAM ====================================

if __name__ == '__main__':

    # Global variables:
    img_size = (1280, 720)
    source_vertices = np.float32([[203, 720],
                                [580, 460],
                                [700, 460],
                                [1077, 720]])
    thresh_x = (20, 100)
    thresh_h = (18, 35)
    thresh_l = (190, 255)
    thresh_s = (120, 255)

    # Load calibration parameters from pickle:
    with open('calibration_params.pkl', 'rb') as pkl:
        M_cam = pickle.load(pkl)
        dist_coef = pickle.load(pkl)

    M_warp, dest_vertices = get_warp_matrix(img_size, source_vertices)

    path_names = glob.glob('./test_images/*.jpg')

    for path in path_names:
        # Isolate file name without extension:
        file_name = path.split('/')[-1].split('.')[0]
        print("Processing ", file_name)
        img = mpimg.imread(path)

        combined_map, img_undist, hls, warped, warped_s, \
            x_binary, yellows, lightness, saturation = binary_map_pipeline(img, 
                                        M_cam, dist_coef, M_warp, dest_vertices)

    display_images([img, 
                        img_undist, 
                        warped_s,
                        x_binary, 
                        # y_binary,
                        # mag_binary,
                        # dir_binary,
                        yellows,
                        lightness,
                        saturation,
                        combined_map
                        ], 
                        titles = ["original", 
                        "undistorted", 
                        "warped Sat", 
                        "grad_x",
                        # "grad_y", 
                        # "grad_magnitude", 
                        # "grad_direction", 
                        "yellows",
                        "lightness",
                        "saturation",
                        "combined",
                        ],
            n_cols = 4 
            #, write_path = './test_images/' + file_name + '_results.png'
            )

