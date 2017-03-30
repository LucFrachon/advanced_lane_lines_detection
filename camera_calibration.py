#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

def display_images(images, titles = None, n_cols = 2, fig_size = (20, 10),
    write_path = None):
    '''
    Helper function to display several images as sublots - very useful in testing.
    images:     A list of images (each an np.array). Each image can have 1 or 3 
                channels
    titles:     A list of image titles to use. Defaults to empty strings.
    n_cols:     Number of columns to display the images on.
    fig_size:   Size of the figure window.
    write_path: Where to write the image on disk. If None, no image is written.
    
    Returns:    Figure and axes, in case further work is required on them.
    '''

    if titles is None:
        titles = ["" for i in range(len(images))]
    
    if len(titles) != len(images):
        print(len(titles), len(images))
        print("titles and images should have the same number of elements")
        return
    
    # We will need the image size to fill the figure if case not all subplots
    # were used:
    img_size = images[0].shape  # Get shape of the first image
    if len(img_size) == 2:
        img_size = img_size + (3,)
        
    # Calculate the number of rows from the number of images and columns:
    n_rows = np.ceil(1. * len(images) / n_cols).astype('uint8')
    
    # Create the subplot:
    f, axes = plt.subplots(n_rows, n_cols, figsize = fig_size, squeeze = False)
    f.tight_layout()

    # Iterate through all images:
    for i, img in enumerate(images):
        id_row = i // n_cols
        id_col = i % n_cols
        # Check channel depth of the image to display it properly:
        if (len(img.shape) < 3) or (img.shape[2] == 1):
            axes[id_row, id_col].imshow(img.squeeze(), cmap = 'gray')
        else:
            axes[id_row, id_col].imshow(img)
        # Display title in a fontsize dependent on the figure size:
        axes[id_row, id_col].set_title(titles[i], fontsize = fig_size[0])
        
    # Fill figure with blank images if not all subplots were used:
    for i in range(len(images), n_rows * n_cols): 
        id_row = i // n_cols
        id_col = i % n_cols
        white = np.ones(img_size)
        axes[id_row, id_col].imshow(white)

    if write_path is not None:
        f.savefig(write_path)

    return f, axes


def get_distortion_params(cb_images, cb_size):
    '''
    Takes a chessboard image and the number of internal corners to detect then 
    computes the distortion coefficients

    cb_images:  List of images of the SAME chessboard
    cb_size:    Tuple, number of internal corners of the chessboard (rows, cols)

    Returns: 
    calibrated:   Whether calibration was successful or not.
    M_cam:        Camera matrix (3 x 3 array)
    dist_coef:    Distortion coefficients (k1, k2, p1, p2, k3)
    '''
    
    # More readable variable names ('cb_' <=> 'chessboard'):
    cb_rows = cb_size[1]
    cb_cols = cb_size[0]

    # Prep the object points array - a stack of (col, row) pairs
    obj_pts = np.zeros((cb_rows * cb_cols, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:cb_cols, 0:cb_rows].T.reshape(-1, 2)


    # Initialize the lists of object points and image points used in calibration:
    obj_points = []
    img_points = []

    for i, img in enumerate(images):
        # Convert image it to grayscale:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Get image size (we have to reverse the order of .shape())!
        img_size = (gray.shape[1], gray.shape[0])

        # Look for the cb's corners:
        found, corners = cv2.findChessboardCorners(gray, (cb_cols, cb_rows), None)

        if found:  # then we add this object's points and this image's corners
            obj_points.append(obj_pts)
            img_points.append(corners)

    calibrated, M_cam, dist_coeff, _ , _ = cv2.calibrateCamera(obj_points, 
        img_points, img_size, None, None)

    return calibrated, M_cam, dist_coeff


def undistort_image(image, M_cam, dist_coeff):
    '''
    Wrapper function for cv2.undistort()

    image:      Any image
    M_cam:      Camera matrix, output from get_distortion_params()
    dist_coeff: Distortion coefficients (k1, k2, p1, p2, k3), output from 
                 get_distortion_params()
    Returns a (hopefully) undistorted image
    '''
    return cv2.undistort(image, M_cam, dist_coeff, None, M_cam)


# ===== Main program =====

if __name__ == '__main__':
    
    images = []
    
    # Get all calibratin images:
    file_names = glob.glob("./camera_cal/calibration*.jpg")
    for f_name in file_names:
        images.append(mpimg.imread(f_name))

    # Calculate parameters:
    ret, M_cam, dist_coeff = get_distortion_params(images, (9, 6))

    # Save these parameters as a Pickle:
    with open('calibration_params.pkl', 'wb') as pkl:
        pickle.dump(M_cam, pkl)
        pickle.dump(dist_coeff, pkl)


    # test_images = [images[0], images[3], images[11], images[17]]
    # to_display = []
    # for img in test_images:
    #     to_display.append(img)
    #     to_display.append(undistort_image(img, M_cam, dist_coeff))

    # display_images(to_display, 
    #     titles = ["Original", "Undistorted"] * int(len(to_display) / 2), 
    #     fig_size = (20, 25), 
    #     write_path = "./camera_cal/result.png")