#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from camera_calibration import *
from Line_class import Line
import pickle


def isolate_roi(image, vertices):
    '''
    Applies an image mask to isolate a region of interest, define as a polygon.
    Only keeps the region of the image defined by the polygon formed from 
    `vertices`. The rest of the image is set to black.
    '''
    # Define a blank mask to start with
    mask = np.zeros_like(image)   
    
    # Define a 3 channel or 1 channel color to fill the mask with depending on 
    #the input image
    if len(image.shape) > 2:
        channel_count = image.shape[2] 
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    # Fill pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    # Return the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def get_warp_matrix(img_size, source_coords):
    '''
    Computes the warp matrix obtained by mapping 'source_coords' to the vertices
    defining a rectangle whose width and horizontal position are the same as the
    lower base of the original trapeze, and whose height is equal to the height
    of the destination image.
    As a convention, vertices are ordered clockwise starting from the lower left,
    on both the original trapeze and the obtained rectangle.
    img_size is the original image size, therefore is passed as (n_rows, n_cols)

    Returns:    
    M_warp:     Warp matrix
    dest_coords: Coordinates of the warped polygon's vertices (np.array of float)
    '''
    source_coords = np.float32(source_coords)
    # Determine the destination coordinates (index 0 for the lower-left vertex)
    dest_coords = np.float32([[source_coords[0, 0], img_size[1]],
                              [source_coords[0, 0], 0],
                              [source_coords[3, 0], 0],
                              [source_coords[3, 0], img_size[1]]])
    # Compute transformation matrix:
    M_warp = cv2.getPerspectiveTransform(source_coords, dest_coords)

    return M_warp, dest_coords


def warp_image(image, warp_matrix, inverse = False, lines = False, 
        source_coords = None, dest_coords = None):
    '''
    Given an original image, applies a warp transformation using 'warp_matrix'
    and returns both the original and the warped images.
    If inverse == True, the inverse operation is performed (rectangle to trapeze).
    If 'lines' == True, the original trapeze and the warped rectangle are drawn on
    the respective image. In that case, 'source_coords' and 'dest_coords' must be
    provided as np.arrays.

    '''
    # Get image size:
    img_size = (image.shape[1], image.shape[0])
    # Apply inverse flag depending on direction of transformation:
    if inverse:  
        flags = (cv2.INTER_LINEAR, cv2.WARP_INVERSE_MAP)
    else:
        flags = cv2.INTER_LINEAR

    # Perform image warping:
    warped = cv2.warpPerspective(image, M_warp, img_size, 
        flags = flags)  

    if lines:  # Do we want to draw polygons on the images?
        if (source_coords is None) or (dest_coords is None):
            print("Warning: If you want lines to be drawn, you must provide  \
                'source_coords' and 'dest_coords'. No lines drawn this time.")
        col = (255, 0, 0)
        src = np.int32(source_coords)
        dst = np.int32(dest_coords)
        image = cv2.polylines(image, [src], isClosed = 1, color = col,
            thickness = 3)
        warped = cv2.polylines(warped, [dst], isClosed = 1, color = col,
            thickness = 3)

    return image, warped


def scale_map(map, max_value = 255, dtype = 'uint8'):
    '''
    Takes a map (outcome of cv2.Sobel() for instance) and scales it to 
    [0, max_value], then converts to dtype.
    '''
    return (float(max_value) * map / np.max(map)).astype(dtype)


def make_grad_map(image, orient = 'x', sobel_kernel = 3, thresh = (0, 255)):
    '''
    Takes a single-channel image and applies a Sobel filter on it in either the x
    or the y direction and applies a threshold to only retain values within the 
    threshold interval.

    img:        A single-channel image
    orient:     Direction of the Sobel filter (either 'x' or 'y')
    sobel_kernel: Size of the Sobel kernel
    thresh:     Tuple with two elements. The threshold interval to build the map

    Returns:
    - A binary map (black / white) of the pixels that passed the filter.
    - The absolute gradient values, scaled to 8-bit, for later use.
    '''
    
    # Compute the 1st order gradient in the direction 'orient':
    sobel = cv2.Sobel(image, cv2.CV_64F, orient == 'x', 
                      orient == 'y', ksize = sobel_kernel)
    # Take the gradient's absolute value:
    abs_sobel = np.absolute(sobel)

    # Scale to 8-bit (0 - 255) then convert to type = np.uint8
    sobel_scaled = scale_map(abs_sobel)

    # Create a mask of 1's where the scaled gradient magnitude is within the 
    #threshold interval:    
    grad_binary = np.zeros_like(sobel_scaled).astype('uint8')
    grad_binary[(sobel_scaled >= thresh[0]) & (sobel_scaled <= thresh[1])] = 1
    
    # Return this binary_output image and absolute values of gradient
    return grad_binary, abs_sobel


def make_grad_mag_map(sobel_x, sobel_y, thresh = (0, 255)):
    '''
    Takes the output of cv2.Sobel() in the x and y directions (or their absolute 
    values) and computes the gradient magnitude, then applies a threshold to 
    create a binary map.
    '''
    # Calculate the L2-norm and scale it to 8-bit
    sobel_mag = scale_map(np.sqrt(sobel_x ** 2 + sobel_y ** 2))
    # Make binary map containing only pixels where the magnitude is within thresh:
    binary_map = np.zeros_like(sobel_mag)
    binary_map[(sobel_mag >= thresh[0]) & (sobel_mag <= thresh[1])] = 1
    
    return binary_map


def make_grad_dir_map(sobel_x, sobel_y, thresh = (0, 90)):
    '''
    Takes the outputs of cv2.Sobel() in the x and y directions (in absolute value)
    and calculates the gradient direction, then filters out gradients whose 
    direction is outside thresh or -thresh. The pixels that pass are then retained
    to make a binary map. Thresh is expressed in degrees.
    '''
    # Compute gradient angle to the horizontal (in degrees):
    grad_dir = np.arctan2(sobel_y, sobel_x) * 180. / np.pi
    # Make binary map of only pixels where direction is within thresh or -thresh:
    binary_map = np.zeros_like(grad_dir)
    binary_map[((grad_dir >= thresh[0]) & (grad_dir <= thresh[1])) |
               ((grad_dir <= -thresh[0]) & (grad_dir >= -thresh[1]))] = 1

    return binary_map


def make_channel_map(image, thresh = (0, 255), scale_to = 255, channel = 1):
    ''' 
    Takes a 3-channel image and a channel and makes a binary map for that channel,
    only retaining pixels that are within thresh.
    '''
    # Isolate the specified channel and scale it:
    channel_img = scale_map(image[:,:, channel], max_value = scale_to)
    # Make binary map:
    binary = np.zeros_like(channel_img)
    binary[(channel_img >= thresh[0]) & (channel_img <= thresh[1])] = 1

    return binary



def combine_maps(maps):
    '''
    Takes a dict of binary maps in the form:
    {'grad_x': <map>, 'grad_y': <map>, 'grad_mag': <map>, 'grad_dir': <map>, 
    'hue': <map>, 'light': <map>, 'sat': <map>}
    and combines them into a single one. All maps should have the same size.
    Returns a binary map.
    '''
    # Prepare an array of zeros in the shape of whatever is the "first" map in 
    # maps (the fact that dicts are unordered doesn't matter here):
    img_size = maps[list(maps)[0]].shape
    # Make a combined binary map for gradient-related measures:
    gradient_map = np.zeros(img_size)
    # gradient_map[((maps['grad_x'] == 1) & (maps['grad_y'] == 1)) |
    #     (maps['grad_mag'] == 1) & (maps['grad_dir'] == 1)] = 1
    gradient_map[maps['grad_x'] == 1] = 1

    # Make a combined binary map for color-related measures:
    color_map = np.zeros(img_size)
    color_map[((maps['sat'] == 1) & (maps['hue'] == 1)) |
        ((maps['sat'] == 1) & (maps['light'] == 1))] = 1

    # Finally, combine these two maps (gradients and colors):
    combined_map = np.zeros(img_size)
    combined_map[(gradient_map == 1) | (color_map == 1)] = 1

    return combined_map



def binary_map_pipeline(image, M_cam, dist_coef, M_warp, dest_vertices):
    '''
    Takes an image and distortion parameter M_cam and dist_coeff. Applies 
    correction for lens distortion, warps it to bird's eye view using M_warp and 
    dest_vertices, then builds a binary map, combined from binary maps based on 
    gradient and on color.
    '''

    # Correct for camera distortion:
    img_undist = undistort_image(image, M_cam, dist_coef)

    # Convert warped image to grayscale before passing to filters:
    hls = cv2.cvtColor(img_undist, cv2.COLOR_RGB2HLS)

    # Warp image to bird's eye view:
    hls, warped = warp_image(hls, M_warp, 
        inverse = False, lines = False, source_coords = source_vertices, 
        dest_coords = dest_vertices)

    warped_s = warped[:,:,2]  # Isolate S channel for gradient calculations

    # Make binary maps from x gradient:
    x_binary, x_sobel  = make_grad_map(warped_s, orient = 'x', 
        sobel_kernel = 7, thresh = thresh_x)
    # y_binary, y_sobel = make_grad_map(warped, orient = 'y', 
    #     sobel_kernel = 7, thresh = (10, 100))
    # mag_binary = make_grad_mag_map(x_sobel, y_sobel, thresh = (10, 100))
    # dir_binary = make_grad_dir_map(x_sobel, y_sobel, thresh = (0, 15))

    # Make binary maps from hue, lightness and saturation:

    yellows = make_channel_map(warped, thresh = thresh_h, channel = 0, 
        scale_to = 180)  # H-channel scaled to [0, 180] (not [0, 255])
    lightness = make_channel_map(warped, thresh = thresh_l, channel = 1)
    saturation = make_channel_map(warped, thresh = thresh_s, channel = 2)

    # Combine into a single binary map:
    maps = {'grad_x': x_binary, 
            # 'grad_y': y_binary,
            # 'grad_mag': mag_binary,
            # 'grad_dir': dir_binary,
            'hue': yellows,
            'light': lightness,
            'sat': saturation}
    combined_map = combine_maps(maps)

    return combined_map, img_undist, hls, warped, warped_s, x_binary, yellows, \
        lightness, saturation



# ========= MAIN PROGRAM ==========

if __name__ == '__main__':
    
    # Get calibration parameters from pickle:
    with open('calibration_params.pkl', 'rb') as pkl:
        M_cam = pickle.load(pkl)
        dist_coef = pickle.load(pkl)

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
        # # Correct for camera distortion:
        # img_undist = undistort_image(img, M_cam, dist_coef)

       
        # # Convert warped image to grayscale before passing to filters:
        # hls = cv2.cvtColor(img_undist, cv2.COLOR_RGB2HLS)

        # # Warp image to bird's eye view:
        # hls, warped = warp_image(hls, M_warp, 
        #     inverse = False, lines = False, source_coords = source_vertices, 
        #     dest_coords = dest_vertices)

        # warped_s = warped[:,:,2]  # Isolate S channel for gradient calculations

        # # Make binary maps from x, y gradeints, grad magnitude and grad direction:
        # x_binary, x_sobel  = make_grad_map(warped_s, orient = 'x', 
        #     sobel_kernel = 7, thresh = thresh_x)
        # # y_binary, y_sobel = make_grad_map(warped, orient = 'y', 
        # #     sobel_kernel = 7, thresh = (10, 100))
        # # mag_binary = make_grad_mag_map(x_sobel, y_sobel, thresh = (10, 100))
        # # dir_binary = make_grad_dir_map(x_sobel, y_sobel, thresh = (0, 15))

        # # Make binary maps from hue, lightness and saturation:

        # yellows = make_channel_map(warped, thresh = thresh_h, channel = 0, 
        #     scale_to = 180)  # H-channel scaled to [0, 180] (not [0, 255])
        # lightness = make_channel_map(warped, thresh = thresh_l, channel = 1)
        # saturation = make_channel_map(warped, thresh = thresh_s, channel = 2)

        # # Combine into a single binary map:
        # maps = {'grad_x': x_binary, 
        #         # 'grad_y': y_binary,
        #         # 'grad_mag': mag_binary,
        #         # 'grad_dir': dir_binary,
        #         'hue': yellows,
        #         'light': lightness,
        #         'sat': saturation}
        # combined_map = combine_maps(maps)

        
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

