#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_agg import FigureCanvasAgg
import pickle
from camera_calibration import *
from moviepy.editor import VideoFileClip
from Line_class import Line

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


def scale_map(binary_map, max_value = 255, dtype = 'uint8'):
    '''
    Takes a binary_map (outcome of cv2.Sobel() for instance) and scales it to 
    [0, max_value], then converts to dtype.
    '''
    return (float(max_value) * binary_map / np.max(binary_map)).astype(dtype)


def make_grad_map(image, orient = 'x', sobel_kernel = 3, thresh = (0, 255)):
    '''
    Takes a single-channel image and applies a Sobel filter on it in either the x
    or the y direction and applies a threshold to only retain values within the 
    threshold interval.

    img:        A single-channel image
    orient:     Direction of the Sobel filter (either 'x' or 'y')
    sobel_kernel: Size of the Sobel kernel
    thresh:     Tuple with two elements. The threshold interval to build the binary_map

    Returns:
    - A binary binary_map (black / white) of the pixels that passed the filter.
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
    create a binary binary_map.
    '''
    # Calculate the L2-norm and scale it to 8-bit
    sobel_mag = scale_map(np.sqrt(sobel_x ** 2 + sobel_y ** 2))
    # Make binary binary_map containing only pixels where the magnitude is within thresh:
    binary_map = np.zeros_like(sobel_mag)
    binary_map[(sobel_mag >= thresh[0]) & (sobel_mag <= thresh[1])] = 1
    
    return binary_map


def make_grad_dir_map(sobel_x, sobel_y, thresh = (0, 90)):
    '''
    Takes the outputs of cv2.Sobel() in the x and y directions (in absolute value)
    and calculates the gradient direction, then filters out gradients whose 
    direction is outside thresh or -thresh. The pixels that pass are then retained
    to make a binary binary_map. Thresh is expressed in degrees.
    '''
    # Compute gradient angle to the horizontal (in degrees):
    grad_dir = np.arctan2(sobel_y, sobel_x) * 180. / np.pi
    # Make binary binary_map of only pixels where direction is within thresh or -thresh:
    binary_map = np.zeros_like(grad_dir)
    binary_map[((grad_dir >= thresh[0]) & (grad_dir <= thresh[1])) |
               ((grad_dir <= -thresh[0]) & (grad_dir >= -thresh[1]))] = 1

    return binary_map


def make_channel_map(image, thresh = (0, 255), scale_to = 255, channel = 1):
    ''' 
    Takes a 3-channel image and a channel and makes a binary binary_map for that channel,
    only retaining pixels that are within thresh.
    '''
    # Isolate the specified channel and scale it:
    channel_img = scale_map(image[:,:, channel], max_value = scale_to)
    # Make binary binary_map:
    binary = np.zeros_like(channel_img)
    binary[(channel_img >= thresh[0]) & (channel_img <= thresh[1])] = 1

    return binary



def combine_maps(maps):
    '''
    Takes a dict of binary maps in the form:
    {'grad_x': <binary_map>, 'grad_y': <binary_map>, 'grad_mag': <binary_map>, 'grad_dir': <binary_map>, 
    'hue': <binary_map>, 'light': <binary_map>, 'sat': <binary_map>}
    and combines them into a single one. All maps should have the same size.
    Returns a binary binary_map.
    '''
    # Prepare an array of zeros in the shape of whatever is the "first" binary_map in 
    # maps (the fact that dicts are unordered doesn't matter here):
    img_size = maps[list(maps)[0]].shape
    # Make a combined binary binary_map for gradient-related measures:
    gradient_map = np.zeros(img_size)
    # gradient_map[((maps['grad_x'] == 1) & (maps['grad_y'] == 1)) |
    #     (maps['grad_mag'] == 1) & (maps['grad_dir'] == 1)] = 1
    gradient_map[maps['grad_x'] == 1] = 1

    # Make a combined binary binary_map for color-related measures:
    color_map = np.zeros(img_size)
    color_map[((maps['sat'] == 1) & (maps['hue'] == 1)) |
        ((maps['sat'] == 1) & (maps['light'] == 1))] = 1

    # Finally, combine these two maps (gradients and colors):
    combined_map = np.zeros(img_size, dtype = 'uint8')
    combined_map[(gradient_map == 1) | (color_map == 1)] = 1

    return combined_map



def global_binary_map(image, M_cam, dist_coef, M_warp, dest_vertices):
    '''
    Takes an image and distortion parameter M_cam and dist_coeff. Applies 
    correction for lens distortion, warps it to bird's eye view using M_warp and 
    dest_vertices, then builds a binary binary_map, combined from binary maps 
    based on gradient and on color.
    '''

    # Correct for camera distortion:
    img_undist = undistort_image(image, M_cam, dist_coef)


    # # Convert warped image to HLS before passing to filters:
    # hls = cv2.cvtColor(img_undist, cv2.COLOR_RGB2HLS)

    # Warp image to bird's eye view:
    img_undist, warped = warp_image(img_undist, M_warp, 
        inverse = False, lines = False, source_coords = source_vertices, 
        dest_coords = dest_vertices)

    # Equalize histogram: convert to HUV then to back RGB
    warped_yuv = cv2.cvtColor(warped, cv2.COLOR_RGB2YCrCb)
    warped_yuv[:, :, 0] = cv2.equalizeHist(warped_yuv[:, :, 0])
    warped_hls = cv2.cvtColor(warped_yuv, cv2.COLOR_RGB2HLS)

    warped_s = warped_hls[:,:,2]  # Isolate S channel for gradient calculations

    # Make binary maps from x gradient:
    x_binary, x_sobel  = make_grad_map(warped_s, orient = 'x', 
        sobel_kernel = 7, thresh = thresh_x)
    # y_binary, y_sobel = make_grad_map(warped, orient = 'y', 
    #     sobel_kernel = 7, thresh = (10, 100))
    # mag_binary = make_grad_mag_map(x_sobel, y_sobel, thresh = (10, 100))
    # dir_binary = make_grad_dir_map(x_sobel, y_sobel, thresh = (0, 15))

    # Make binary maps from hue, lightness and saturation:

    yellows = make_channel_map(warped_hls, thresh = thresh_h, channel = 0, 
        scale_to = 180)  # H-channel scaled to [0, 180] (not [0, 255])
    lightness = make_channel_map(warped_hls, thresh = thresh_l, channel = 1)
    saturation = make_channel_map(warped_hls, thresh = thresh_s, channel = 2)

    # Combine into a single binary binary_map:
    maps = {'grad_x': x_binary, 
            # 'grad_y': y_binary,
            # 'grad_mag': mag_binary,
            # 'grad_dir': dir_binary,
            'hue': yellows,
            'light': lightness,
            'sat': saturation}
    combined_map = combine_maps(maps)

    return combined_map, img_undist, warped_hls, warped_s, x_binary, yellows, \
        lightness, saturation



#*******************************************************************************



def window_search(binary_map, file_index = None):
    '''
    binary_map is a warped binary map, one channel only.
    '''
    # Make histogram
    histogram = np.sum(binary_map[int(binary_map.shape[0] / 2):, :], axis = 0)
    # Create an output image to draw on and visualize the result:
    out_img = np.dstack((binary_map, binary_map, binary_map)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_map.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_map.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 20
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_map.shape[0] - (window + 1) * window_height
        win_y_high = binary_map.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,
            (win_xleft_low, win_y_low),
            (win_xleft_high, win_y_high),
            (0, 0, 255), 2) 
        cv2.rectangle(out_img,
            (win_xright_low, win_y_low),
            (win_xright_high, win_y_high),
            (0, 0, 255), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
            (nonzerox >= win_xleft_low) & \
            (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
            (nonzerox >= win_xright_low) & \
            (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    # leftx = nonzerox[left_lane_inds]   ## Modify with call to Line()
    # lefty = nonzeroy[left_lane_inds] 
    # rightx = nonzerox[right_lane_inds]
    # righty = nonzeroy[right_lane_inds] 
    line_left.all_x = nonzerox[left_lane_inds]
    line_left.all_y = nonzeroy[left_lane_inds]
    line_right.all_x = nonzerox[right_lane_inds]
    line_right.all_y = nonzeroy[right_lane_inds]
    

    # Fit a second order polynomial to each
     # Get all y values (essentially, each pixel along the y axis):
    ploty = np.linspace(0, img_size[1] - 1, img_size[1])

    _, _ = line_left.fit_poly(ploty)
    _, _ = line_right.fit_poly(ploty)

    left_fitx = line_left.predict_avg_poly(ploty)
    right_fitx = line_right.predict_avg_poly(ploty)

    # left_fit = np.polyfit(lefty, leftx, 2)  # Modify with call to Line()
    # right_fit = np.polyfit(righty, rightx, 2)

    # left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    # right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


    out_img[line_left.all_y, line_left.all_x] = [255, 0, 0]
    out_img[line_right.all_y, line_right.all_x] = [0, 255, 0]

    # Display the lines on the image:
    lanes_img = np.zeros_like(out_img)
    lanes_img[np.int32(ploty), np.int32(left_fitx)] = [255, 255, 0]
    lanes_img[np.int32(ploty), np.int32(right_fitx)] = [255, 255, 0]
    out_img = cv2.addWeighted(out_img, 1, lanes_img, 1, 0.0)

    return out_img


def search_around_line(binary_map, left_fit, right_fit, display = True,
    file_index = None):

    # It's now much easier to find line pixels!
    nonzero = binary_map.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + \
        left_fit[1] * nonzeroy + left_fit[2] - margin)) & \
        (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + \
        left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + \
        right_fit[1] * nonzeroy + right_fit[2] - margin)) & \
        (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + \
        right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_map.shape[0] - 1, binary_map.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    # Visualize:

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_map, binary_map, binary_map)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, 
        ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + \
        margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, 
        ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + \
        margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.imshow(result)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)    

    # Save the resulting plot as a Numpy array for further use:
    fig = plt.figure(figsize = (1280 / 96, 720 / 96), dpi = 96)
    canvas = FigureCanvasAgg(fig)
    ax = fig.gca()
    w, h = np.int32(fig.get_size_inches() * fig.get_dpi())
    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow', linewidth = 5)
    plt.plot(right_fitx, ploty, color='yellow', linewidth = 5)
    fig.tight_layout()
    plt.axis('off')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    canvas.draw()

    return left_fit, right_fit, image_out



# def window_mask(width, height, center, level):
#     output = np.zeros((img_size[1], img_size[0]), dtype = 'uint8')
#     output[int(img_size[1] - (level + 1) * height) : \
#         int(img_size[1] - level * height),
#         max(0, int(center - width / 2)) : min(int(center + width / 2),
#             img_size[0])] = 1
#     return output


# def find_window_centroids(bin_map, window_width, window_height, margin):
    
#     window_centroids = [] # Store the (left,right) window centroid positions per level
#     window = np.ones(window_width) # Create our window template that we will use for convolutions
    
#     # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
#     # and then np.convolve the vertical image slice with the window template 
    
#     # Sum quarter bottom of image to get slice, could use a different ratio
#     l_sum = np.sum(bin_map[int(3 * img_size[1] / 4):,
#         :int(img_size[0] / 2)], axis = 0)
#     l_center = np.argmax(np.convolve(window,l_sum)) - window_width / 2
#     r_sum = np.sum(bin_map[int(3 * img_size[1] / 4):,
#         int(img_size[0] / 2):], axis = 0)
#     r_center = np.argmax(np.convolve(window, r_sum)) - \
#         window_width / 2 + int(img_size[0] / 2)
    
#     # Add what we found for the first layer
#     window_centroids.append((l_center, r_center))
    
#     # Go through each layer looking for max pixel locations
#     for level in range(1, int(img_size[1] / window_height)):
#         # convolve the window into the vertical slice of the image
#         image_layer = np.sum(bin_map[int(img_size[1] - (level + 1) * \
#             window_height) : \
#             int(img_size[1] - level * window_height), :], axis = 0)
#         conv_signal = np.convolve(window, image_layer)
#         # Find the best left centroid by using past left center as a reference
#         # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
#         offset = window_width / 2
#         l_min_index = int(max(l_center + offset - margin, 0))
#         l_max_index = int(min(l_center + offset + margin, img_size[0]))
#         l_center = np.argmax(conv_signal[l_min_index : l_max_index]) + \
#             l_min_index - offset
#         # Find the best right centroid by using past right center as a reference
#         r_min_index = int(max(r_center + offset - margin, 0))
#         r_max_index = int(min(r_center + offset + margin, img_size[0]))
#         r_center = np.argmax(conv_signal[r_min_index : r_max_index]) + \
#             r_min_index - offset
#         # Add what we found for that layer
#         window_centroids.append((l_center, r_center))

#     return window_centroids


# def convolutional_search(binary_map, display = True, file_index = 0):

#     # window settings
#     window_width = 70 
#     window_height = 90 # Break image into 9 vertical layers since image height is 720
#     margin = 100 # How much to slide left and right for searching

#     window_centroids = find_window_centroids(binary_map, window_width, window_height, margin)

#     # If we found any window centers
#     if len(window_centroids) > 0:

#         # Points used to draw all the left and right windows
#         l_points = np.zeros_like(binary_map)
#         r_points = np.zeros_like(binary_map)

#         # Go through each level and draw the windows    
#         for level in range(0,len(window_centroids)):
#             # Window_mask is a function to draw window areas
#             l_mask = window_mask(window_width, window_height, 
#                 window_centroids[level][0], level)
#             r_mask = window_mask(window_width, window_height, 
#                 window_centroids[level][1], level)
#             # Add graphic points from window mask here to total pixels found 
#             l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
#             r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

#         if display:
#             # Draw the results
#             template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
#             zero_channel = np.zeros_like(template) # create a zero color channel 
#             template = np.array(cv2.merge((zero_channel, template, zero_channel)),
#                 np.uint8) # make window pixels green
#             warpage = np.array(cv2.merge((binary_map, binary_map, binary_map)),
#             np.uint8) # making the original road pixels 3 color channels
#             print(np.min(warpage), np.max(warpage), np.mean(warpage))
#             output = cv2.addWeighted(warpage, 255, template, 0.5, 0.0) # overlay the orignal road image with window results
         
#     # If no window centers found, just display orginal road image
#     else:
#         output = np.array(cv2.merge((binary_map, binary_map, binary_map)), 
#             np.uint8)

#     # Display the final results
#     plt.imshow(output)
#     plt.title('window fitting results')
#     plt.savefig('./test_images/convo_test_' + str(file_index) + '.png')


def compose_image(images, sub_size_ratio = 0.3):
    '''
    Takes a list  of images whose first element is considered as the main image.
    Sub-images are then overlaid on top of the main image with width and height 
    equal to 'sub_image_size' * main image size. The main image must be 3 deep.

    Returns the composite image.
    '''
    sub_size = np.int32(sub_size_ratio * np.array(img_size))  # ratio * (1280,720)
    n_cols = int(np.floor(1 / sub_size_ratio))
    composed = np.copy(images[0])

    if (1.*sub_size[1] / sub_size[0] <= img_size[1] / img_size[0] - 0.01) \
            | (1.*sub_size[1] / sub_size[0] >= img_size[1] / img_size[0] + 0.01):
            print("Error: Image ratio mush be equal to the original image's.")
            return

    for i, img in enumerate(images[1:]):       
        id_row = i // n_cols
        id_col = i % n_cols

        top_left_x = id_col * sub_size[0]  # id_col * (512)
        top_left_y = id_row * sub_size[1]  # id_col * (288)
                
        img_mini = scale_map(cv2.resize(img, (0, 0), fx = sub_size_ratio, 
            fy = sub_size_ratio), max_value = 255, dtype = 'uint8')

        if (len(img_mini.shape) == 2):
            composed[top_left_y : top_left_y + sub_size[1], 
                     top_left_x : top_left_x + sub_size[0], 
                     :] = np.dstack((img_mini, img_mini, img_mini))
        else:
            composed[top_left_y : top_left_y + sub_size[1], 
                     top_left_x : top_left_x + sub_size[0], 
                     :] = img_mini

    return composed

def lambda_wrapper(img, M_cam, dist_coef, M_warp, dest_vertices, 
    file_index = None, sub_size_ratio = .4):
    bin_map = global_binary_map(img, M_cam, dist_coef, M_warp, dest_vertices)[0]
    img_lines = window_search(bin_map, file_index = None)
    composed = compose_image([img, img_lines], sub_size_ratio = .4)
    return composed

# ============================== MAIN PROGRAM ====================================

if __name__ == '__main__':
    
    # Load calibration parameters from pickle:
    with open('calibration_params.pkl', 'rb') as pkl:
        M_cam = pickle.load(pkl)
        dist_coef = pickle.load(pkl)

    # Define global variables:
    img_size = (1280, 720)

    # For video 1:
    source_vertices = np.float32([[0, 670],
                                  [538, 460],
                                  [752, 460],
                                  [1280, 670]])

    #For video 2:
    # source_vertices = np.float32([[0, 720],
    #                               [559, 482],
    #                               [721, 482], 
    #                               [1280, 720]])

    # # Without histogram equalization:
    # thresh_x = (30, 100)
    # thresh_h = (18, 35)
    # thresh_l = (100, 255)
    # thresh_s = (90, 255)

    # With histogram equalization:
    thresh_x = (50, 100)
    thresh_h = (25, 35)
    thresh_l = (240, 255)
    thresh_s = (120, 255)  

    # Global Line class variables:
    line_left = Line([0, 255, 0], 5, pos_tol = .8, rad_tol = 0.5)
    line_right = Line([255, 0, 0], 5, pos_tol = .8, rad_tol = 0.5)




    # Calculate warping parameters:
    M_warp, dest_vertices = get_warp_matrix(img_size, source_vertices)

    # Load movie file to work on:

    # clip_in = VideoFileClip('short_video.mp4')
    # clip_out = clip_in.fl_image(lambda x: lambda_wrapper(x, M_cam, dist_coef,
    #     M_warp, dest_vertices, file_index = None, sub_size_ratio = .4))
    # clip_out.write_videofile('short_video_test.mp4', audio = False)

    file_index = 0
    
    path_names = glob.glob('./short_test_images/*.jpg')
    for path in path_names:
        # Isolate file name without extension:
        file_name = path.split('/')[-1].split('.')[0]
        print("Processing ", file_name)
        img = mpimg.imread(path)

        combined_map, img_undist, warped_hls, warped_s, \
            x_binary, yellows, lightness, saturation = global_binary_map(img, 
                                        M_cam, dist_coef, M_warp, dest_vertices)

        # binary_map, _, _, _, _, _, _, _, _ = global_binary_map(img, M_cam, 
        # dist_coef, M_warp, dest_vertices)
        # binary_map_mini = scale_map(cv2.resize(binary_map, (384, 216)), max_value = 255, 
        #     dtype = 'uint8')

        # composed = np.copy(img)
        # composed[:216, :384, :] = np.dstack((binary_map_mini, binary_map_mini, binary_map_mini))

        img_lines = window_search(combined_map, 
            file_index = None)
        # convolutional_search(binary_map, file_index = file_index)
        # composed = compose_image([img, img_lines], sub_size_ratio = .4)
        # f = plt.figure(figsize = (12.80,7.20), dpi = 100)
        # plt.imshow(composed)
        # f.savefig('./test_images/' + file_name + '_composed.png')
        # file_index += 1

        # display_images([composed], n_cols = 1, 
        #     write_path = './test_images/' + file_name + '_composed.png')
        
        print(combined_map.shape)

        display_images([img, 
                        warped_hls,
                        warped_s,
                        x_binary, 
                        yellows,
                        lightness,
                        saturation,
                        combined_map,
                        img_lines
                        ], 
                        
            n_cols = 4 
            , write_path = './short_test_images/' + file_name + '_results.png'
            )

