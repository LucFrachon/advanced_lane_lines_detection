#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Line(object):
    def __init__(self, color, last_n = 10):
        '''
        color: color in which to display the line
        last_n: int, how many previous fits should be remembered?
        '''

        # Color to use when displaying the line:
        self.color = color
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.last_x_fitted = [] 
        #average x values of the fitted line over the last n iterations
        self.avg_x = None     
        #polynomial coefficients averaged over the last n iterations
        self.avg_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.all_x = None  
        #y values for detected line pixels
        self.all_y = None