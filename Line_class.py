#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class Queue():
    def __init__(self, width):
        '''
        Initiates an instance of Queue() with width = width (number of
        sub-elements in each of the queue element)
        '''
        self.items = []
        self.width = width

    def isEmpty(self):
        return self.items == []

    def enqueue(self, item):
        if (type(item) == np.ndarray) & (item.shape == (width,)):
            self.items.insert(0, item)
        else:
            print("Wrong dimensions, item NOT enqueued")

    def dequeue(self):
        self.items.pop()

    def size(self):
        return len(self.items)

    def mean(self):  
        np_queue = np.array(self.items)
        return np.mean(np_queue, 0)

    def median(self):
        np_queue = np.array(self.items)
        return np.median(np_queue, 0)


class Line():
    def __init__(self, color, last_n = 10, tolerance = 0.2):
        '''
        color:      color in which to display the line
        order:      polynomial order for fitted lines
        last_n:     int, how many previous fits should be remembered?
        tolerance:  tolerance on the fit coefficients for a fit to be considered
                    valid (as a proportion of the coefficients)
        '''
        # Number of sets of fit coefficients to remember:
        self.last_n = last_n

        # Color to use when displaying the line:
        self.color = color

        # was the line detected in the last iteration?
        self.detected = False  

        # x values of the latest fitted line
        self.last_x_fitted = None

        #average x values of the fitted line over the last n iterations
        #self.avg_x = None     

        #polynomial coefficients from last n detections, stored as a 3-wide 
        #Queue (3 coefficients per fit):
        self.last_n_fits = Queue(3)

        #polynomial coefficients for the most recent fit
        self.current_fit = None

        #radius of curvature of the line in some units
        self.radius_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels (current frame)
        self.all_x = None  

        #y values for detected line pixels (current frame)
        self.all_y = None

        # Number of frames since successful detection:
        self.frames_since_detection = 0


    def update_line(self, fit):
        '''
        Once a valid line has been found and fitted, this function updates the 
        line attibutes.
        '''

        self.detected = True
        self.frames_since_detection += 1
        self.last_n_fits.enqueue(fit)

        # Don't let the queue exceed the maximum length:
        if len(self.last_n_fits.items) > self.last_n:
            self.last_n_fits.dequeue()
        return


    def sanity_check(self):
        '''
        Compare coefficients from the latest polynomial fit and the x position of 
        the line to the average n latest good fits. Return True if they pass
        the validation condition.
        '''

        last_n_mean = self.last_n_fits.mean()

        # We want coefficients to be within tolerance of the average coeffs:
        order_2_ok = np.absolute(
            (self.current_fit[0] - last_n_mean[0]) / \
            last_n_mean[0]) <= self.tolerance
        order_1_ok = np.absolute((self.current_fit[1] - last_n_mean[1]) / \
            last_n_mean[1]) <= self.tolerance
        order_0_ok = np.absolute((self.current_fit[2] - last_n_mean[2]) / \
            last_n_mean[2]) <= self.tolerance

        return xy_ok & order_2_ok & order_1_ok & order_0_ok


    def fit_poly(self):
        '''
        Fit a 2nd-order polynomial to the found x and y pixel coordinates and 
        update the line's coefficient queue if the coefficients pass the sanity 
        check.
        Returns:
        - valid:    Boolean, indicates successful fit with valid coefficients
        - self.current_fit: The fitted coefficients
        '''
        
        # Initialize variables:
        valid = False
        self.current_fit = None

        # First make sure we actually have points to fit:
        xy_ok = (self.all_x.shape != (0,)) & (self.all_y.shape != (0,))

        if xy_ok: # If so, fit polynomial and get coefficients
            self.current_fit = np.polyfit(self.all_y, self.all_x, 2)
            if sanity_check:  # If coefficients seem sensible:
                self.update_line(self.current_fit)
                valid = True

        return valid, self.current_fit


    def predict_poly(self, y):
        '''
        Calculates the outcome values associated to each y for the latest valid
        polynomial coefficients. Warning: y refers to the image coordinates, 
        therefore it is the predictor here. x is the outcome.
        Returns self.last_x_fitted, an array containing the calculated x values.

        '''
        fit = self.last_n_fits[0]
        self.last_x_fitted = fit[0] * y**2 + fit[1] * y + fit[2]

        return self.last_x_fitted