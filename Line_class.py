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

    def is_empty(self):
        return self.items == []

    def enqueue(self, item):
        if (item.shape == (self.width,)):
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
    def __init__(self, color, img_size, last_n = 5, search_n = 10, 
        pos_tol = 0.2, rad_tol = 0.2):
        '''
        color:      color in which to display the line
        order:      polynomial order for fitted lines
        last_n:     int, how many previous fits should be remembered?
        pos_tol:    tolerance on the order 0 coefficient for a fit to be 
                    considered valid (as a proportion of the mean of the n last
                    good fits)
        rad_tol:    tolerance on the curvature radius of a fit to be considered
                    valid (as a proportions of the mean of the last n good fits)
        '''
        # Image size:
        self.img_size = img_size

        # Number of sets of fit coefficients to remember:
        self.last_n = last_n

        # Tolerance on poly coefficients:
        self.pos_tol = pos_tol

        # Tolerance on curvature radius:
        self.rad_tol = rad_tol

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

        #radius of curvature of the last n iterations
        self.last_n_radius = Queue(1)

        #radius of curvature of the line in some units
        self.current_curvature = None

        #distance in meters of vehicle center from the line
        self.line_base_pos = self.img_size[0] / 2.

        #difference in fit coefficients between last and new fits
        #self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels (current frame)
        self.all_x = None  

        #y values for detected line pixels (current frame)
        self.all_y = None

        # Number of frames since successful detection:
        self.frames_since_detection = 0


    def compute_curvature(self, fit, y):

        y_eval = np.max(y)
        radius = np.asscalar(((1 + (2 * fit[0] * y_eval + \
            fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0] + 0.0001))
        return radius


    def update_line(self, fit, curve_rad):
        '''
        Once a valid line has been found and fitted, this function updates the 
        line attibutes.
        '''

        self.detected = True
        self.last_n_fits.enqueue(fit)
        self.last_n_radius.enqueue(np.array(curve_rad)[np.newaxis])
        self.line_base_pos = self.base_line_position()
        # Don't let the queue exceed the maximum length:
        if len(self.last_n_fits.items) > self.last_n:
            self.last_n_fits.dequeue()
            self.last_n_radius.dequeue()
        return


    def sanity_check(self):
        '''
        Compare coefficients from the latest polynomial fit and the x position of 
        the line to the average n latest good fits. Return True if they pass
        the validation condition.
        '''
        
        fits_empty = self.last_n_fits.is_empty()  # Is the queue of coefs empty?
        radius_empty = self.last_n_radius.is_empty()  # Is the queue of radiuses empty?

        if (fits_empty | radius_empty) | \
        (self.frames_since_detection >= self.last_n):
            return True  # If either queue is empty or we haven't been able to 
            # validate a fit since more than last_n frames, we accept these 
            # coefficients even thought we weren't able to make sure they made
            # sense
        
        else:  # If there are elements in the queue, sanity-check them:
            last_n_fits_mean = self.last_n_fits.mean()
            last_n_rad_mean = self.last_n_radius.mean()
            # We want the base line position to be within tolerance of the latest
            # n average:
            max_y = self.img_size[1] - 1
            current_pos = self.current_fit[0] * max_y ** 2 + \
                self.current_fit[1] * max_y + self.current_fit[2]
            avg_pos = last_n_fits_mean[0] * max_y ** 2 + \
                last_n_fits_mean[1] * max_y + last_n_fits_mean[2]

            position_ok = np.absolute((current_pos - avg_pos) / avg_pos) \
                <= self.pos_tol
            radius_ok = np.absolute((self.current_curvature - \
                last_n_rad_mean) / last_n_rad_mean) <= self.rad_tol
        
        return (position_ok & radius_ok)


    def fit_poly(self, y):
        '''
        Fit a 2nd-order polynomial to the found x and y pixel coordinates and 
        update the line's coefficient queue if the coefficients pass the sanity 
        check.
        Returns:
        - valid:    Boolean, indicates successful fit with valid coefficients
        - self.current_fit: The fitted coefficients
        '''
        
        # First make sure we actually have points to fit:
        xy_ok = (self.all_x.shape != (0,)) & (self.all_y.shape != (0,))

        if xy_ok: # If so, fit polynomial and get coefficients
            self.current_fit = np.polyfit(self.all_y, self.all_x, 2)
            self.current_curvature = self.compute_curvature(self.current_fit, y)

            if self.sanity_check():  # If coefficients seem sensible:
                self.update_line(self.current_fit, self.current_curvature)
                self.frames_since_detection = 0
            else:
                # self.update_line(self.last_n_fits.items[0], 
                #     self.last_n_radius.items[0])
                self.frames_since_detection = np.max((1, 
                    self.frames_since_detection + 1))
                self.detected = False
                print("Frames since last good detection:", 
                    self.frames_since_detection)

        return self.detected, self.current_fit


    def predict_poly(self, y):
        '''
        Calculates the outcome values associated to each y for the latest valid
        polynomial coefficients. Warning: y refers to the image coordinates, 
        therefore it is the predictor here. x is the outcome.
        Returns self.last_x_fitted, an array containing the calculated x values.

        '''
        fit = self.last_n_fits.items[0]
        self.last_x_fitted = fit[0] * y**2 + fit[1] * y + fit[2]

        return self.last_x_fitted


    def predict_avg_poly(self, y):
        '''
        Calculates the outcome values associated to each y for the mean of the 
        n latest polynomial coefficients. 
        Warning: y refers to the image coordinates, therefore it is the predictor 
        here. x is the outcome.
        Returns self.last_x_fitted, an array containing the calculated x values.

        '''

        fit = [self.last_n_fits.mean()[0], self.last_n_fits.mean()[1], 
            self.last_n_fits.mean()[2]]
        self.last_x_fitted = np.maximum(0, 
            np.minimum(fit[0] * y**2 + fit[1] * y + fit[2], 
            self.img_size[0] - 1))

        return self.last_x_fitted

    def base_line_position(self):
        '''
        Computes the position of the line at the bottom of the image, in relation 
        to the left border, based on the latest valid line fit.
        '''

        fit = self.last_n_fits.items[0]

        position = fit[0] * (self.img_size[1] - 1)**2 + \
            fit[1] * (self.img_size[1] - 1) + fit[2]

        return position