#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from moviepy.editor import VideoFileClip
import math
import os

video_file = './project_video.mp4'
images_folder = './short_test_images/'

def frange(start, stop, step = 1.):
    '''
    Generator that mimics range() but accepts floating point numbers.
    '''
    i = start
    while i < stop:
        yield i
        i += step


# Extract frames at regular intevals from a video clip and save them to disk:
clip = VideoFileClip(video_file)

for i, ts in enumerate(frange(0., 2., 1.)):
    clip.save_frame(images_folder + str(i) + ".jpg", t = ts)

files = os.listdir(images_folder)