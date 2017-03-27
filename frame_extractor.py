#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from moviepy.editor import VideoFileClip
import math
import os

video_file = './challenge_video.mp4'
images_folder = './challenge_test_images/'

def frange(start, stop, step = 1.):
    i = start
    while i < stop:
        yield i
        i += step

clip = VideoFileClip(video_file)

for ts in frange(16., 17., 0.3):
    clip.save_frame(images_folder + str(ts) + ".jpg", t = ts)

files = os.listdir(images_folder)