##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./camera_cal/result.png "Correction for camera distortion"
[image2]: ./output_images/straight_lines1_distortion_correction.jpg "Undistorted image of straight line"
[image3]: ./output_images/test6_binary_map_construction.jpg "Binary map construction process"
[image4]: ./output_images/straight_lines1_warping.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This step is performed by the code in the `camera_calibration.py` file. Note that in addition to the functions strictly associated to calibration, this file also contains a useful helper function to plot several images as subplots into a single figure save it to disk (`display_images()`). This is what I used for some of the images contained in this file.

To perform camera calibration, I first load chessboard images into a list and run `get_distortion_params()` on it, whose code was adapted from code provided in the project lectures.
`get_distortion_params()` takes a list of chessboard images and the number of inside corners (must be the same for all!).
I prepare stack of (col, row) pairs filled with numbers from (0, 0) to (`cb_col`, `cb_row`) which constitute the real-life chessboard's corner coordinates in some local unit.

The images are then converted to grayscale and, using `cv2.findChessboardCorners()`, I detect inside corners and store their positions. Each time an image's corners are found, I append its real-life and image coordinates to the lists `obj_points` and `img_points`.

Using these two lists, I then call `cv2.calibrateCamera()` to calculate the calibration matrix and coefficients.

These parameters are then pickled for later use.

There is also a functin called `undistort_image()`, which as its name suggests, uses the calculated distortion parameters to correct the lens distortion. This will be used later in the pipeline.

The result looks like this:

![alt text][image1]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image

Here is an example of distortion correction on a road image. I chose a straight line to make to correction easier to see but it is far from obvious -- the car's hood is maybe where the correction is the most visible:

![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I experimented a great deal on different combinations of gradient and color thresholding. However one of the first experiments I ran was to compare the binary maps obtained by warping the image after applying thresholding to those obtained by warping the image first, then applying.

Warping the image first gave me better results. Even though locations further down the road appear blurry after warping (which might make gradient thresholding less effective), thresholding the warped image had one considerable advantage: Because the lines are usually close to the vertical, the only really useful gradient is along the x direction. This means that I don't need y-gradients, nor total gradient magnitude (because I can just use the magnitude along x), nor gradient direction. I thus have 4 fewer parameter sets to tune and worry about, which makes the pipeline both more robust and easier to optimize.

Concretely, the final binary map is created by the function `global_binary_map()` (line 201) in `lane_detection_pipeline.py` (this is the only file we will be considering from now on).
In this function, the previously detailed `undistort_image()` function is applied to the original image. Then the image is warped to a bird's eye view using `warp_image()` (line 44).

I then convert the image to HLS and isolate the L channel to apply x-gradient thresholding. This gives me the first binary map.

Next, I apply thresholding on each of the H, L and S channels. The H channel in particular is useful for identifying yellow lines, which are not alway picked up by the L and S channels.

I then combine these 4 binary maps (gradient, hue, lightness and saturation) using the function `combine_maps()` (line 171) to return a single map that can be used for line detection.

The figure below shows how these different maps are combined.

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

As previously explained, I found it more effective to perform the perspective transformation _before_ thresholding the image to make the binary maps.

This operation is performed in two steps:

- First, `get_warp_matrix()` (line 18 of `lane_detection_pipeline.py`) is executed once to calculate the warp matrix and the destination coordinates. Source coordinates are provided as a global variable to make tuning easier. Destinations coordinates are assumed to be the corners of the destination image but since they could be redefined, I chose to make them an outcome variable of the function.
- Secondly, `warp_image()` runs on each image. This function takes any image (any number of channels) and uses outcomes of `get_warp_matrix()` to transform the image to bird's-eye view. However, the function also has an `inverse` parameter that can be used to perform the inverse transformation (bird's-eye to in-car view). The function also offers the possibility to trace lines showing the transformation on both the original and warped images.

Here are the values I used for the project video (I noticed that the challenge video requires slightly different values):

```
source_coords = np.float32([[0, 670],
                            [538, 460],
                            [752, 460],
                            [1279, 670]])
dest_coords = np.float32([[source_coords[0, 0], img_size[1]],
                          [source_coords[0, 0], 0],
                          [source_coords[3, 0], 0],
                          [source_coords[3, 0], img_size[1]]])
```

This resulted in the following source and destination points:

| Source              | Destination   | 
|:-------------------:|:-------------:| 
| 0, 670 (bottom left)| 0, 719        | 
| 538, 460            | 0, 0          |
| 752, 460            | 1279, 0       |
| 1279, 670           | 1279, 719     |

And we get the following result (on the right-hand side, the blue line follows the image borders): 

![alt text][image4]

###################

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

