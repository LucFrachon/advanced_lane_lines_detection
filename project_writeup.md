## Advanced Line Detection
### Self-Driving Car Nanodegree Project4 - Computer Vision

---

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
[image5]: ./output_images/test6_sliding_window.jpg  "Sliding window search"
[image6]: ./output_images/test6_search_around_line.jpg "Search around line"
[image7]: ./output_images/test6_lane_projection.jpg "Lane projection"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading the project writeup. I also wrote a quick README for my Github repo.

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

The result looks like this on a few of the chessboard images:

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

The figure below shows how these different maps are combined to generate a more robust binary map.

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

As suggested in the lectures, I used two techniques for line identification: A sliding-window search, and search around previously fitted lines. I did implement a convolutional search but I found it to perform worse than these methods so I commented the code out.

To improve performance and efficiency, I first created a Line() class, whose code can be found in `Line_class.py` and is adapted from code provided in the lectures.

** The Line() class:**

Each instance of this class contains the following attributes:

- `img_size`: Image size
- `last_n`: Number of sets of polynomial coefficients to remember
- `pos_tol`: Tolerance on the base line position of the fitted lines, to help discriminate between valid and invalid lines
- `rad_tol`: Tolerance on the radius of curvature of the fitted lines, to help discriminate between valid and invalid lines
- `color`: Color in which to color the candidate pixels for this line
- `detected`: Whether or not a valid polynomial fit was found on this image
- `last_x_fitted`: x values of the latest (valid) fitted line (note: x refers to the image's axis and is the outcome variable for the line fit)
- `last_n_fits`: A Queue object of width 3 containing sets of triplets for polynomial coefficients of orders 2, 1, and 0 respectively, for the last `last_n` successfully fitted lines. See below for details of the Queue class.
- `current_fit`: The current polynomial fit candidate, to be tested for validation.
- `last_n_radius`: A Queue object of width 1 containing radii of curvature, for the last `last_n` successfully fitted lines. See below for details of the Queue class.
- `current_curvature`: The curvature radius of the current fit candidate to be tested for validation.
- `line_base_pos`: The estimated horizontal position of the line at the bottom of the image, in pixels. At initialization, set to the horizontal middle of the image because the Line() class is used for both left and right lines.
- `all_x`: x-coordinates for all detected pixels within the search area, for the current image or frame.
- `all_y`: y-coordinates for all detected pixels within the search area, for the current image or frame.
- `frames_since_detection`: A counter, counts the number of frames elapsed since we last found a valid polynomial fit. This is used to switch from search around last fitted line to full sliding window search.


The Line() class also contains the following methods:

- `compute_curvature()`: From a set of fit coefficients, returns the estimated radius of curvature.
- `update_line()`: Once a valid line has been found and fitted, updates the line attributes.
- `sanity_check()`: Compares the latest fit candidate with the average of the last `last_n` valid fits on two criteria (base line position and radius of curvature) to decide whether they are within tolerance, in which case the fit is considered as valid.
- `fit_poly()`: Fit a 2nd-order polynomial to the x and y pixel coordinates found within the search area, using `np.polyfit()`. If the fit passes the `sanity_check()`, return it with a `detected` flag and update the Line attributes.
- `predict_poly()`: From the last valid set of polynomial coefficients and an array of y values (here, y is the predictor variable, not the outcome), calculate the resulting x values (`last_x_fitted`).
- `predict_avg_poly()`: From the mean of the last `last_n` valid coefficients, calculate the resulting x values. Gives a smoother line accross frames than `predict_poly()` so this is what I used in the submitted project.
- `base_line_position()`: Calculate the base line position using the last valid set of polynomial coefficients. Updates `line_base_pos`.


** The Queue() class:**

The Queue() class is custom-defined at the begining of the same file. It contains the following attributes:

- `items`: A list of the items contained in the Queue
- `width`: The number of sub-elements contained in each element of `Queue.items`. For instance, a Queue of quadratic coefficients would have a width of 3 (orders 2, 1 and 0) and a Queue of curvature radii a width of 1 (single value).

And the following methods:

- `is_empty()`: Whether Queue.items contains elements or not.
- `enqueue()`: Insert an element at the end of the Queue (figuratively -- in reality it is inserted at position 0).
- `dequeue()`: Remove (pop) the oldest element in the Queue, i.e. first inserted in the Queue.
- `size()`: The number of items in `Queue.items`.
- `mean()`: Returns a Numpy array of width == `Queue.width` containing the means by sub-element position of the elements in the Queue.
- `median()`: Returns a Numpy array of width == `Queue.width` containing the medians by sub-element position of the elements in the Queue. Unused in the submitted project but might be useful to better eliminate outliers when validating a polynomial fit.


** Sliding window search:**

The code for the sliding windows approach is located from line 247 in `lane_detection_pipeline.py`. It is based on code provided in the lectures but tweaked to make use of the Line() class. I defined two line objects, one for the left line and one for the right. The main steps are:

- Make a "histogram" of the lower half of the binary map to find the peaks corresponding to the left and right lines
- Define windows. I used a width of 80 pixels and 9 windows vertically. We will therefore have 9 layers vertically, starting from the bottom.
- For the first layer (bottom layer), draw windows around the two histogram peaks
- Store the non-zero pixel coordinates
- Move one layer up and store the non-zero pixel coordinates, then recenter the windows around the mean x coordinate of these pixels as long as we found a sufficient number of pixels within the window (I used `minpix = 30`). Repeat this operation until the top of the image is reached.
- Store all the detected non-zero pixels from search windows in the `.all_x` and `.all_y` attributes of the Line objects
- Fit a polynomial to each of these sets of pixels, using `Line.fit_poly()`
- Calculate the pixels for each of the fitted line using `Line.predict_poly()`
- Draw the lane as a filled polygon bounded by the fitted lines
- Calculate the car's position relative to the lane
- Calculate the average radius of curvature (average of left and right)

![alt text][image5]


** Search around latest fitted line:**

Once we have a valid line fit, the next search is easier as we know roughly where to look for candidate line pixels. This approach is implemented in `search_around_line()`, from line 382 in `lane_detection_pipeline.py`. The main steps are:

- Get the latest valid fit
- Identify all pixels within `margin` (horizontally) of the fitted lines
- From these, extract the non-zero pixels
- Fit a new polynomial to each of these sets of pixels, using `Line.fit_poly()`
- Calculate the pixels for each of the newly fitted line using `Line.predict_poly()`
- Draw the lane as a filled polygon bounded by the newly fitted lines
- Calculate the car's position relative to the lane
- Calculate the average radius of curvature (average of left and right)

Here is an example using the same test image as in section 2:

![alt text][image6]

**Coordinates conversion:**

One important point to keep in mind is that we want real-world numbers for the curvature radius and the position in line, in meters (being European myself). The conversion factors are different for the x and y axes therefore we need to convert each set of coordinates to meters accordingly before fitting the polynomial, then convert back to pixel when displaying plots on screen.

The conversion constants that I used were determined using information from the US government highway specifications for width (3.7m on average for highway lanes). For depth (or height in the bird's-eye view), I used 30 meters as the image depth as suggested in the lectures. I found that this gave me sensible curvature numbers.

As a result, the constants are:

| Direction | Ratio              |
|:---------:|:------------------:|
|x	    | 3.7 / 815 = 4.54e-3|
|y          | 30 / 720 = 4.17e-2 |


####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

As explained earlier, each line's curvature and position are calculated in class methods `Line.compute_curvature()` (line 103) and `Line.base_line_position()` respectively, in `Line_class.py`.
Each of these values is then average between the left and right lines.

**Radius of curvature:**

This value is calculated at the bottom edge of the image for each line. The displayed value is the average of the left and right line curvatures.
The formula is:

$$
R = \frac{\sqrt{\big[ 1 + (2 \, \beta_2 \, y_{max} + \beta_1)^2 \big]^3}}{2|\beta_2|}
$$

where $\beta_2, \beta_1, \beta_0$ are the 2nd order, 1st order and intercept coefficients of the fitted polynomial, respectively.

In code, this means:

```
y_eval = np.max(y)
radius = np.asscalar(((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0] + 0.0001))
```

*Notes:* 

 - The addition of 0.0001 to the denominator is to avoid divisions by zero, even though this would be a very unlikely event.
 - This assumes `y_eval` is expressed in real-world coordinates

**Position in lane:**

Again, this is calculated for each line and the average of both is displayed. I simply pass $y_{max}$ (corresponding to the bottom edge of the image converted to real-world coordinates) to the fitted polynomials to get the corresponding x values for left and right.


####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I then applied a reverse warping on the lane identified in step 4 and combined the resulting image to the original. I then added the position in lane and curvature radius calculated in step 5. The end-result looks like this:

![alt_text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

It is when working on a movie clip that the Line class defined earlier really becomes usefull. It allows sanity checks of fitted lines by comparison to previous frames and smoothing by averaging line coefficients over several frames (I used 5). Whenever the `search_around_line()` function didn't find a valid line for 3 successive frames, `window_search()` is used.

To make to clip more interesting, I overlaid the colored binary map with the fitted lines. We can see when the model had to resort to sliding window search because searching around previous lines didn't give good enough results.

Here's a [link to my video result](./project_video_test.mp4)


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As with any model, the main issue is tuning. In this case, we have many hyper-parameters: gradient kernel size, gradient and H, L and S thresholds, search window dimensions, minimum number of pixels to find before recentering a search window, number of frames to average upon, number of frames to try searching around previous lines for, position and radius tolerances.
As explained earlier, I was able to do without y-gradient, gradient magnitude and gradient direction, which would have added another 6 parameters).

It was a lengthy process trying different values for each. I also found that working on movie clips posed new problems entirely compared to still images.

The other problem, which is related, is that it is hard to find a set of parameters that generalizes well. After tuning my parameters for the first video, I found that the pipeline did not work as well with the challenge video. In particular, longitudinal tarmac junctions were confused with lines. To mitigate this problem, the starting points for the window search could be restricted to (for instance) the left-most and right-most thirds of the screen, instead of halves.

The harder challenge video is another beast entirely. The curves are much sharper which means a lot of external elements make it into the search zone and confuse the polynomial fit. We cannot restrict the region of interest much further because the road is so twisty that only small portions of it would fit into the ROI. I believe that somehow correlating the ROI's orientation to the previously recorded radius of curvature (or the steering angle, in a real car) would help. 

The dramatic changes in brightness and camera adjustment delays means we sometimes lose track of lines entirely. They also cause a lot of reflections in the windshield. This particular problem can be fixed by installing the camera right against the glass, although other issues such as vibrations might arise.

Finally, in a real-world application, we would need to find ways to significantly increase processing speed. On my 2 year-old desktop PC, I was reaching around 6 processed frames per second (including writing to the hard drive, which is not an SSD), which I believe would be somewhat insufficient in a situation where everything needs to be done in real-time, especially as once the lane has been identified, some other algorithms must use that information to guide the vehicle, which would add significant processing time. From that point of view, the convolutional approach might be better if a matrix multiplication implementation can be found.



