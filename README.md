## Advanced Lane Finding

This is my submission for project 4 in Udacity's Self-Driving Car Engineer Nanodegree.


Files
---

The repository contains a number of files, some useful, some less so:

- `lane_detection_pipeline.py`: The main code for the model. On Linux, make sure you have execution rights and type `./lane_detection_pipeline.py`
- `camera_calibration.py`: Code for calculating the camera's calibration parameters. These are then saved as a Pickle: `calibration_params.pkl`. Unless you delete the Pickle, you shouldn't need to execute this piece of code.
- `calibration_params.pkl`: See above.
`Line_class.py`: Creates an object class called Line() which is used in `lane_detection_pipeline.py`
- `frame_extractor.py`: A set of simple helper functions that extract frames from a video clip at regular intervals between two timestamps.
- `project_writeup.md`: The fullproject report.
- `project_video.mp4`, `challenge_video.mp4`, `harder_challenge_video.mp4`: Increasingly difficult videos to test the lane detection pipeline on
- `camera_cal` folder: Chessboard images for camera calibration
- `test_images`: Images to test the pipeline on.
- `output_images`: Various images produced by different parts of the pipeline and used to illustrate `project_writeup.md`.

This pipeline works well on the project video but further work is required before it performs well on the other two.