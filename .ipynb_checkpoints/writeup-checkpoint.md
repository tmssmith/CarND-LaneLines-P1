# **Finding Lane Lines on the Road** 

## Writeup

[//]: # (Image References)

[image1]: ./test_images/solidYellowCurve2.jpg "Input image"

[image2]: ./test_images_output/solidYellowCurve2_colorMask.jpg "Color Detection"

[image3]: ./test_images_output/solidYellowCurve2_canny.jpg "Canny Edges"

[image4]: ./test_images_output/solidYellowCurve2_comb.jpg "Combined color detection and Canny edges"

[image5]: ./test_images_output/solidYellowCurve2_masked.jpg "Region of interest"

[image6]: ./test_images_output/solidYellowCurve2_hough.jpg "Hough lines"

[image7]: ./test_images_output/solidYellowCurve2_overlay.jpg "Final image"

---

### Reflection

### 1 - Pipeline

The pipeline consists of 5 main steps, outlined in the subsections below. Two tools were written in python to aid with the parameterisation of the pipeline. These are briefly in outlined in Section 2 - Parameterisation.

#### 1.1 - Color detection

When attempting the challenge stage of this project the edge detection did not perform satisfactorily, particularly in areas of high brightness or deep shadow. A color detection step was added to the pipeline to improve performance.

In this step, pixels that may represent lane lines are selected on the basis of color. The input image is converted from RGB to HSL color as this better reflects human interpretation of color. Thresholds are defined for the colors white and yellow and pixels with color in these ranges are returned as white in the color mask. Threshold values for white and yellow were found using the color_detection.py parameterisation tool developed for this project. 

An example result is shown below for solidYellowCurve2.jpg:

![alt text][image1]
![alt text][image2]

#### 1.2 - Edge detection

Edge detection is carried out using Canny edge detection on a Gaussian-filtered grayscale version of the input image. Parameters for Gaussian kernel size and Canny thresholds were found using the lanefind_gui.py parameterisation tool developed for this project.

An example result is shown below for solidYellowCurve2.jpg:

![alt_text][image3]

The output of the color detection step is combined with the outputs of this edge detection step. This combined image is used as the input for the following stages of the pipeline. 

An example combined image is shown below for solidYellowCurve2.jpg:

This was found to perform better in subsequent stages than using either image alone.

![alt_text][image4]

#### 1.3 - Region of Interest

A region of interest mask is applied to the combined image. This mask is a 4 sided polygon, defined by its 4 vertices. It corresponds to the region of the input image where we expect to see lane lines. By removing information outside of this region we improve the perfomance of the following stage, line detection.

An example region of interest is shown below, applied to the input image solidYellowCurver2.jpg for ease of visualisation. Note that in the pipleine the region of interest is applied to the combined image, not the input image.

![alt_text][image5]


#### 1.4 - Line detection

A probabilistic Hough transform is used to identify lines within the region of interest of the combined image. Parameters for the Hough transform were found using the lanefind_gui.py parameterisation tool developed for this project.

An example result of the Hough transform is shown below for solidYellowCurve2.jpg:

![alt_text][image6]


#### 1.5 - Fit lane lines

The output of the Hough transform is a series of individual lines, while the desired output is a single line for both the left lane and the right lane. To achieve this a line fitting step is added to the pipeline.

In this step the Hough lines are separated on the basis of gradient, with lines with a negative gradient assigned to the left lane, and positive gradients assigned to the right lane. A threshold value with magnitude 0.5 is applied to the gradients before assignment to remove any near-horizontal lines detected by the Hough transform, as horizontal lines are unlikely to correspond to lane lines.

With the Hough lines separated into left and right lanes, a first order polynomial line of best fit is plotted through the points of all the Hough lines of each lane individually. A second order polynomial line was trialled but did not perform well. This is discussed further in the shortcoming and improvements sections below.

The final output image for solidYellowCurve2.jpg is shown below.

![alt_test][image7]

### 2 - Parameterisation

Manual parameterisation is a critically important part of this pipeline, with many interconnected parameters needing tuning to give good results. Two parameterisation tools were developed to make it easier and faster for a user to tune parameters. Both tools are included in this repository and are discussed below:

#### 2.1 - Color Detection (color_detection.py)

This tool opens the given image (\*.jpg) or video (\*.mp4) file and presents the user with a set of trackbars to change the HSL values for two sets of color thresholds. The image or video is updated in real-time giving instant user feedback.   

Example command: `python color_detection.py test_images/solidYellowCurve2.jpg`

#### 2.2 - Lane Finding (lanefind_gui.py)

This tool opens the given image (\*.jpg) or video (\*.mp4) file and presents the user with a set of trackbars to change the parameters for the lane finding pipeline. The image or video is updated in real-time with the proposed lane lines overlayed on the base image giving instant user feedback.

Example command: `python lanefind_gui.py test_videos/solidYellowLeft.jpg`

### 3 - Shortcomings 

In general this pipeline is very sensitive to parameterisation. Although the parameterisation tools developed help with this, it is still a difficult task and small changes in certain values can have a large effect on the performance of the pipeline. Also, different images and videos work best with different parameters.

The pipeline struggles with images that contain areas of low contrast for example light or heavily shadowed road surfaces as present in the challenge video.

The challenge video also contains curved roads. The current pipeline can only draw straight lane lines which doesn't give a good representation of curved lanes.

Although not present in any of the videos, the pipeline is likely to struggle if other lines are present in the road. Examples of this could include road surface cracking, exapansion joints at bridges, lane arrows or writing in the road.

The use of a fixed region of interest is likely to cause issues during lane change maneuvers, as only one lane may be present in the region of interest during a lane change.

### 4 - Improvements

Improving the lane line fit by using a second order polynomial line as the lane line best fit to match curved lane lines. This was attempted in this project but did not give good results and further work is needed.

Additional smoothing or filtering to the lane lines, for example a sliding window average on lane line position to improve consistency of lane line finding. However this would reduce responsiveness of the pipeline to changes in lane position.

To help with defining the region of interest and lane changes we could find lane lines at a cross section of the image near to the car (i.e the bottom of the frame) and define the region of interest from those points.