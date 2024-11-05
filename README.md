
# Convergence Algorithm 
Convergence Algorithm makes use of python libraries such as opencv and skimage to process images of point clouds and calculate the structural similarity index (SSIM). This algorithm allows for visualization of 2D defects/dissimialrites in two point clouds.


After the defect detection algorithm (one of my repository), this convergence algorithm is the next step for 2D defect detection of point clouds. The true defect output file from defect detection algorithm is taken as input in the convergence algorithm along with the defected and ideal PCD.


## Installation

The installation guide for required libraries is present in the file "Steps to set up convergence final deliverable algorithm on PC"


    
## Pre-Requisites
After the defect detection algorithm (one of my repository), this convergence algorithm is the next step for 2D defect detection of point clouds. The true defect output file from defect detection algorithm is taken as input in the convergence algorithm along with the defected and ideal PCD.
## Usage/Examples
The algorithm is general for any point cloud. The algorithm can take automatic snapshots of the 3D point clouds from different views like the right view, left view, front view, back view and top view. All the user has to do is input the point cloud files required by the algorithm. 

Next the algorithm can align the point cloud images and implement the necessary pre-processing required for better and accurate results, such as the conversion to grayscale images.

The algorithm can generate heatmaps as well for better visualization of SSIM values and defects.

The algorithm offers a pixel-by-pixel comparison of the ideal and defective images. It can adjust the tolerance factor to control the level of detail in this comparison. A higher tolerance factor makes the comparison stricter, allowing us to identify even the most minor differences between the two images.

