# 3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms
This repository contains an implementation of two-view and multi-view stereo algorithms for 3D reconstruction from multiple 2D viewpoints. 


# Multi-View Stereo Reconstruction System

## Introduction

This repository showcases the development and implementation of an advanced Multi-View Stereo Reconstruction system. Leveraging state-of-the-art techniques like Two-View Stereo and Plane-Sweep Stereo, the system adeptly transforms 2D image perspectives into intricate and accurate 3D reconstructions. The insights drawn from such spatial data representation have vast implications in the realm of digital image processing.

## Technical Details

### 1. Two-View Stereo:

#### Image Rectification:
- Utilized the essential matrix to estimate the epipolar geometry of the image pair.
- Applied a transformation matrix to both images, aligning epipolar lines horizontally for simplified disparity estimation.

#### Disparity Estimation:
- Introduced three similarity metrics:
  - Sum of Squared Differences (SSD)
  - Sum of Absolute Differences (SAD)
  - Zero-Mean Normalized Cross-Correlation (ZNCC)
- By analyzing pixel-wise differences, generated accurate disparity maps for both left-to-right and right-to-left image pairs.

#### Depth Estimation:
- Calculated depth maps using camera calibration matrices and disparity values, resulting in pinpoint depth information for each pixel.
- Applied bilateral filtering for smooth depth transitions and to reduce noise.

#### 3D Reconstruction:
- Converted depth maps into 3D point clouds using back-projection.
- Incorporated multi-pair aggregation which involves fusing depth information from multiple image pairs to improve depth estimation reliability.

### 2. Plane-sweep Stereo:

#### Multiple Views Processing:
- Set a central view as the primary reference and processed five different camera viewpoints simultaneously.

#### Homography & Warping:
- For each depth hypothesis, applied homography to warp all images onto the reference plane of the central image.
- Leveraged GPU acceleration to expedite the warping process, ensuring real-time processing for high-resolution images.

#### Cost Computation & Aggregation:
- Designed an innovative cost aggregation algorithm that computes the disparity of each pixel in the reference image by comparing it with every other warped image.
- Cost values were then aggregated across all views, refining depth estimation for areas with occlusions or repetitive patterns.

#### Disparity Optimization:
- Implemented graph cuts and dynamic programming methods to optimize the disparity map, ensuring spatial consistency and minimizing disparities across object boundaries.

## Conclusion

Through this repository, the prowess of the Multi-View Stereo Reconstruction project is evident. Merging traditional techniques with cutting-edge algorithms, the system offers both accuracy and efficiency in 3D image reconstruction. It holds potential for transformational impacts in areas such as gaming, architecture, and augmented reality applications.
