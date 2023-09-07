# 3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms
This repository contains an implementation of two-view and multi-view stereo algorithms for 3D reconstruction from multiple 2D viewpoints. 


# Multi-View Stereo Reconstruction System

## Introduction

This repository showcases the development and implementation of an advanced Multi-View Stereo Reconstruction system. Leveraging state-of-the-art techniques like Two-View Stereo and Plane-Sweep Stereo, the system adeptly transforms 2D image perspectives into intricate and accurate 3D reconstructions. The insights drawn from such spatial data representation have vast implications in the realm of digital image processing.

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/f09543d9-c0ff-475b-afa3-07382acea426)

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/3c7c831f-3b20-4faf-86e5-1d3175ec586a)


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

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/5d077ee8-5552-498b-aad8-820b89171f0b)

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/4cd1aad7-46b9-45e0-bce8-ef8daab737f4)


#### 3D Reconstruction:
- Converted depth maps into 3D point clouds using back-projection.
- Incorporated multi-pair aggregation which involves fusing depth information from multiple image pairs to improve depth estimation reliability.

### 2. Plane-sweep Stereo:

#### Multiple Views Processing:
- Set a central view as the primary reference and processed five different camera viewpoints simultaneously.

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/8dd5f5b8-38bf-43c9-b7ce-1da2b49bcbfd)



#### Homography & Warping:
- For each depth hypothesis, applied homography to warp all images onto the reference plane of the central image.
- Leveraged GPU acceleration to expedite the warping process, ensuring real-time processing for high-resolution images.

#### Cost Computation & Aggregation:
- Designed an innovative cost aggregation algorithm that computes the disparity of each pixel in the reference image by comparing it with every other warped image.
- Cost values were then aggregated across all views, refining depth estimation for areas with occlusions or repetitive patterns.

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/edc04a40-e61f-4145-8153-cadb60094292)

#### Disparity Optimization:
- Implemented graph cuts and dynamic programming methods to optimize the disparity map, ensuring spatial consistency and minimizing disparities across object boundaries.

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/ab43f43f-e754-40a7-a9a3-de2fa6aee5b5)

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/bf88b0b1-d47b-4aa8-ada9-6c4bfa1189ea)

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/9ede7887-db1c-452b-80cb-d448462cf129)


![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/7c42796e-1489-47ba-a551-57bfc518a5b3)


## Conclusion

Through this repository, the prowess of the Multi-View Stereo Reconstruction project is evident. Merging traditional techniques with cutting-edge algorithms, the system offers both accuracy and efficiency in 3D image reconstruction. It holds potential for transformational impacts in areas such as gaming, architecture, and augmented reality applications.

### RECONSTRUCTION WITH CLOUD POINTS:

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/3f252f0f-fe9e-47b4-8b2c-4da7811112ac)


### SSD Kernel

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/a7cc7fbd-cad0-4dc1-a6c7-fcfe5e9d01ec)

### SAD Kernel

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/9b032851-f0dd-4632-9c4e-312dbac665ae)

### ZNCC Kernel

![image](https://github.com/Saibernard/3D-Scene-Reconstruction-from-Multi-View-Stereo-Algorithms/assets/112599512/abc6c894-fb03-4b76-93b2-bdf07c40b11e)





