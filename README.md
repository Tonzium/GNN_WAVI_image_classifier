# Enhanced Data Augmentation Techniques for Improved Deep Learning Classification in Wafer Automated Visual Inspection (WAVI)

## Overview
This repository contains a Python-based solution designed to address the challenge of limited labeled data in the context of Wafer Automatic Visual Inspection (WAVI).
By leveraging advanced data augmentation techniques, approach significantly enhances the training dataset, enabling high classification accuracy with fewer original samples.

## Features
- Multiple Augmentation Techniques: Implements a variety of augmentation methods, including zoom, rotation, noise addition, shear transformations, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
- Focused on WAVI: Tailored specifically for the unique challenges of wafer inspection, where precision and reliability are paramount.
- Efficient Training: Achieve remarkable classification accuracy with a substantially augmented dataset derived from a relatively low number of labeled examples.

## How It Works
The codebase includes scripts that creates augmented data from a given set of wafer images. These augmentations simulate various realistic distortions and adjustments, thereby allowing the model to learn from a more diverse set of data points.

## Data Augmentation Methods
- Zoom: Adjusts the scale of images to simulate different distances during inspection.
- Rotation: Alters the orientation of wafers to ensure robustness against positional variances.
- Noise Addition: Introduces random pixel-level noise to mimic real-world imperfections.
- Shear: Applies shear transformations to model wafer deformations.
- CLAHE: Enhances the contrast of the images, improving feature visibility and discrimination.
