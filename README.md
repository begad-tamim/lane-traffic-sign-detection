# Lane Traffic Sign Detection Project

## Overview
This project aims to implement real-time detection of road lanes, curves, and traffic signs using computer vision techniques. The project utilizes the YOLOv8 dataset for traffic sign detection and OpenCV for lane detection.

## Objectives
- Detect and track road lanes in real-time video streams.
- Identify and classify traffic signs using YOLOv8.
- Provide a comprehensive solution for lane and traffic sign detection that can be integrated into autonomous driving systems.

## Methodology
1. **Data Preparation**: The dataset is organized into training, validation, and test sets. Each set contains images and corresponding labels formatted for YOLO.
2. **Lane Detection**: Implement lane detection using OpenCV techniques such as Canny edge detection and Hough Line Transform.
3. **Traffic Sign Detection**: Utilize YOLOv8 for detecting and classifying traffic signs in images.
4. **Integration**: Combine lane and traffic sign detection into a single application that processes video input in real-time.

## Project Structure
```
lane-traffic-sign-detection
├── data
│   ├── data.yaml
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   └── labels
│   └── valid
│       ├── images
│       └── labels
├── notebooks
│   └── lanes_traffic_signs.ipynb
├── src
│   ├── lane_detection.py
│   ├── traffic_sign_detection.py
│   └── utils.py
├── requirements.txt
└── README.md
```

## Installation
To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/begad-tamim/lane-traffic-sign-detection.git
cd lane-traffic-sign-detection
pip install -r requirements.txt
```

## Usage
1. Open the Jupyter notebook located in the `notebooks` directory.
2. Follow the instructions in the notebook to load the dataset, preprocess the images, and run the lane and traffic sign detection algorithms.
3. Visualize the results in real-time.

## Acknowledgments
- This project utilizes the YOLOv8 dataset for traffic sign detection.
- OpenCV is used for lane detection and image processing tasks.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
