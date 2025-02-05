# Personal Protective Equipment (PPE) Detection

## Introduction

The PPE Detection System is a computer vision-based solution designed to enhance workplace safety by ensuring workers are wearing the required personal protective equipment (PPE), specifically focusing on helmet detection. The system leverages advanced machine learning algorithms to detect helmets in real-time from images or video streams captured in the working environment.

## Features
- **Real-Time Detection**: Continuously monitors the working environment to detect helmets.
- **High Accuracy**: Utilizes state-of-the-art object detection models (YOLOv10) to ensure high accuracy.
- **Scalable**: Can be scaled to monitor multiple areas simultaneously.
- **Alerts and Notifications**: Sends alerts when a worker is detected without a helmet.
- **Customizable**: Can be extended to detect other types of PPE such as safety glasses, gloves, and vests.

## System Architecture
1. **Image Acquisition**: High-resolution cameras capture images or video streams from the work environment.
2. **Preprocessing**: Images are preprocessed to enhance quality and reduce noise.
3. **Helmet Detection Algorithm**: Uses machine learning models (e.g., YOLO, SSD, Faster R-CNN) to detect helmets.
4. **Alert System**: Triggers alerts if a worker is detected without a helmet.

## Requirements
- **Hardware**:
  - High-resolution cameras (Openncc camera)
  - Computer or server with a GPU for processing (nvidia jetson orin nano)
- **Software**:
  - C++
  - OpenCV
  - Libtorch
  - Pre-trained object detection models (can be any architecture but needs to be torchscript model which is exported by torch.jit.save())

## Installation
- **install openncc lib**:
  - git clone https://github.com/EyecloudAi/opennccframe.git
  - sudo apt-get install libusb-dev libusb-1.0-0-dev
  - cd opennccframe
  - sudo ./install.sh
  - sudo apt-get install libopencv-dev
  - sudo ln -s /usr/include/opencv4/opencv2/ /usr/include/opencv2

- **install pytorch for jetson**:
  - wget https://nvidia.box.com/shared/static/zvultzsmd4iuheykxy17s4l2n91ylpl8.whl -O torch-2.3.0-cp310-cp310-linux_aarch64.whl
  - sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
  - pip3 install 'Cython<3'
  - pip3 install numpy torch-2.3.0-cp310-cp310-linux_aarch64.whl

- **build executable file**:
  - mkdir build && cd build
  - (if cmake cant find nvcc): export PATH=/usr/local/cuda/bin:$PATH
  - cmake ..
  - make

## Usage
  - sudo ./HD /dev/openncc ../HD.torchscript
