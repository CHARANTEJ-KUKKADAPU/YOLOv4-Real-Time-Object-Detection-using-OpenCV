# YOLOv4 Real-Time Object Detection using OpenCV

## Overview

This project performs **real-time object detection** using the **YOLOv4 (You Only Look Once)** deep learning model with Python and OpenCV.
The webcam feed is processed frame-by-frame, and detected objects are highlighted with bounding boxes and confidence scores.

The model uses the **COCO dataset**, which can detect 80 common object categories such as people, cars, dogs, bottles, chairs, etc.

---

## Features

* Real-time webcam object detection
* YOLOv4 deep learning model
* OpenCV DNN module
* Non-Maximum Suppression (NMS)
* Bounding boxes with labels and confidence scores
* Live visualization using Matplotlib

---

## Technologies Used

* Python
* OpenCV
* NumPy
* Matplotlib
* YOLOv4
* COCO Dataset

---

## Project Structure

```bash
├── yolov4.weights
├── yolov4.cfg
├── coco.names
├── object_detection.py
└── README.md
```

---

## Requirements

Install the required libraries using:

```bash
pip install opencv-python numpy matplotlib ipython
```

---

## Download YOLOv4 Files

Download the following files and place them in the project directory.

### 1. YOLOv4 Weights

Download:

```bash
https://github.com/AlexeyAB/darknet/releases
```

### 2. YOLOv4 Configuration File

Download:

```bash
https://github.com/AlexeyAB/darknet/blob/master/cfg/yolov4.cfg
```

### 3. COCO Names File

Download:

```bash
https://github.com/pjreddie/darknet/blob/master/data/coco.names
```

---

## How to Run

Run the Python script:

```bash
python object_detection.py
```

Press **Ctrl + C** to stop the program.

---

## Working Principle

1. Capture video frames from webcam
2. Convert frames into blobs
3. Pass blobs into YOLOv4 network
4. Detect objects and confidence scores
5. Apply Non-Max Suppression
6. Draw bounding boxes and labels
7. Display output frame

---

## Sample Output

The webcam feed displays:

* Object labels
* Confidence percentages
* Green bounding boxes around detected objects

Example:

```text
person 0.95
bottle 0.82
chair 0.77
```

---

## Code Highlights

### Load YOLOv4 Model

```python
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
```

### Webcam Capture

```python
cap = cv2.VideoCapture(0)
```

### Blob Creation

```python
blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
```

### Non-Max Suppression

```python
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
```

---

## Applications

* Smart surveillance systems
* Autonomous vehicles
* Human detection systems
* Traffic monitoring
* AI-based security systems

---

## Future Improvements

* GPU acceleration using CUDA
* Save detected video output
* Count detected objects
* Custom object training
* FPS optimization

---

## Author

**Charan Tej**

---

## License

This project is for educational and learning purposes.
