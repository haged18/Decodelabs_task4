# Object Detection — MobileNet-SSD

A Python script that detects objects in images using a pre-trained MobileNet-SSD model with OpenCV. Built as part of the DecodeLabs AI Industrial Training Kit (Project 4).

---

## What it does

- Loads a pre-trained MobileNet-SSD deep learning model
- Pre-processes images using blob construction (mean subtraction + resize to 300×300)
- Detects and labels objects with bounding boxes
- Filters results to only show detections with **≥ 80% confidence**
- Saves output images with bounding boxes drawn on them
- Processes all `.jpg` and `.png` images in the folder automatically

---

## Requirements

Install dependencies:

```bash
pip install opencv-python numpy
```

---

## Files in this repo

| File | Description |
|------|-------------|
| `object_detection.py` | Main detection script |
| `MobileNetSSD_deploy.prototxt` | Model architecture file |
| `MobileNetSSD_deploy.caffemodel` | Pre-trained model weights |
| `how-to-be-a-people-person-1662995088.jpg` | Sample test image |

---

## How to run

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

2. Install dependencies:
```bash
pip install opencv-python numpy
```

3. Add any `.jpg` or `.png` images you want to test into the folder.

4. Run the script:
```bash
python object_detection.py
```

5. Output images will be saved as `output_yourimage.jpg` in the same folder.

---

## Detectable objects

The model can detect 20 object classes:

`aeroplane` `bicycle` `bird` `boat` `bottle` `bus` `car` `cat` `chair` `cow` `diningtable` `dog` `horse` `motorbike` `person` `pottedplant` `sheep` `sofa` `train` `tvmonitor`

---

## Example output

Each detected object is labeled with its class name and confidence score:

```
[INFO] Loading model...
[INFO] Model loaded successfully.
[INFO] Running detection (confidence threshold: 80%)...

[RESULTS] 1 object(s) detected:
  1. person    Confidence: 91.3%   BBox: x=45, y=12, w=320, h=480

[INFO] Output saved to: output_how-to-be-a-people-person-1662995088.jpg
```

---

## How it works

1. **Blob construction** — image is resized to 300×300 and mean-subtracted
2. **Forward pass** — MobileNet-SSD runs inference in a single shot
3. **Confidence filter** — detections below 80% are discarded
4. **Bounding box scaling** — normalized coordinates are scaled to actual pixel dimensions
5. **Visual output** — boxes and labels are drawn and saved

---

## Built with

- [OpenCV](https://opencv.org/) — computer vision and DNN module
- [MobileNet-SSD](https://github.com/djmv/MobilNet_SSD_opencv) — pre-trained object detection model
- [NumPy](https://numpy.org/) — array operations
