import cv2
import numpy as np
import urllib.request
import os

# ──────────────────────────────────────────────
# STEP 1: Download MobileNet-SSD model files
# ──────────────────────────────────────────────
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
MODEL_URL    = "https://drive.usercontent.google.com/download?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc&export=download"

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL    = "MobileNetSSD_deploy.caffemodel"

def download_file(url, filename):
    if not os.path.exists(filename):
        print(f"[INFO] Downloading {filename}...")
        urllib.request.urlretrieve(url, filename)
        print(f"[INFO] Downloaded {filename}")
    else:
        print(f"[INFO] {filename} already exists, skipping download.")

PROTOTXT = "MobileNetSSD_deploy.prototxt"
MODEL    = "MobileNetSSD_deploy.caffemodel"


# ──────────────────────────────────────────────
# STEP 2: Define class labels (COCO subset)
# ──────────────────────────────────────────────
CLASSES = [
    "background", "aeroplane", "bicycle",  "bird",   "boat",
    "bottle",     "bus",       "car",       "cat",    "chair",
    "cow",        "diningtable","dog",      "horse",  "motorbike",
    "person",     "pottedplant","sheep",   "sofa",   "train",
    "tvmonitor"
]

# Assign a unique BGR color to each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype="uint8")


# ──────────────────────────────────────────────
# STEP 3: Load pre-trained model
# ──────────────────────────────────────────────
def load_model(prototxt, model):
    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    print("[INFO] Model loaded successfully.")
    return net


# ──────────────────────────────────────────────
# STEP 4: Pre-process image
# ──────────────────────────────────────────────
def preprocess(image):
    """
    Convert image to a 4D blob:
    - Mean subtraction: (127.5, 127.5, 127.5)
    - Scale factor:     1/127.5 = 0.007843
    - Resize to:        300x300 (MobileNet-SSD input requirement)
    """
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5)
    )
    return blob


# ──────────────────────────────────────────────
# STEP 5: Run detection + confidence filter
# ──────────────────────────────────────────────
def detect_objects(net, image, confidence_threshold=0.80):
    """
    Run forward pass and filter detections by confidence >= 80%.
    Returns list of (label, confidence, x, y, w, h).
    """
    (h, w) = image.shape[:2]
    blob = preprocess(image)

    net.setInput(blob)
    detections = net.forward()

    results = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_threshold:
            class_idx = int(detections[0, 0, i, 1])
            label     = CLASSES[class_idx]

            # Scale normalized coordinates → actual pixel coordinates
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            results.append({
                "label":      label,
                "confidence": float(confidence),
                "x": startX,
                "y": startY,
                "w": endX - startX,
                "h": endY - startY,
                "endX": endX,
                "endY": endY
            })

    return results


# ──────────────────────────────────────────────
# STEP 6: Draw bounding boxes on image
# ──────────────────────────────────────────────
def draw_detections(image, results):
    output = image.copy()

    for det in results:
        label      = det["label"]
        confidence = det["confidence"]
        color      = [int(c) for c in COLORS[CLASSES.index(label)]]

        # Draw bounding box
        cv2.rectangle(output,
                      (det["x"], det["y"]),
                      (det["endX"], det["endY"]),
                      color, thickness=2)

        # Draw label + confidence score
        text = f"{label}: {confidence * 100:.1f}%"
        y_pos = det["y"] - 10 if det["y"] - 10 > 10 else det["y"] + 20
        cv2.putText(output, text,
                    (det["x"], y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.6,
                    color=color,
                    thickness=2)

    return output


# ──────────────────────────────────────────────
# STEP 7: Main pipeline
# ──────────────────────────────────────────────
def run_detection(image_path, output_path="output.jpg", confidence_threshold=0.80):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return

    print(f"[INFO] Image loaded: {image.shape[1]}x{image.shape[0]} px")

    # Load model
    net = load_model(PROTOTXT, MODEL)

    # Detect objects
    print(f"[INFO] Running detection (confidence threshold: {confidence_threshold*100:.0f}%)...")
    results = detect_objects(net, image, confidence_threshold)

    # Print results
    if results:
        print(f"\n[RESULTS] {len(results)} object(s) detected above {confidence_threshold*100:.0f}% confidence:\n")
        for i, det in enumerate(results, 1):
            print(f"  {i}. {det['label']:<15} Confidence: {det['confidence']*100:.1f}%  "
                  f"BBox: x={det['x']}, y={det['y']}, w={det['w']}, h={det['h']}")
    else:
        print(f"[INFO] No objects detected above {confidence_threshold*100:.0f}% confidence.")

    # Draw and save output
    output_image = draw_detections(image, results)
    cv2.imwrite(output_path, output_image)
    print(f"\n[INFO] Output saved to: {output_path}")

    # Display image (comment out if running headless)
    cv2.imshow("Object Detection", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return results


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    import glob

    images = [f for f in glob.glob("*.jpg") + glob.glob("*.png") 
              if not f.startswith("output_")]

    for img in images:
        run_detection(img, output_path="output_" + img, confidence_threshold=0.80)