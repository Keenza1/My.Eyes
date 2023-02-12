import cv2
import numpy as np
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model("ssd_mobilenet_v2_coco_2018_03_29.h5")

# Load the input image
img = cv2.imread("input.jpg")

# Resize the image to the size expected by the model
img = cv2.resize(img, (300, 300))

# Preprocess the image for the model
img = np.array(img, dtype=np.float32)
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Make predictions
predictions = model.predict(img)

# Get the label and confidence of the prediction
labels = ["person", "bicycle", "car", "motorcycle", "airplane", "bus",          "train", "truck", "boat", "traffic light", "fire hydrant",          "stop sign", "parking meter", "bench", "bird", "cat", "dog",          "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",          "skis", "snowboard", "sports ball", "kite", "baseball bat",          "baseball glove", "skateboard", "surfboard", "tennis racket",          "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",          "banana", "apple", "sandwich", "orange", "broccoli", "carrot",          "hot dog", "pizza", "donut", "cake", "chair", "couch",          "potted plant", "bed", "dining table", "toilet", "tv", "laptop",          "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",          "toaster", "sink", "refrigerator", "book", "clock", "vase",          "scissors", "teddy bear", "hair drier", "toothbrush"]

classes, scores, boxes = [], [], []
for prediction in predictions[0]:
    if prediction[2] > 0.5:
        class_id = prediction[0].astype(int)
        score = prediction[2]
        x1, y1, x2, y2 = prediction[3:]

        classes.append(labels[class_id])
        scores.append(score)
        boxes.append((x1, y1, x2, y2))

# Draw boxes around the detected objects
for i, box in enumerate(boxes):
    x1, y1, x2, y2 = box
    x1 = int(x1 * img.shape[1])
    y1 = int(y1 * img.shape[0])
    x2 = int(x2 * img.shape[2])
    y2 = int(y2 * img.shape[0])
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = classes[i] + ": " + str(scores[i])
    y = y1 - 15 if y1 > 15 else y1 + 15
    cv2.putText(img, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

   # Show the image with the detected objects
   cv2.imshow("Detected Objects", img)
   cv2.waitKey(0)
   cv2.destroyAllWindows()
    
