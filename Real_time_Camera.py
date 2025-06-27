# coding: utf-8
import argparse
import cv2
import numpy as np
import os
import sys
import threading
import time
import winsound  # For beep alert (Windows only)

from keras.applications.resnet50 import ResNet50
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam

# =================== MODEL TRAINING ===================
def train_model():
    model_resnet50_conv = ResNet50(weights='imagenet', include_top=False)
    input = Input(shape=(224, 224, 3), name='image_input')
    output_resnet50_conv = model_resnet50_conv(input)
    x = Flatten()(output_resnet50_conv)
    x = Dense(10, activation='softmax', name='predictions')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# =================== PREDICTION DECODING ===================
def decode_predictions(preds):
    assert len(preds.shape) == 2 and preds.shape[1] == 10
    results = ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
    return results[np.argmax(preds)]

# =================== DISTRACTION ALERT LOGIC ===================
def trigger_alert(label):
    if label != 'c0':  # c0 is safe driving
        print(f"[ALERT] Distracted driving detected: {label}")
        winsound.Beep(1000, 400)  # Frequency 1000Hz, Duration 400ms
        with open("distraction_log.txt", "a") as f:
            f.write(f"{label} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

# =================== THREAD FOR MODEL INFERENCE ===================
label = ''
frame = None
kfold_weights_path = os.path.join('weights_kfold_vgg16_2.h5')

class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global label
        print("[INFO] Loading model...")
        self.model = train_model()
        self.model.load_weights(kfold_weights_path)
        print("[INFO] Model loaded. Starting predictions...")
        while frame is not None:
            label = self.predict(frame)

    def predict(self, frame):
        X_test = [frame]
        test_data = np.array(X_test, dtype=np.float16)
        mean_pixel = [103.939, 116.779, 123.68]
        test_data[:, :, :, 0] -= mean_pixel[0]
        test_data[:, :, :, 1] -= mean_pixel[1]
        test_data[:, :, :, 2] -= mean_pixel[2]
        preds = self.model.predict(test_data)
        result = decode_predictions(preds)
        return result

# =================== MAIN CAMERA LOOP ===================
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("Camera initialized.")
else:
    cap.open()

keras_thread = MyThread()
keras_thread.start()

while True:
    ret, original = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    frame = cv2.resize(original, (224, 224))

    # Show the label on the video
    cv2.putText(original, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Alert on distraction
    trigger_alert(label)

    cv2.imshow("Driver Distraction Detection", original)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
frame = None
cv2.destroyAllWindows()
sys.exit()
