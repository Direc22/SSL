import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import tkinter as tk
from tkinter import ttk

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
counter = 0

img_folder = "IMG"

labels = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20']

labels_v2 = ["අ", "ආ", "ඇ", "ඉ", "ඊ", "උ", "ඌ", "එ", "ඒ", "ක්", "ග්", "ජ්", "ට්", "ද්", "ප්", "බ්", "ස්", "ය්", "හ්",
             "ව්"]

# Create a function to update the UI label with the prediction
def update_label(prediction_text):
    label.config(text=prediction_text)

# Create a function to start/stop the camera
def toggle_camera():
    global is_camera_on
    if is_camera_on:
        is_camera_on = False
        camera_button.config(text="Start Camera")
        cap.release()
        cv2.destroyAllWindows()
    else:
        is_camera_on = True
        camera_button.config(text="Stop Camera")
        cap = cv2.VideoCapture(0)
        update_camera()

# Create a function to continuously update the camera feed and predictions
def update_camera():
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            update_label(labels_v2[index])

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            update_label(labels_v2[index])

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

        image_path = os.path.join(img_folder, labels[index] + ".png")
        img_to_display = cv2.imread(image_path)

        cv2.imshow("Image Output", img_to_display)

    cv2.imshow("Image", imgOutput)

    if is_camera_on:
        window.after(20, update_camera)  # Schedule the next update

# Create the main window
window = tk.Tk()
window.title("Hand Gesture Recognition")
window.geometry("800x400")

# Create a label to display the prediction
label = ttk.Label(window, text="", font=("Helvetica", 24))
label.pack(pady=20)

# Create a button to start/stop the camera
is_camera_on = False
camera_button = ttk.Button(window, text="Start Camera", command=toggle_camera)
camera_button.pack(pady=20)

# Run the Tkinter main loop
window.mainloop()
