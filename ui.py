import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import gradio as gr

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300

labels_v2 = [
    "අ", "ආ", "ඇ", "ඉ", "ඊ", "උ", "ඌ", "එ", "ඒ",
    "ක්", "ග්", "ජ්", "ට්", "ද්", "ප්",
    "බ්", "ස්", " ය්", " හ්", " ව්"
]


def recognize_sign_language(frame):
    global letter
    imgOutput = frame.copy()
    hands, _ = detector.findHands(frame)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = frame[y - offset:y + h + offset, x - offset:x + w + offset]

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
            letter = labels_v2[index]

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            letter = labels_v2[index]

        cv2.putText(imgOutput, letter, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

    return frame, letter


iface = gr.Interface(fn=recognize_sign_language,
                     inputs="webcam",
                     outputs=["image", "text"])
iface.launch()
