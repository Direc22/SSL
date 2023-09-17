import cap
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os
import tkinter as tk
from tkinter import ttk

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)
classifier = Classifier("model2/keras_model.h5", "model2/labels.txt")

offset = 20
imgSize = 300
counter = 0

img_folder = "IMG2"

labels = ['0','1','2','3','4','5','6','7','8','9','10']
labels_v2 = ["ආයුබෝවන්", "ඔයා හොදින්ද", "කරුණාකරල", "කොහෙද යන්නේ", "බඩගිනි","බොහොම ස්තූතියි", "මට අසනීපයි", "මම හොදින්", "සමාවන්න", "සුබ උදෑසනක්",]

def update_label(prediction_text):
    label.config(text=prediction_text)

def toggle_camera(cap=None):
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

def toggle_hand_state():
    global is_hand_open, word
    is_hand_open = not is_hand_open
    if not is_hand_open:
        word += current_letter

def print_predicted_word():
    global word
    if word:
        text_box.insert("end", word + " ")
        word = ""

window = tk.Tk()
window.title("Sign Language Recognition")
window.geometry("800x400")

label = ttk.Label(window, text="", font=("Helvetica", 24))
label.pack(pady=20)

is_camera_on = False
camera_button = ttk.Button(window, text="Start Camera", command=toggle_camera)
camera_button.pack(pady=20)

is_hand_open = True
current_letter = ""
word = ""
hand_toggle_button = ttk.Button(window, text="Toggle Hand", command=toggle_hand_state)
hand_toggle_button.pack(pady=20)

print_word_button = ttk.Button(window, text="Print the Word", command=print_predicted_word)
print_word_button.pack(pady=20)

text_box = tk.Text(window, height=10, width=40, font=("Helvetica", 16))
text_box.pack(pady=20)

def update_camera():
    global is_hand_open, current_letter
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
            if is_hand_open:
                current_letter = labels_v2[index]
                update_label(current_letter)

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            if is_hand_open:
                current_letter = labels_v2[index]
                update_label(current_letter)

        cv2.putText(imgOutput, labels_v2[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

    cv2.imshow("Image", imgOutput)

    if is_camera_on:
        window.after(20, update_camera)

window.mainloop()
