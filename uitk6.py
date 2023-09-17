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
    global cap  # Declare cap as a global variable
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
    global is_hand_open
    global current_letter
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

        cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
        new_x = 10
        new_y = 10
        cv2.putText(imgOutput, labels[index], (x + new_x, y + new_y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)


        image_path = os.path.join(img_folder, labels[index] + ".png")
        img_to_display = cv2.imread(image_path)

        cv2.imshow("Image Output", img_to_display)

    cv2.imshow("Image", imgOutput)

    if is_camera_on:
        window.after(20, update_camera)  # Schedule the next update


# Create a function to toggle hand open/close state
def toggle_hand_state():
    global is_hand_open, word
    is_hand_open = not is_hand_open
    if not is_hand_open:
        word += current_letter  # Append the current letter to the word when hand is closed


# Create a function to print the predicted word
def print_predicted_word():
    global word  # Declare word as a global variable
    if word:
        text_box.insert("end", word + " ")  # Insert the word into the text box with space
        word = ""  # Clear the word after printing


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

# Create a button to toggle hand open/close state
is_hand_open = True  # Default to
current_letter = ""
word = ""
hand_toggle_button = ttk.Button(window, text="Toggle Hand", command=toggle_hand_state)
hand_toggle_button.pack(pady=20)

# Create a button to print the predicted word
print_word_button = ttk.Button(window, text="Print the Word", command=print_predicted_word)
print_word_button.pack(pady=20)

# Create a text box to display the predicted words
text_box = tk.Text(window, height=10, width=40, font=("Helvetica", 16))
text_box.pack(pady=20)

# Run the Tkinter main loop
window.mainloop()
