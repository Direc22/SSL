import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # Set maxHands to 2 to capture both hands
classifier = Classifier("model2/keras_model.h5", "model2/labels.txt")

offset = 20
imgSize = 300
counter = 0

img_folder = "IMG2"

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']

labels_v2 = ["ආයුබෝවන්", "ඔයා හොදින්ද", "කරුණාකරල", "කොහෙද යන්නේ", "බඩගිනි", "බොහොම ස්තූතියි", "මට අසනීපයි", "මම හොදින්", "සමාවන්න", "සුබ උදෑසනක්", "සුබ රා ත්‍රියක්"]

# Initialize imgCrop with an empty image
imgCrop = np.zeros((imgSize, imgSize, 3), np.uint8)

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    for hand in hands:
        x, y, w, h = hand['bbox']

        # Update imgCrop with the hand region
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]

        # Add a check to ensure imgCrop is not empty and has a valid size
        if w > 0 and h > 0 and imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

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
                print(labels_v2[index])

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                print(labels_v2[index])

            cv2.putText(imgOutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 255), 1)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

            image_path = os.path.join(img_folder, labels[index] + ".png")
            img_to_display = cv2.imread(image_path)

            # cv2.imshow("ImageCrop", imgCrop)
            # cv2.imshow("ImageWhite", imgWhite)
            cv2.imshow("Image Output", img_to_display)

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(20) & 0xFF == 27:  # Press 'Esc' key to exit the loop
        break

cap.release()
cv2.destroyAllWindows()
