import cv2
import numpy as np

img = cv2.imread('zebra.jpg', 0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))

dilation = cv2.dilate(img, kernel, iterations=1)
erosion = cv2.erode(img, kernel, iterations=1)

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Сначала эрозия, потом дилатация
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) # Сначала дилатация, потом эрозия

# Отображение результатов
cv2.imshow('Original', img)
cv2.imshow('Dilation', dilation)
cv2.imshow('Erosion', erosion)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.waitKey(0)
