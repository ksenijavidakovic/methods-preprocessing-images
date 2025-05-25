import cv2
import numpy as np

sigma = 1.0
smooth_type = 'BLUR'
kernel_size = int(sigma * 5) | 1

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    if smooth_type == 'GAUSSIAN':
        smoothed = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    elif smooth_type == 'BLUR':
        smoothed = cv2.blur(frame, (kernel_size, kernel_size))
    else:
        smoothed = cv2.medianBlur(frame, kernel_size)
    
    laplace = cv2.Laplacian(smoothed, cv2.CV_16S, ksize=5)
    laplace_abs = cv2.convertScaleAbs(laplace)
    
    cv2.imshow('Original', frame)
    cv2.imshow('Laplacian Edge Detection', laplace_abs)
    
    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('+'):
        sigma += 0.1
        kernel_size = int(sigma * 5) | 1
    elif key == ord('-'):
        sigma = max(0.1, sigma - 0.1)
        kernel_size = int(sigma * 5) | 1

cap.release()
cv2.destroyAllWindows()
