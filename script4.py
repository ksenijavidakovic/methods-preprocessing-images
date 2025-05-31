import cv2
import numpy as np

sigma = 1.0
smooth_type = 'GAUSSIAN'  # Can be 'GAUSSIAN', 'BLUR', or 'MEDIAN'
kernel_size = int(sigma * 5) | 1

cv2.namedWindow('Laplacian Edge Detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Laplacian Edge Detection', 800, 600)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if smooth_type == 'GAUSSIAN':
        smoothed = cv2.GaussianBlur(frame, (kernel_size, kernel_size), sigma)
    elif smooth_type == 'BLUR':
        smoothed = cv2.blur(frame, (kernel_size, kernel_size))
    else:  # MEDIAN
        smoothed = cv2.medianBlur(frame, kernel_size)

    # Apply Laplacian operator (edge detection)
    laplace = cv2.Laplacian(smoothed, cv2.CV_16S, ksize=5)
    laplace_abs = cv2.convertScaleAbs(laplace)

    cv2.imshow('Original', frame)
    cv2.imshow('Laplacian Edge Detection', laplace_abs)

    params = f"Sigma: {sigma:.1f} | Kernel: {kernel_size} | Type: {smooth_type}"
    cv2.putText(laplace_abs, params, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    key = cv2.waitKey(1)
    if key == 27:
        break
    elif key == ord('+'):
        sigma += 0.1
        kernel_size = int(sigma * 5) | 1
    elif key == ord('-'):
        sigma = max(0.1, sigma - 0.1)
        kernel_size = int(sigma * 5) | 1
    elif key == ord('g'):
        smooth_type = 'GAUSSIAN'
    elif key == ord('b'):
        smooth_type = 'BLUR'
    elif key == ord('m'):
        smooth_type = 'MEDIAN'

cap.release()
cv2.destroyAllWindows()