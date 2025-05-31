import cv2
import numpy as np


def main():
    img = cv2.imread('giraffe.jpg', cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Could not load image 'giraffe.jpg'")
        return -1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    dilation = cv2.dilate(img, kernel, iterations=1)
    erosion = cv2.erode(img, kernel, iterations=1)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # Erosion followed by dilation
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # Dilation followed by erosion

    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Dilation', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Erosion', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Opening', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Closing', cv2.WINDOW_NORMAL)

    for window in ['Original', 'Dilation', 'Erosion', 'Opening', 'Closing']:
        cv2.resizeWindow(window, 600, 400)

    cv2.imshow('Original', img)
    cv2.imshow('Dilation', dilation)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)

    print("Window controls:")
    print("- Drag window edges to resize")
    print("- Press any key to close windows")

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()