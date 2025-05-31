import cv2
import numpy as np

image = cv2.imread("music.jpg")
if image is None:
    print("Error: Could not load image")
    exit()

window_name1 = "Edge map : Canny default (Sobel gradient)"
window_name2 = "Edge map : Canny with custom gradient (Scharr)"
cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
cv2.namedWindow(window_name2, cv2.WINDOW_NORMAL)

edgeThresh = 50
edgeThreshScharr = 50

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def update_images(_):
    blurImage = cv2.GaussianBlur(gray, (3, 3), 0)

    edge1 = cv2.Canny(blurImage, edgeThresh, edgeThresh * 3, apertureSize=3)
    cedge = cv2.merge([edge1, edge1, edge1])  # Convert to 3-channel for display

    dx = cv2.Scharr(blurImage, cv2.CV_16S, 1, 0)
    dy = cv2.Scharr(blurImage, cv2.CV_16S, 0, 1)
    edge2 = cv2.Canny(dx, dy, edgeThreshScharr, edgeThreshScharr * 3)
    cedge2 = cv2.merge([edge2, edge2, edge2])  # Convert to 3-channel

    cv2.imshow(window_name1, cedge)
    cv2.imshow(window_name2, cedge2)


cv2.createTrackbar("Canny threshold default", window_name1, edgeThresh, 100, update_images)
cv2.createTrackbar("Canny threshold Scharr", window_name2, edgeThreshScharr, 400, update_images)

update_images(0)

print("Controls:")
print("- Adjust thresholds using trackbars")
print("- Press any key to exit")

cv2.waitKey(0)
cv2.destroyAllWindows()