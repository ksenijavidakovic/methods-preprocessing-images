import cv2
import numpy as np
import sys

DELAY_CAPTION = 1500 
DELAY_BLUR = 100     
MAX_KERNEL_LENGTH = 31
WINDOW_NAME = "Smoothing Demo"

def display_caption(src_img, caption):
    dst = np.zeros(src_img.shape, dtype=np.uint8)
    cv2.putText(dst, caption, 
               (src_img.shape[1]//4, src_img.shape[0]//2),
               cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    return display_dst(dst, DELAY_CAPTION)

def display_dst(image, delay):
    cv2.imshow(WINDOW_NAME, image)
    key = cv2.waitKey(delay)
    if key >= 0:
        return -1
    return 0

def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    filename = sys.argv[1] if len(sys.argv) >= 2 else "pretty-dog.jpg"
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)
    
    if src is None:
        print("Error opening image")
        print(f"Usage: {sys.argv[0]} [image_name -- default pretty-dog.jpg]")
        return
    
    if display_caption(src, "Original Image") != 0:
        return
    
    dst = src.copy()
    if display_dst(dst, DELAY_CAPTION) != 0:
        return
    
    if display_caption(src, "Homogeneous Blur") != 0:
        return
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.blur(src, (i, i))
        if display_dst(dst, DELAY_BLUR) != 0:
            return
    
    if display_caption(src, "Gaussian Blur") != 0:
        return
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.GaussianBlur(src, (i, i), 0)
        if display_dst(dst, DELAY_BLUR) != 0:
            return
    
    if display_caption(src, "Median Blur") != 0:
        return
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.medianBlur(src, i)
        if display_dst(dst, DELAY_BLUR) != 0:
            return
    
    if display_caption(src, "Bilateral Blur") != 0:
        return
    
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        dst = cv2.bilateralFilter(src, i, i*2, i//2)
        if display_dst(dst, DELAY_BLUR) != 0:
            return
    
    display_caption(src, "Done!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
