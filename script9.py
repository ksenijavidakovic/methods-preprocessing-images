"""
@file morph_lines_detection.py
@brief Use morphology transformations for extracting horizontal and vertical lines sample code
"""
import numpy as np
import sys
import cv2 as cv

def show_wait_destroy(winname, img):
    cv.imshow(winname, img)
    cv.moveWindow(winname, 500, 0)
    cv.waitKey(0)
    cv.destroyWindow(winname)

def main(argv):

    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1


    src = cv.imread(argv[0], cv.IMREAD_COLOR)


    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1


    cv.imshow("src", src)


    # [gray]
    if len(src.shape) != 2:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    else:
        gray = src

    show_wait_destroy("gray", gray)

    # [bin]
    gray = cv.bitwise_not(gray)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 15, -2)
    # Show binary image
    show_wait_destroy("binary", bw)

    # [init]
    horizontal = np.copy(bw)
    vertical = np.copy(bw)

    # [horiz]
    cols = horizontal.shape[1]
    horizontal_size = cols // 30

    horizontalStructure = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_size, 1))

    horizontal = cv.erode(horizontal, horizontalStructure)
    horizontal = cv.dilate(horizontal, horizontalStructure)

    show_wait_destroy("horizontal", horizontal)

    # [vert]
    rows = vertical.shape[0]
    verticalsize = rows // 30

    verticalStructure = cv.getStructuringElement(cv.MORPH_RECT, (1, verticalsize))

    vertical = cv.erode(vertical, verticalStructure)
    vertical = cv.dilate(vertical, verticalStructure)

    show_wait_destroy("vertical", vertical)

    # [smooth]
    vertical = cv.bitwise_not(vertical)
    show_wait_destroy("vertical_bit", vertical)

    '''
    Extract edges and smooth image according to the logic
    1. extract edges
    2. dilate(edges)
    3. src.copyTo(smooth)
    4. blur smooth img
    5. smooth.copyTo(src, edges)
    '''

    edges = cv.adaptiveThreshold(vertical, 255, cv.ADAPTIVE_THRESH_MEAN_C, \
                                cv.THRESH_BINARY, 3, -2)
    show_wait_destroy("edges", edges)

    kernel = np.ones((2, 2), np.uint8)
    edges = cv.dilate(edges, kernel)
    show_wait_destroy("dilate", edges)

    smooth = np.copy(vertical)

    smooth = cv.blur(smooth, (2, 2))

    (rows, cols) = np.where(edges != 0)
    vertical[rows, cols] = smooth[rows, cols]

    show_wait_destroy("smooth - final", vertical)

    return 0

if __name__ == "__main__":
    main(sys.argv[1:])