import cv2
import numpy as np
import sys


def main():
    default_file = "sudoku.webp"
    filename = default_file if len(sys.argv) < 2 else sys.argv[1]

    # Load image in grayscale
    src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Error opening image")
        print(f"Program Arguments: [image_name -- default {default_file}]")
        return -1

    # Edge detection using Canny
    dst = cv2.Canny(src, 50, 200, apertureSize=3)

    # Create color versions for line drawing
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = cdst.copy()

    # 1. Standard Hough Line Transform
    lines = cv2.HoughLines(dst, 1, np.pi / 180, 150)

    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    # 2. Probabilistic Hough Line Transform
    linesP = cv2.HoughLinesP(dst, 1, np.pi / 180, 50, None, 50, 10)

    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    # Create resizable windows
    cv2.namedWindow("Source", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Standard Hough", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Probabilistic Hough", cv2.WINDOW_NORMAL)

    # Set initial window sizes (optional)
    cv2.resizeWindow("Source", 600, 600)
    cv2.resizeWindow("Standard Hough", 600, 600)
    cv2.resizeWindow("Probabilistic Hough", 600, 600)

    # Display results
    cv2.imshow("Source", src)
    cv2.imshow("Standard Hough", cdst)
    cv2.imshow("Probabilistic Hough", cdstP)

    print("Window controls:")
    print("- Drag window edges to resize")
    print("- Press any key to close windows")

    cv2.waitKey()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    main()