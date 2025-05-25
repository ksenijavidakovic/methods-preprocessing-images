import cv2
import numpy as np

def main():
    default_file = "sudoku-real.webp"
    filename = default_file if len(sys.argv) < 2 else sys.argv[1]
    
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        print("Error opening image")
        print(f"Program Arguments: [image_name -- default {default_file}]")
        return -1

    dst = cv2.Canny(src, 50, 200, apertureSize=3)
    
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    cdstP = cdst.copy()

    lines = cv2.HoughLines(dst, 1, np.pi/180, 150)
    
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
            cv2.line(cdst, pt1, pt2, (0, 0, 255), 3, cv2.LINE_AA)

    linesP = cv2.HoughLinesP(dst, 1, np.pi/180, 50, None, 50, 10)
    
    if linesP is not None:
        for line in linesP:
            x1, y1, x2, y2 = line[0]
            cv2.line(cdstP, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)

    cv2.imshow("Source", src)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)

    cv2.waitKey()
    return 0

if __name__ == "__main__":
    import sys
    main()
