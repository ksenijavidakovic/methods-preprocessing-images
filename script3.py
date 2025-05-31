import cv2
import numpy as np
import argparse


def main():
    parser = argparse.ArgumentParser(description='Sobel Edge Detection Demo')
    parser.add_argument('--input', default='flowers.jpg',
                        help='Path to input image (default: flowers.jpg)')
    parser.add_argument('--ksize', type=int, default=1,
                        help='Kernel size (default: 1). Press "K" to increase')
    parser.add_argument('--scale', type=int, default=1,
                        help='Scale factor (default: 1). Press "S" to increase')
    parser.add_argument('--delta', type=int, default=0,
                        help='Delta value (default: 0). Press "D" to increase')
    parser.add_argument('--resize', type=float, default=0.7,
                        help='Resize factor for display window (default: 0.7)')
    args = parser.parse_args()

    window_name = "Sobel Demo - Simple Edge Detector"
    ksize = args.ksize
    scale = args.scale
    delta = args.delta
    ddepth = cv2.CV_16S

    image = cv2.imread(cv2.samples.findFile(args.input), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Error opening image: {args.input}")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        src = cv2.GaussianBlur(image, (3, 3), 0)
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        grad_x = cv2.Sobel(src_gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta)
        grad_y = cv2.Sobel(src_gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # Resize the output image for display
        height, width = grad.shape[:2]
        resized_grad = cv2.resize(grad, (int(width * args.resize), int(height * args.resize)))

        cv2.imshow(window_name, resized_grad)
        key = cv2.waitKey(0)

        if key == 27:
            break
        elif key == ord('k') or key == ord('K'):
            ksize = ksize + 2 if ksize < 30 else -1
        elif key == ord('s') or key == ord('S'):
            scale += 1
        elif key == ord('d') or key == ord('D'):
            delta += 1
        elif key == ord('r') or key == ord('R'):
            scale = 1
            ksize = -1
            delta = 0
        elif key == ord('+') or key == ord('='):
            args.resize = min(args.resize + 0.1, 2.0)
        elif key == ord('-') or key == ord('_'):
            args.resize = max(args.resize - 0.1, 0.1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()