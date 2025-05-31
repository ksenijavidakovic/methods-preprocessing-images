import cv2
import numpy as np
import sys
import os

DELAY_CAPTION = 1500
DELAY_BLUR = 100
MAX_KERNEL_LENGTH = 31
WINDOW_NAME = "Smoothing Demo"


def display_caption(src_img, caption):
    dst = np.zeros(src_img.shape, dtype=np.uint8)
    cv2.putText(dst, caption,
                (src_img.shape[1] // 4, src_img.shape[0] // 2),
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
    return display_dst(dst, DELAY_CAPTION)


def display_dst(image, delay):
    cv2.imshow(WINDOW_NAME, image)
    key = cv2.waitKey(delay)
    if key >= 0:
        return -1
    return 0


def save_final_image(image, blur_type, output_dir="output_blur"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    filename = f"{output_dir}/{blur_type}_final.jpg"
    cv2.imwrite(filename, image)
    print(f"Saved final {blur_type} result: {filename}")


def main():
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    filename = sys.argv[1] if len(sys.argv) >= 2 else "kid.jpg"
    src = cv2.imread(cv2.samples.findFile(filename), cv2.IMREAD_COLOR)

    if src is None:
        print("Error opening image")
        print(f"Usage: {sys.argv[0]} [image_name -- default kid.jpg]")
        return

    if display_caption(src, "Original Image") != 0:
        return

    dst = src.copy()
    if display_dst(dst, DELAY_CAPTION) != 0:
        return

    # Save original image
    save_final_image(src, "original")

    if display_caption(src, "Homogeneous Blur") != 0:
        return

    final_dst = None
    for i in range(1, MAX_KERNEL_LENGTH, 2):
        final_dst = cv2.blur(src, (i, i))
        if display_dst(final_dst, DELAY_BLUR) != 0:
            return
    save_final_image(final_dst, "homogeneous")

    if display_caption(src, "Gaussian Blur") != 0:
        return

    for i in range(1, MAX_KERNEL_LENGTH, 2):
        final_dst = cv2.GaussianBlur(src, (i, i), 0)
        if display_dst(final_dst, DELAY_BLUR) != 0:
            return
    save_final_image(final_dst, "gaussian")

    if display_caption(src, "Median Blur") != 0:
        return

    for i in range(1, MAX_KERNEL_LENGTH, 2):
        final_dst = cv2.medianBlur(src, i)
        if display_dst(final_dst, DELAY_BLUR) != 0:
            return
    save_final_image(final_dst, "median")

    if display_caption(src, "Bilateral Blur") != 0:
        return

    for i in range(1, MAX_KERNEL_LENGTH, 2):
        final_dst = cv2.bilateralFilter(src, i, i * 2, i // 2)
        if display_dst(final_dst, DELAY_BLUR) != 0:
            return
    save_final_image(final_dst, "bilateral")

    display_caption(src, "Done!")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()