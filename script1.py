import cv2
import numpy as np

WIDTH = 1000
HEIGHT = 700
RNG = np.random.default_rng()
WINDOW_NAME = "Drawing Random Stuff"

def random_color(rng):
    """Генерирует случайный цвет в формате BGR"""
    icolor = rng.integers(0, 256, 3)
    return (int(icolor[0]), int(icolor[1]), int(icolor[2]))

def drawing_random_lines(img, rng):
    for i in range(0, 100):
        pt1 = (rng.integers(0, WIDTH), rng.integers(0, HEIGHT))
        pt2 = (rng.integers(0, WIDTH), rng.integers(0, HEIGHT))
        color = random_color(rng)
        thickness = rng.integers(1, 10)
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def drawing_random_rectangles(img, rng):
    for i in range(0, 100):
        pt1 = (rng.integers(0, WIDTH), rng.integers(0, HEIGHT))
        pt2 = (rng.integers(0, WIDTH), rng.integers(0, HEIGHT))
        color = random_color(rng)
        thickness = rng.integers(-1, 10)  
        cv2.rectangle(img, pt1, pt2, color, thickness, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def drawing_random_ellipses(img, rng):
    for i in range(0, 100):
        center = (rng.integers(0, WIDTH), rng.integers(0, HEIGHT))
        axes = (rng.integers(0, 200), rng.integers(0, 200))
        angle = rng.integers(0, 360)
        start_angle = rng.integers(0, 360)
        end_angle = rng.integers(0, 360)
        color = random_color(rng)
        thickness = rng.integers(-1, 10)
        cv2.ellipse(img, center, axes, angle, start_angle, end_angle, color, thickness, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def drawing_random_polylines(img, rng):
    for i in range(0, 100):
        count = rng.integers(3, 6) 
        pts = rng.integers(0, WIDTH, (count, 2))
        color = random_color(rng)
        is_closed = rng.choice([True, False])
        thickness = rng.integers(1, 10)
        cv2.polylines(img, [pts], is_closed, color, thickness, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def drawing_random_filled_polygons(img, rng):
    for i in range(0, 100):
        count = rng.integers(3, 6) 
        pts = rng.integers(0, WIDTH, (count, 2))
        color = random_color(rng)
        cv2.fillPoly(img, [pts], color, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def drawing_random_circles(img, rng):
    for i in range(0, 100):
        center = (rng.integers(0, WIDTH), rng.integers(0, HEIGHT))
        radius = rng.integers(0, 200)
        color = random_color(rng)
        thickness = rng.integers(-1, 10)
        cv2.circle(img, center, radius, color, thickness, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def displaying_random_text(img, rng):
    text = "Testing text rendering"
    org = (50, HEIGHT // 2)
    font_face = rng.integers(0, 8)  
    font_scale = rng.uniform(0.5, 3.0)
    color = random_color(rng)
    thickness = rng.integers(1, 10)
    line_type = cv2.LINE_AA
    
    cv2.putText(img, text, org, font_face, font_scale, color, thickness, line_type)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def displaying_big_end(img, rng):
    fonts = [
        cv2.FONT_HERSHEY_SIMPLEX,
        cv2.FONT_HERSHEY_PLAIN,
        cv2.FONT_HERSHEY_DUPLEX,
        cv2.FONT_HERSHEY_COMPLEX,
        cv2.FONT_HERSHEY_TRIPLEX,
        cv2.FONT_HERSHEY_COMPLEX_SMALL,
        cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
        cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    ]
    
    text = "OpenCV forever!"
    for i, font in enumerate(fonts):
        org = (50, 50 + i * 50)
        cv2.putText(img, text, org, font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow(WINDOW_NAME, img)
    return cv2.waitKey(0)

def main():
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    
    c = drawing_random_lines(image, RNG)
    c = drawing_random_rectangles(image, RNG)
    c = drawing_random_ellipses(image, RNG)
    c = drawing_random_polylines(image, RNG)
    c = drawing_random_filled_polygons(image, RNG)
    c = drawing_random_circles(image, RNG)
    c = displaying_random_text(image, RNG)
    c = displaying_big_end(image, RNG)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
