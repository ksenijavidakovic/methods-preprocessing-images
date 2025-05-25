import cv2  

image = cv2.imread("pretty-dog.jpg")  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
blurred = cv2.GaussianBlur(gray, (3, 3), 0)  

edges_sobel = cv2.Canny(blurred, 50, 150)  
edges_scharr = cv2.Canny(blurred, 50, 150, L2gradient=True)  

cv2.imshow("Canny (Sobel)", edges_sobel)  
cv2.imshow("Canny (Scharr)", edges_scharr)  
cv2.waitKey(0)  
