import cv2
import matplotlib.pyplot as plt

img = 'D:\Capture.JPG'
img = cv2.imread(img)



silhoutte = cv2.Canny(img, 150, 200)
plt.imshow(silhoutte, cmap='gray')
plt.show()