import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('C:\Users\Peter\Desktop\square.jpg',0)
edges = cv2.Canny(img,100,200)
ret, data = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('pic', data)
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()