import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio as io
import scipy.ndimage as nd
import cv2

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

img = mpimg.imread('bacterial_colonies.jpg')     
gray = rgb2gray(img) 

plt.imshow(gray, cmap = 'gray')
cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,10,100)