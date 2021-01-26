import timeit
Elapsed = timeit.timeit()
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as nd
import cv2

img = cv2.imread('bacterial_colonies.jpg',0)

#plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')

cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1.5,100,param1=300,param2=100,minRadius=100,maxRadius=200)

circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.imshow(cimg)
print(Elapsed)