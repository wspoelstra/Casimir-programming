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

CircleShape = np.shape(circles)
NumPlates = CircleShape[1]

CirclesMatrix = circles.reshape(NumPlates,3)
np.shape(CirclesMatrix)
Centers = np.zeros([NumPlates,2])
Radii = np.zeros(NumPlates)

for c in range(NumPlates):
    Centers[c,0] = CirclesMatrix[c,0]
    Centers[c,1] = CirclesMatrix[c,1]
    Radii[c] = CirclesMatrix[c,2]

Centers = Centers.astype(np.int)
Radii = Radii.astype(np.int)

imagesize = np.shape(img)
maskimg = np.zeros([imagesize[0],imagesize[1]])

for i in range(NumPlates):
    Plate = np.zeros([2*Radii[i], 2*Radii[i]])
    for x in range(imagesize[1]):
        for y in range(imagesize[0]):
            if (x-Centers[i,0])**2+(y-Centers[i,1])**2<Radii[i]**2:
                maskimg[y,x] = 1
                Plate[y-Centers[i,1]-Radii[i],x-Centers[i,0]+Radii[i]] = img[y,x]
    
    plt.imsave('Plate'+str(i)+'.png',Plate)

    del Plate

plt.imshow(maskimg*img, cmap = 'gray')

print(Elapsed)