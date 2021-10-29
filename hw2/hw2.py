#################### 
# Davey Anguiano   #
# CS 404           #
# HW 2             #
####################

import cv2 as cv
import numpy as np

### Part I: ###
# Original Image
pic = cv.imread('tiger.jpg')
cv.imshow("Original Image", pic)
cv.waitKey(0)

# Image size
scale_percent = 40  # percent of original size
width = int(pic.shape[1] * scale_percent / 100)
height = int(pic.shape[0] * scale_percent / 100)
dim = (width, height)

### Transformations ###
# 1. RESIZE IMAGE
resized = cv.resize(pic, dim, interpolation=cv.INTER_AREA)
cv.imshow("Resized Image", resized)
cv.waitKey(0)

# 2. PERSPECTIVE WARP
# Points
rows, cols = resized.shape[:2]
src_points = np.float32([[0, 260], [640, 260], [0, 400], [640, 400]])
dst_points = np.float32([[0, 0], [400, 0], [0, 640], [400, 640]])
# Transformation
warpMatrix = cv.getPerspectiveTransform(src_points, dst_points)
warpImg = cv.warpPerspective(resized, warpMatrix, dim)
cv.imshow("Warp Perspective Image", warpImg)
cv.waitKey(0)

# 3. REDUCE IMAGE COLOR (ZERO OUT GREEN CHANNEL)
noGreen = resized.copy()
noGreen[:,:,1] = 0
cv.imshow("Image with No Green", noGreen)
cv.waitKey(0)

# -------------------------------------------------------- #
### Part II: ###
def contrastAndBrightness(contrast, brightness):

   image = resized.copy()
   newImg = np.zeros(image.shape, image.dtype)
   
   # Adjust brightness and contrast algorithm
   for y in range(image.shape[0]):
    for x in range(image.shape[1]):
        for c in range(image.shape[2]):
           newImg[y,x,c] = np.clip(contrast*image[y,x,c] + brightness, 0, 255)

   cv.imshow("Function to Control Contrast and Brightness", newImg)
   cv.waitKey(0)

contrastAndBrightness(2.2, 2.2)
contrastAndBrightness(2.2, 22)
contrastAndBrightness(22, 22)
contrastAndBrightness(22, 2.2)