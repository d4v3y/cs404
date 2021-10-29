####################
# Davey Anguiano   #
# CS 404           #
# HW 5             #
####################

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Reading images
og_img = cv.imread('./images/og_pic.JPEG')
img2 = cv.imread('./images/pic2.JPEG')
img3 = cv.imread('./images/pic3.JPEG')
img4 = cv.imread('./images/pic4.JPEG')

## Convert to grayscale
og_img = cv.cvtColor(og_img, cv.COLOR_BGR2GRAY)
img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
img4 = cv.cvtColor(img4, cv.COLOR_BGR2GRAY)

# SIFT
sift = cv.SIFT_create()
keypoints_sift_1, descriptors_sift_1 = sift.detectAndCompute(og_img, None)
keypoints_sift_2, descriptors_sift_2 = sift.detectAndCompute(img2, None)

# ORB
orb = cv.ORB_create()
keypoints_orb_1, descriptors_orb_1 = orb.detectAndCompute(og_img, None)
keypoints_orb_2, descriptors_orb_2 = orb.detectAndCompute(img3, None)

# KAZE
kaze = cv.KAZE_create();
keypoints_kaze_1, descriptors_kaze_1 = kaze.detectAndCompute(og_img, None)
keypoints_kaze_2, descriptors_kaze_2 = kaze.detectAndCompute(img4, None)

# Feature matching
bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)

## Sift Matching
match_sift = bf.match(descriptors_sift_1, descriptors_sift_2)
match_sift = sorted(match_sift, key=lambda x: x.distance)
img_sift = cv.drawMatches(og_img, keypoints_sift_1, img2, keypoints_sift_2, match_sift[:50], img2, flags=2)

## Orb Matching
match_orb = bf.match(descriptors_orb_1, descriptors_orb_2)
match_orb = sorted(match_orb, key=lambda x: x.distance)
img_orb = cv.drawMatches(og_img, keypoints_orb_1, img3, keypoints_orb_2, match_orb[:50], img3, flags=2)

## Kaze Matching
match_kaze = bf.match(descriptors_kaze_1, descriptors_kaze_2)
match_kaze = sorted(match_kaze, key=lambda x: x.distance)
img_kaze = cv.drawMatches(og_img, keypoints_kaze_1, img4, keypoints_kaze_2, match_kaze[:50], img4, flags=2)

# Output
plt.imshow(img_sift)
plt.title("Sift Detection")
plt.show()

plt.imshow(img_orb)
plt.title("Orb Detection")
plt.show()

plt.imshow(img_kaze)
plt.title("Kaze Detection")
plt.show()
