import numpy as np
import cv2 as cv

im = cv.imread('After erosion.png')
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 50, 100, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

print("Number of Contours found = " + str(len(contours)))

cv.drawContours(imgray, contours, -1, (88, 255, 51), 2)
cv.imshow('Contour', imgray)

cv.waitKey(0)