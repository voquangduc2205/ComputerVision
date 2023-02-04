import cv2

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd

# reads an input image
img = cv2.imread('./image/img3.png',0)

def plot_1_row(img):
  frame = np.array(img)
  print(frame.shape)
  df = pd.DataFrame(frame)
  df.to_csv("image.csv")
  plt.plot(frame[100])
  plt.show()

def plot_histogram(img):
  
  # find frequency of pixels in range 0-255
  histr = cv2.calcHist([img],[0],None,[256],[0,256])

  # show the plotting graph of an image
  plt.plot(histr)
  plt.show()

def median_filter(img):
  kernel = np.ones((3,3),np.float32)/9
  new_img = cv2.filter2D(img, -1, kernel=kernel)

  print(new_img)
  cv2.imshow("After median filter",new_img)

  cv2.waitKey(0)

def sobel_edge_detection(image_path, blur_ksize=7, sobel_ksize=1, skipping_threshold=10):
    """
    image_path: link to image
    blur_ksize: kernel size parameter for Gaussian Blurry
    sobel_ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    skipping_threshold: ignore weakly edge
    """
    # read image
    img = cv2.imread(image_path)
    
    # convert BGR to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    
    # sobel algorthm use cv2.CV_64F
    sobelx64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    abs_sobel64f = np.absolute(sobelx64f)
    img_sobelx = np.uint8(abs_sobel64f)

    sobely64f = cv2.Sobel(img_gaussian, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    abs_sobel64f = np.absolute(sobely64f)
    img_sobely = np.uint8(abs_sobel64f)
    
    # calculate magnitude
    img_sobel = (img_sobelx + img_sobely)/2
    
    # ignore weakly pixel
    for i in range(img_sobel.shape[0]):
        for j in range(img_sobel.shape[1]):
            if img_sobel[i][j] < skipping_threshold:
                img_sobel[i][j] = 0
            else:
                img_sobel[i][j] = 255
    return img_sobel


# Edit threshold1 and threshold2
def canny_edge_detection(image_path, blur_ksize=3, threshold1=100, threshold2=200):
    """
    image_path: link to image
    blur_ksize: Gaussian kernel size
    threshold1: min threshold 
    threshold2: max threshold
    """
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray,(blur_ksize,blur_ksize),0)

    img_canny = cv2.Canny(img_gaussian,threshold1,threshold2)

    return img_canny


def hough_transform(image):

  lines_list =[]
  lines = cv2.HoughLinesP(
              image, # Input edge image
              1, # Distance resolution in pixels
              np.pi/180, # Angle resolution in radians
  # Edit from here
              threshold=100, # Min number of votes for valid line
              minLineLength=0, # Min allowed length of line
              maxLineGap=1000 # Max allowed gap between line for joining them
              )
    
  # Iterate over points
  for points in lines:
        # Extracted points nested in the list
      x1,y1,x2,y2=points[0]
      # Draw the lines joing the points
      # On the original image
      cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
      # Maintain a simples lookup list for points
      lines_list.append([(x1,y1),(x2,y2)])
        
  # Save the result image
  cv2.imwrite('detectedLines.png',image)

  return len(lines)

# cv2.imshow("edge detect", sobel_edge_detection("./image/img1.png"))

img_canny = canny_edge_detection("./image/img1.png", 3)
cv2.imwrite('canny_detect1.png', img_canny)

print(hough_transform(img_canny))

cv2.waitKey(0)