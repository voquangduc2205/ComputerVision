import cv2

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import math
from skimage.exposure import is_low_contrast

# reads an input image

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

def sobel_edge_detection(img, blur_ksize=7, sobel_ksize=1, skipping_threshold=10):
    """
    image_path: link to image
    blur_ksize: kernel size parameter for Gaussian Blurry
    sobel_ksize: size of the extended Sobel kernel; it must be 1, 3, 5, or 7.
    skipping_threshold: ignore weakly edge
    """
    # read image
    
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


def canny_edge_detection(img, blur_ksize=3, threshold1=100, threshold2=200):
    """
    image_path: link to image
    blur_ksize: Gaussian kernel size
    threshold1: min threshold 
    threshold2: max threshold
    """
    
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


def median_filter(img):
  median_img = cv2.medianBlur(img, 5)

  return median_img


def gaussian_filter(img):
   gaussian_img = cv2.GaussianBlur(img, (5,5), cv2.BORDER_DEFAULT)

   return gaussian_img


def histogram_equalization(img):

  height, width = img.shape[:2]
# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
  clahe = cv2.createCLAHE(clipLimit = 5, tileGridSize=(math.floor(height/50), math.floor(width/50)))
  final_img = clahe.apply(img) + 30

  return final_img


def gamma_correction(img):
  for gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
      
    # Apply gamma correction.
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
  
    # Save edited images.
    cv2.imwrite('gamma_transformed'+str(gamma)+'.png', gamma_corrected)


def fourier_transform(img):

  kernel = np.array([[0.125, 0.125, 0.125],
                    [0.125, -1, 0.125],
                    [0.125, 0.125, 0.125]
                    ])

  f = np.fft.fft2(img)
  fshift = np.fft.fftshift(f)
  fft_img_shift = 20*np.log(np.abs(fshift))

  
  filter_img = cv2.filter2D(fft_img_shift, -1, kernel)

  frame = np.array(filter_img)
  df = pd.DataFrame(frame)
  df.to_csv('filter.csv')

  new_frame = []
  for x in frame:
    temp = []
    for y in x:
      if y < 40:
        temp.append(y)  
      else:
        temp.append(1)
    new_frame.append(temp)
  
  
  # frame = cv2.idft(new_frame)

  # cv2.imwrite("fft.png", frame)

  # extract real and phases
  real = fft_img_shift.real
  phases = fft_img_shift.imag

  # modify real part, put your modification here
  real_mod = real/3

  # create an empty complex array with the shape of the input image
  fft_img_shift_mod = np.empty(real.shape, dtype=complex)

  # insert real and phases to the new file
  fft_img_shift_mod.real = real_mod
  fft_img_shift_mod.imag = phases

  # reverse shift
  fft_img_mod = np.fft.ifftshift(fft_img_shift_mod)

  # reverse the 2D fourier transform
  img_mod = np.fft.ifft2(fft_img_mod)

  # using np.abs gives the scalar value of the complex number
  # with img_mod.real gives only real part. Not sure which is proper
  img_mod = np.abs(img_mod)

  # show differences
  plt.subplot(121)
  plt.imshow(img, cmap='gray')
  plt.subplot(122)
  plt.imshow(img_mod, cmap='gray')
  plt.show()


def adaptive_threshold(img):
   
   thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 55, -12)
   return thresh
   
def closing_morphological(img):
   
   kernel = np.ones((3, 3), np.uint8)

   closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)

   return closing


def erosion(img):
   
   kernel = np.ones((5, 5), np.uint8)

   invert = cv2.bitwise_not(img)

   erosion = cv2.erode(invert, kernel=kernel, iterations=1)

   return erosion


def find_contours(img):
   
   cv2.imshow('Input', img)

   thresh = cv2.Canny(img, 127, 255)

   contours, hierachy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

   print("Number of Contours found = " + str(len(contours)))

   cv2.drawContours(img, contours, -1, (88, 255, 51), 2)
   cv2.imshow('Contours', img)

   cv2.waitKey(0)

img = cv2.imread('./image/img1.png',0)

# Pipeline to process image
img = median_filter(img)
cv2.imwrite("After Median Filter.png", img=img)

img = gaussian_filter(img)
cv2.imwrite("After Gaussian Filter.png", img=img)

if is_low_contrast(img, 0.35):
  img = histogram_equalization(img)
  cv2.imwrite("After Equal Histogram.png", img=img)

img = adaptive_threshold(img)
cv2.imwrite("After adaptive threshold.png", img=img)


img = closing_morphological(img)
cv2.imwrite("After closing morphological.png", img=img)

img = erosion(img)
cv2.imwrite("After erosion.png", img=img)

find_contours(img)

print("Complete processing!")
