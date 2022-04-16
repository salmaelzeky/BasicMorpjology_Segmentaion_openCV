import numpy as np
import os
import cv2
from google.colab.patches import cv2_imshow


def filter_img(noiseType, image):
   if noiseType == "gauss":   
      row,col,ch= image.shape
      mean = 0
      var = 0.005
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy  

      
   elif noiseType == "S&P":
      row,col,ch = image.shape
      s_vs_p = 0.05
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out 
   else: 
      print("enter either a S&P filter or a Gaussian filter")

from itertools import product
from google.colab.patches import cv2_imshow
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
from skimage import color


if __name__ == "__main__":
    # read original image
    img = imread(r"/content/DSCN0482-001.JPG")
    # turn image in gray scale value
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print (gray_image.shape) 
    cv2.imwrite('gray.jpg', gray_image)

image = cv2.imread('gray.jpg')
print (image.shape)

import matplotlib.pyplot as plt
    import math
    from PIL import Image, ImageFilter
    from google.colab.patches import cv2_imshow
    import time
    
    #Gaussian Filter
    gaussinFilterStartTime = time.process_time()
    gaussianFilter = filter_img("gauss", image)
    gaussianFilterTime = time.process_time() - gaussinFilterStartTime
    print("Needed time to apply gaussian filter:")
    print(gaussianFilterTime)

    cv2.imwrite('gaussPic.jpg', gaussianFilter)

    #Box Filter
    afterGaussFilter = Image.open('gaussPic.jpg')
    applying3x3BoxFilterOnGaussinFilterStartTime = time.process_time()
    boxFilter3x3 = afterGaussFilter.filter(ImageFilter.BoxBlur(3))
    applying3x3BoxFilterOnGaussinFilterTime = time.process_time() - applying3x3BoxFilterOnGaussinFilterStartTime
    print("Needed time to apply 3x3 box filter on gaussian filter:")
    print(applying3x3BoxFilterOnGaussinFilterTime)

    applying7x7BoxFilterOnGaussinFilterStartTime = time.process_time()
    boxFilter7x7 = afterGaussFilter.filter(ImageFilter.BoxBlur(7))
    applying7x7BoxFilterOnGaussinFilterTime = time.process_time() - applying7x7BoxFilterOnGaussinFilterStartTime
    print("Needed time to apply 7x7 box filter on gaussian filter:")
    print(applying7x7BoxFilterOnGaussinFilterTime)

    #Median Filter
    applyingMedianFilterOnGaussinFilterStartTime = time.process_time()

    medianFilter = cv2.medianBlur(cv2.imread("/content/gaussPic.jpg"), 5)

    
    applyingMedianFilterOnGaussinFilterTime = time.process_time() - applyingMedianFilterOnGaussinFilterStartTime
    print("Needed time to apply median filter on gaussian filter:")
    print(applyingMedianFilterOnGaussinFilterTime)



    #S&P Filter
    saltAndPeperFilterStartTime = time.process_time()
    saltAndPeperFilter = filter_img("S&P", image)
    saltAndPeperFilterTime = time.process_time() - saltAndPeperFilterStartTime
    print("Needed time to apply S&P filter:")
    print(saltAndPeperFilterTime)
    cv2.imwrite('S&P.jpg', saltAndPeperFilter)

    #Box Filter
    afterSaltAndPeperFilter = Image.open('S&P.jpg')
    applying3x3BoxFilterOnSaltAndPeperFilterStartTime = time.process_time()
    boxFilter3x3OnSaltAndPeper = afterSaltAndPeperFilter.filter(ImageFilter.BoxBlur(3))
    applying3x3BoxFilterOnSaltAndPeperFilterTime = time.process_time() - applying3x3BoxFilterOnSaltAndPeperFilterStartTime
    print("Needed time to apply 3x3 box filter on S&P filter:")
    print(applying3x3BoxFilterOnSaltAndPeperFilterTime)


    applying7x7BoxFilterOnSaltAndPeperFilterStartTime = time.process_time()
    boxFilter7x7OnSaltAndPeper = afterSaltAndPeperFilter.filter(ImageFilter.BoxBlur(7))
    applying7x7BoxFilterOnSaltAndPeperFilterTime = time.process_time() - applying7x7BoxFilterOnSaltAndPeperFilterStartTime
    print("Needed time to apply 3x3 box filter on S&P filter:")
    print(applying7x7BoxFilterOnSaltAndPeperFilterTime)

    #Median Filter
    applyingMedianFilterOnSaltAndPeperFilterStartTime = time.process_time()
    medianFilterOnSaltAndPeper = cv2.medianBlur(cv2.imread("/content/S&P.jpg"), 5)
    applyingMedianFilterOnSaltAndPeperFilterTime = time.process_time() - applyingMedianFilterOnSaltAndPeperFilterStartTime
    print("Needed time to apply median filter on S&P filter:")
    print(applyingMedianFilterOnSaltAndPeperFilterTime)





    def mse(img1, img2):
        subtractValue = np.subtract(img1, img2) 
        squared_diff = np.square(subtractValue)
        mseValue = squared_diff.mean()
        return mseValue
 

    def psnr(img1, img2):
        mseValueHere = mse(img1, img2)
        if mseValueHere == 0:
            return 100
        PIXEL_MAX = 255.0
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mseValueHere))


    #MSE AND PSNR FOR GAUSS FILTER
    mseForGaussFilterAndGaussFilterWithBoxFilter3x3= mse(cv2.imread("/content/gaussPic.jpg"), np.array(boxFilter3x3))
    mseForGaussFilterAndGaussFilterWithBoxFilter7x7= mse(cv2.imread("/content/gaussPic.jpg"), np.array(boxFilter7x7))
    mseForGaussFilterAndGaussFilterWithMedianFilter= mse(cv2.imread("/content/gaussPic.jpg"), medianFilter)
    psnrForGaussFilterAndGaussFilterWithBoxFilter3x3= psnr(cv2.imread("/content/gaussPic.jpg"), np.array(boxFilter3x3))
    psnrForGaussFilterAndGaussFilterWithBoxFilter7x7= psnr(cv2.imread("/content/gaussPic.jpg"), np.array(boxFilter7x7))
    psnrForGaussFilterAndGaussFilterWithMedianFilter= psnr(cv2.imread("/content/gaussPic.jpg"), medianFilter)


    #MSE AND PSNR FOR S&P FILTER
    mseForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter3x3= mse(cv2.imread("/content/S&P.jpg"), np.array(boxFilter3x3OnSaltAndPeper))
    mseForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter7x7= mse(cv2.imread("/content/S&P.jpg"), np.array(boxFilter7x7OnSaltAndPeper))
    mseForSaltAndPeperFilterAndSaltAndPeperFilterWithMedianFilter= mse(cv2.imread("/content/S&P.jpg"), medianFilterOnSaltAndPeper)
    psnrForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter3x3= psnr(cv2.imread("/content/S&P.jpg"), np.array(boxFilter3x3OnSaltAndPeper))
    psnrForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter7x7= psnr(cv2.imread("/content/S&P.jpg"), np.array(boxFilter7x7OnSaltAndPeper))
    psnrForSaltAndPeperFilterAndSaltAndPeperFilterWithMedianFilter= psnr(cv2.imread("/content/S&P.jpg"), medianFilterOnSaltAndPeper)

"""**Result images after applying 3x3 box filter, 7x7 box filter, and median filter on gaussian filter:**
"""

# show result images
cv2_imshow(img)
print('Original image \n')
cv2_imshow(image)
print('Converting image to grayscale \n')
cv2_imshow(gaussianFilter)
print('After applying gaussian filter \n')
cv2_imshow(np.array(boxFilter3x3))
print('After applying a 3x3 box filter on gaussian filter')
print('MSE of 3x3 box filter on gaussian filter:')
print(mseForGaussFilterAndGaussFilterWithBoxFilter3x3)
print('PSNR 3x3 box filter on gaussian filter:')
print(psnrForGaussFilterAndGaussFilterWithBoxFilter3x3)
print("\n")
cv2_imshow(np.array(boxFilter7x7))
print('After applying a 7x7 box filter on gaussian filter')
print('MSE of 7x7 box filter on gaussian filter:')
print(mseForGaussFilterAndGaussFilterWithBoxFilter7x7)
print('PSNR of 7x7 box filter on gaussian filter:')
print(psnrForGaussFilterAndGaussFilterWithBoxFilter7x7)
print("\n")
cv2_imshow(medianFilter)
print('After applying median filter on gaussian filter')
print('MSE of median filter on gaussian filter:')
print(mseForGaussFilterAndGaussFilterWithMedianFilter)
print('PSNR of median filter on gaussian filter:')
print(psnrForGaussFilterAndGaussFilterWithMedianFilter)
waitKey()

"""**Result images after applying 3x3 box filter, 7x7 box filter, and median filter on salt and peper filter:**
"""

cv2_imshow(img)
print('Original image \n')
cv2_imshow(image)
print('Converting image to grayscale \n')
cv2_imshow(saltAndPeperFilter)
print('After applying S&P filter \n')
cv2_imshow(np.array(boxFilter3x3OnSaltAndPeper))
print('After applying a 3x3 box filter on S&P filter')
print('MSE of 3x3 box filter on S&P filter:')
print(mseForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter3x3)
print('PSNR 3x3 box filter on S&P filter:')
print(psnrForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter3x3)
print("\n")
cv2_imshow(np.array(boxFilter7x7OnSaltAndPeper))
print('After applying a 7x7 box filter on S&P filter')
print('MSE of 7x7 box filter on S&P filter:')
print(mseForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter7x7)
print('PSNR of 7x7 box filter on S&P filter:')
print(psnrForSaltAndPeperFilterAndSaltAndPeperFilterWithBoxFilter7x7)
print("\n")
cv2_imshow(medianFilterOnSaltAndPeper)
print('After applying median filter on S&P filter')
print('MSE of median filter on S&P filter:')
print(mseForSaltAndPeperFilterAndSaltAndPeperFilterWithMedianFilter)
print('PSNR of median filter on S&P filter:')
print(psnrForSaltAndPeperFilterAndSaltAndPeperFilterWithMedianFilter)
waitKey()
