def hist_eq(im, bin):
    fig = plt.figure()
    fig.set_size_inches(15,10)
    
    a=fig.add_subplot(3,2,1) # Original Color Image
    imarray = np.array(im)
    plt.imshow(imarray)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    imgray_array = np.array(imgray)
 
    hist, bins = np.histogram(im.flatten(),256,[0,256])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()

    a=fig.add_subplot(3,2,2)
    h = (256-1)*(cdf - cdf.min())/(cdf.max()-cdf.min())
    transformed_img = h[imgray_array]
    plt.title('Transformed Image');
    plt.imshow(transformed_img, cmap=plt.get_cmap('gray'));

    # Original Image
    a=fig.add_subplot(3,2,3)
    plt.title('Histogram, CDF of Original Image');
    plt.hist(imarray.flatten(), 256, [0,256], color = 'r'); # Plot Histogram
    plt.plot(cdf_normalized, color = 'b'); # Plot normalized CDF
    plt.xlim([0,256]);
    plt.legend(('cdf','histogram'), loc = 'upper right');


    # Transformed Image
    a=fig.add_subplot(3,2,4)
    plt.title('Histogram, CDF of Transformed Image');
    hist, bins = np.histogram(transformed_img.flatten(),256,[0,256])
    if bin == 256 :
        plt.hist(transformed_img.flatten(), 256, [0,256], color = 'r');
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        plt.plot(cdf_normalized, color = 'b');
        plt.xlim([0,256]);
        plt.legend(('cdf','histogram'), loc = 'upper right');
    elif bin == 128 :
        plt.hist(transformed_img.flatten(), 128, [0,128], color = 'r');
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        plt.plot(cdf_normalized, color = 'b');
        plt.xlim([0,128]);
    elif bin == 64 :
        plt.hist(transformed_img.flatten(), 64, [0,64], color = 'r');
        cdf = hist.cumsum()
        cdf_normalized = cdf * hist.max()/ cdf.max()
        plt.plot(cdf_normalized, color = 'b');
        plt.xlim([0,64]);
    else:
         print ("Sorry you entered an invalid bin")

import cv2
from PIL import Image # Load PIL(Python Image Library)
import matplotlib.pyplot as plt # Load pyplot library
import numpy as np # Load Numerical Library
from cv2 import COLOR_BGR2GRAY, cvtColor, imread, imshow, waitKey
image = cv2.imread('/content/jetplane.tif')
hist_eq(image,256)
hist_eq(image,128)
hist_eq(image,64)
