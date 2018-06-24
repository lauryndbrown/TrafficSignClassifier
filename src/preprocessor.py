import cv2
import numpy as np
IMAGE_SIZE = (32, 32)
class Preprocessor:
    def histogram_equalization(self, img):
        img = img.copy()
        #https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output
   
    def greyscale(self, img):
        img = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
       
    def normalize(self, img):
        img = img.copy()
        img = img.astype(np.float32)
        for index, pixel in enumerate(img):
            img[index] = pixel/256.0 
        return img
    
    def resize(self, img, size=IMAGE_SIZE):
        img = img.copy()
        resized_image = cv2.resize(img, size) 
        return resized_image
