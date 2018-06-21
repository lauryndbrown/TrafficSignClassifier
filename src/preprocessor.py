import cv2
import numpy as np
class Preprocessor:
    def histogram_equalization(self, img):
        #https://stackoverflow.com/questions/31998428/opencv-python-equalizehist-colored-image
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output
   
    def greyscale(self, img):
        #https://stackoverflow.com/questions/8486294/how-to-add-an-extra-column-to-a-numpy-array
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grey_with_channel = np.c_[ grey, np.ones(len(grey)) ] 
        return grey_with_channel
        #https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
        #return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
    
    def normalize(self, img):
        #converting pixels to -1 to 1 range as opposed to 0-255
        #np.fromfunction(lambda i, j: i == j, (3, 3), dtype=int)
        for index, pixel in enumerate(img):
            img[index] = (pixel-128)/128 
        return img