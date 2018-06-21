import cv2
import numpy as np
from matplotlib import pyplot as plt
class DataAugmenter:
    HORIZONTAL_FLIP = 1
    VERTICAL_FLIP = 0
    def brightness(self, img, value, increase=True):
        #with help from https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if increase is True:
            limit = 255 - value
            v[v > limit] = 255
            v[v <= limit] += value
        else:
            limit = 0
            v[v == limit] = 0
            v[v >= limit] -= value

        final_hsv = cv2.merge((h, s, v))
        new_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return new_img

    def blur(self, img, box_size=(5,5)):
        blur = cv2.blur(img, box_size)
        return blur

    def add_noise(self, img, mean=(50,50,50), stddev=(50,50,50)):
        noisy_img = cv2.randn(img, mean, stddev)
        return noisy_img

    def translation(self, img, x=10, y=10):
        rows,cols = img.shape
        M = np.float32([[1,0,x],[0,1,y]])
        translated_image = cv2.warpAffine(img, M, (cols,rows))
        return translated_image

    def skew(self, img, point1, point2):
        rows,cols = img.shape
        M = cv2.getAffineTransform(point1, point2)
        skewed_img = cv2.warpAffine(img, M, (cols,rows))
        return skewed_img

    def rotation(self, img, degrees=90):
        rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
        rotated_image = cv2.warpAffine(img,M,(cols,rows))
        return rotated_image
    
    def flip(self, img,  direction):
        assert direction in (self.HORIZONTAL_FLIP, self.VERTICAL_FLIP)
        flipped_image = cv2.flip(img, direction)
        return flipped_image

    def show_image_changes(self, original_image, new_image):
        plt.subplot(121),plt.imshow(original_image),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(new_image),plt.title('Blurred')
        plt.xticks([]), plt.yticks([])
        plt.show()


#img = cv2.imread('opencv_logo.png')