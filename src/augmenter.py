import cv2
import numpy as np
from matplotlib import pyplot as plt
class DataAugmenter:
    CORNER_FLIP = -1
    HORIZONTAL_FLIP = 1
    VERTICAL_FLIP = 0
    def brightness(self, img, value, increase=True):
        #with help from https://stackoverflow.com/questions/32609098/how-to-fast-change-image-brightness-with-python-opencv
        img = img.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        if increase is True:
            limit = 255 - value
            v[v > limit] = 255
            v[v <= limit] += value
        else:
            limit = value
            v[v < limit] = 0
            v[v >= limit] -= value

        final_hsv = cv2.merge((h, s, v))
        new_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return new_img

    def blur(self, img, box_size=(2,2)):
        img = img.copy()
        blur = cv2.blur(img, box_size)
        return blur

    def translation(self, img, x=10, y=10):
        img = img.copy()
        rows,cols = img.shape[0], img.shape[1]
        M = np.float32([[1,0,x],[0,1,y]])
        translated_image = cv2.warpAffine(img, M, (cols,rows))
        return translated_image

    def rotation(self, img, degrees=90):
        img = img.copy()
        rows,cols = img.shape[0], img.shape[1]
        M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
        rotated_image = cv2.warpAffine(img,M,(cols,rows))
        return rotated_image
    
    def flip(self, img,  direction):
        img = img.copy()
        assert direction in (self.HORIZONTAL_FLIP, self.VERTICAL_FLIP, self.CORNER_FLIP)
        flipped_image = cv2.flip(img, direction)
        return flipped_image

    def show_image_changes(self, original_image, new_image, effect):
        plt.subplot(121),plt.imshow(original_image),plt.title('Original')
        plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(new_image),plt.title(effect)
        plt.xticks([]), plt.yticks([])
        plt.show()
    
    def show_multiple_image_changes(self, columns, rows, images, names):     
        fig=plt.figure(figsize=(8, 8))
        for i, img, name in zip(range(1, columns*rows +1), images, names):
            fig.add_subplot(rows, columns, i)
            plt.imshow(img)
            plt.title(name)
        plt.show()
