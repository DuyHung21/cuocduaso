import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from skimage import feature

#X = np.load("X.npy")
#y = np.load("y.npy")
#hog_img = []

class HOG:
    def __init__(self, orientations = 9, pixelsPerCell = (8, 8), cellsPerBlock = (3, 3), normalize = False):
        self.orienations = orientations
        self.pixelsPerCell = pixelsPerCell
        self.cellsPerBlock = cellsPerBlock
        self.normalize = normalize

    def describe(self, image):
        hist = feature.hog(image, orientations = self.orienations,
                    pixels_per_cell = self.pixelsPerCell,
                    cells_per_block = self.cellsPerBlock,
                    block_norm = 'L2-Hys',
                    transform_sqrt = self.normalize)

        return hist

#hog = HOG(orientations = 18, pixelsPerCell = (8, 8), cellsPerBlock = (2, 2), normalize = True)

#def rgb2gray(img):
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


#for img in X:
    #img = hog.describe(rgb2gray(img))
    #hog_img.append(img)

#np.save("HOG_X.npy",np.asarray(hog_img))
#print (img.shape)
    
