import cv2
import numpy as np
import os
import scipy.misc
import pandas as pd
import tensorflow as tf
import pandas as pd
import mahotas
from imutils.perspective import four_point_transform
from skimage import feature
 
import pickle


LABEL_MAP = {
    "0": 7,
    "1": 7,
    "2": 7,
    "3": 7,
    "4": 7,
    "5": 7,
    "6": 7,
    "7": 7,
    "8": 7,
    "14": 1,
    "34": 2,
    "33": 3,
    "43": 4,
    "44": 5,
    "17": 6,
    "45": 9,
    "46": 9
}
SIGN_NAMES = ["Dung", "Re trai", "Re phai", "Cam re trai", "Cam re phai", "Mot chieu", "Toc do", "Khac", "Khong phai"]


INPUT_SHAPE = (32, 32, 3)

font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
minW = 20
minH = 20
kernel = np.ones((3,3),np.uint8)

svm = pickle.load(open("svm.p", "rb"))

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

def resize_image(image, shape=INPUT_SHAPE[:2]):
    return cv2.resize(image, shape)
def load_image(image_file):
    """
    Read image file into numpy array (RGB)
    """
    return plt.imread(image_file)

hog = HOG(orientations = 9, pixelsPerCell = (3, 3), cellsPerBlock = (2, 2), normalize = True)

def rgb2gray(img):
    return cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

def extract_hog(img):
    return hog.describe(img)




def process_video(path = "./Videos/MVI_1049.avi"):
    cap = cv2.VideoCapture(path)
    #print("Hi")
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
                    
            #Turn BGR to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            #Filter red, blue
            mask_blue = cv2.inRange(hsv, (100, 120, 100), (120, 255, 255))

            mask_red_lower = cv2.inRange(hsv, (0, 100, 100), (15, 255, 255))
            mask_red_upper = cv2.inRange(hsv, (160, 100, 120), (180, 255, 255))

            mask = cv2.add(mask_red_lower, mask_red_upper)
            mask = cv2.add(mask, mask_blue)
                    
            # Apply Gausian blur
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
                        
            T = mahotas.thresholding.otsu(mask)

            # Erode to reduce noise and dilate to focus

            mask = cv2.Canny(mask, T * 0.5, T)
                    
            # Erode to reduce noise and dilate to focus
            #mask = cv2.erode(mask, None, iterations = 1)
            #mask = cv2.dilate(mask, None, iterations = 3)

            #Find countour
            cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]
                
            #Draw contours
            if len (cnts) > 0:
                mask = cv2.drawContours(mask, cnts, -1, 255, -1)
                mask = cv2.dilate(mask, kernel, iterations=5)
                mask = cv2.erode(mask, kernel, iterations=5)
                
            cnts = cv2.findContours(image = mask.copy(), mode = cv2.RETR_EXTERNAL, method = cv2.CHAIN_APPROX_SIMPLE)[-2]

            #Filter contours
                                
            #Extract traffic sign

            for i in range(0, len(cnts)):
                cnt = cnts[i]
                x,y,w,h = cv2.boundingRect(cnt)
                for offset in range(10):
                    if w > minW and h > minH and float(h)/w > 0.9 and float(h)/w < 1.5:
                        beginY, endY, beginX, endX = y,y+h+offset,x,x+w+offset
                        if beginY > offset:
                            beginY -= offset
                        if beginX > offset:
                            beginX -= offset
                        cropped = frame[beginY:endY, beginX:endX]
                        #cropped = cv2.GaussianBlur(cropped, (3, 3), 0)
                        thresh = rgb2gray(cropped)
                        T = mahotas.thresholding.otsu(thresh)
                        thresh[thresh > T] = 255
                        thresh = cv2.bitwise_not(thresh)
                        thresh = resize_image(thresh)
                        cv2.imshow("candidate", thresh)
                        hog_img = extract_hog(thresh)
                        hog_img = np.expand_dims(hog_img, axis=0)
                        prediction = svm.predict(hog_img)
                        #proba = svm.predict_proba(hog_img)
                        if int(prediction) == 1:
                            #print (proba[0][1])
                            #cv2.imshow("is_ts", cropped)
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                            break
            cv2.imshow("Origion", frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()
process_video("./Videos/MVI_1055.avi")
        










