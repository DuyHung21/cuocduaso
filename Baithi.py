import cv2
import numpy as np
import os
import scipy.misc
import pandas as pd
import tensorflow as tf
import pandas as pd
import mahotas
import imutils
from imutils.perspective import four_point_transform
from hog_img_extraction_self_implement import HOG
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
    "45": 7,
    "46": 7,
    "48": 7
}
SIGN_NAMES = ["Dung", "Re trai", "Re phai", "Cam re trai", "Cam re phai", "Mot chieu", "Toc do", "Khac", "Khong phai"]

TARGET_SIGN = 7

INPUT_SHAPE = (32, 32, 1)


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
minW = 20
minH = 20
kernel = np.ones((3,3),np.uint8)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_MVI_1049.avi',fourcc, 20.0, (640,480))

hog = HOG(orientations = 9, pixelsPerCell = (3, 3), cellsPerBlock = (2, 2), normalize = True)

svm = pickle.load(open("svm_001.p", "rb"))
#svm = pickle.load(open("svm_proba.p", "rb"))
def rgb2gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,0]
    return img

def bgr2gray(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)
def resize_image(image, shape=INPUT_SHAPE[:2]):
    return cv2.resize(image, shape)
def extract_hog(img):
    return hog.describe(img)

def is_ts(image):
    gray = rgb2gray(image)
    #thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    thresh = gray.copy()
    T = mahotas.thresholding.otsu(thresh)
    thresh[thresh > T] = 255
    thresh = cv2.bitwise_not(thresh)
    
    thresh = extract_hog(thresh)
    thresh = np.expand_dims(thresh, axis=0)
    gray = extract_hog(gray)
    gray = np.expand_dims(gray, axis=0)
    prediction_thresh = svm.predict(thresh)
    prediction_gray = svm.predict(gray)
    return int(prediction_thresh) == 1 or int(prediction_gray) == 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile('checkpoint/frozen_model.pb', 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    with tf.Session(graph=detection_graph) as sess:
        for op in sess.graph.get_operations():
            print(op.name)
        image_tensor = detection_graph.get_tensor_by_name('input/x:0')
        is_training = detection_graph.get_tensor_by_name('input/is_training:0')
        output_prediction = detection_graph.get_tensor_by_name('prediction/prediction:0')
        output_probability = detection_graph.get_tensor_by_name('prediction/probability:0')
        
        def process_video(path = "./Videos/MVI_1049.avi"):
            cap = cv2.VideoCapture(path)
            print("Hi")
            n = 0
            lines = []
            lines.append(str(n)+"\n")
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


                #Extract traffic sign
                    
                    for i in range(0, len(cnts)):
                        found = False
                        cnt = cnts[i]
                        x,y,w,h = cv2.boundingRect(cnt)
                        offset = 5
                        if w > minW and h > minH and float(h)/w > 0.9 and float(h)/w < 1.5:
                            beginY, endY, beginX, endX = y,y+h+offset,x,x+w+offset
                            if beginY > offset:
                                beginY -= offset
                            if beginX > offset:
                                beginX -= offset
                            cropped = frame[beginY:endY, beginX:endX]
                            #cropped = cv2.GaussianBlur(cropped, (3, 3), 0)
                            resize_img = resize_image(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                        
                            if not is_ts(resize_img):
                                continue
                            cv2.imshow("resized", resize_img)
                            resize_img = np.expand_dims(resize_img, axis=0)
                            (prediction, probability) = sess.run(
                                [output_prediction, output_probability],
                                feed_dict={image_tensor: resize_img, is_training: False})
                            #print (np.max(probability))
                            if np.max(probability) > 0.9:
                                #cv2.imshow("Cropped", cropped)
                                if str(int(prediction)) in LABEL_MAP:
                                    name_id = LABEL_MAP[str(int(prediction))]
                                else:
                                    print (int(prediction))
                                    name_id = 8
                                    #print (SIGN_NAMES[name_id-1])
                                #print (SIGN_NAMES[name_id-1])
                                if name_id == TARGET_SIGN:
                                    cv2.imshow(SIGN_NAMES[name_id-1], cropped)
                                    cv2.putText(frame, SIGN_NAMES[name_id-1], (x, y), font, fontScale, fontColor, lineType)
                                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 1)
                                    lines.append(str(n)+" "+str(name_id)+" "+str(x)+" "+str(y)+" "+str(x+w)+" "+str(y+h)+"\n")
                                    n+=1
                                    found = True
                                    break
                        if found:
                            break
                        prediction = None
                        probability = None
                    cv2.imshow("mask", mask)
                    cv2.imshow("Origion", frame)
                    out.write(frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            lines[0] = str(n)+"\n"
            with open("result.txt", "w") as f:
                f.writelines(lines)
        process_video("./Videos/MVI_1049.avi")










