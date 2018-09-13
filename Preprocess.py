import numpy as np
import cv2
'''
imageEnhanced use output as imageED when use with fundus image
'''

def resizeImage(img,shape):
    return cv2.resize(img,shape)
    
def imageEnhanced(img):
    imgE = cv2.addWeighted(src1 = img,alpha = 4, src2 = cv2.GaussianBlur(src=img,ksize=(0,0),sigmaX=5),beta = -4,gamma = 128)    
    mark = np.zeros(img.shape)
    mark = cv2.circle(mark,center=(int(imgE.shape[1]/2),int(imgE.shape[0]/2)),
           radius=int(imgE.shape[1]*0.36),color=(1,1,1),thickness=-1,lineType=8,shift=0)
    imgED = (imgE*mark)+(128*(1-mark))
    return imgE.astype('uint8')