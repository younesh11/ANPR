import cv2
import numpy as np
import imutils 
import pytesseract 
import OCR as OCR

flag = 0
img = cv2.imread('Data/1.jpg')

img = cv2.resize(img, (620,480))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edgeDetect = cv2.Canny(gray, 90, 200)

contours = cv2.findContours(edgeDetect.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
screenCnt = None
cv2.imshow('img',edgeDetect)
cv2.waitKey(0)

for c in contours:
                
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.018 * peri, True)
                if len(approx) == 4:
                      screenCnt = approx
                      break
if screenCnt is None:
    flag = 0

    print('no contours detected')

else:
    flag = 1

if flag == 1:
    
    cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)

#MASKING THE IMAGE OTHER THAN  NUMBERPLATE 

mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

#CROP  NUMBERPLATE
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
cropped = gray[topx:bottomx+1, topy:bottomy+1]

text = OCR.imgToText(cropped)


cv2.imshow('image', img)
cv2.waitKey(0)
cv2.imshow('image1', new_image)
cv2.waitKey(0)
cv2.imshow('cropped', cropped)
cv2.waitKey(0)

cv2.destroyAllWindows()


