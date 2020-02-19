import cv2
import sys
import numpy as np

cap = cv2.VideoCapture('market.mp4')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

ret, frame = cap.read()
while (ret):
	ret, frame = cap.read()	
	frame= cv2.resize(frame,(400, 300))
	(rects, weights) = hog.detectMultiScale(frame, scale = 1, winStride=(4, 4))
	for (x, y, w, h) in rects:
		cv2.rectangle(frame, (x, y), (x + w, y + h), (127,0,0), 2)
	cv2.imshow("Frame", frame)
	if (cv2.waitKey(1) == ord('q')):
		break 

cap.release()
cv2.destroyAllWindows()