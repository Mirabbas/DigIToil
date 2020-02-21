from django.shortcuts import render
from django.http import HttpResponse,StreamingHttpResponse, HttpResponseServerError,HttpResponseRedirect
from django.views.decorators import gzip

from django.shortcuts import redirect

from imutils.video import VideoStream
from imutils.video import FPS
import mimetypes

import cv2
from time import time
import imutils
from math import sqrt
import numpy as np


myauth = False




BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]]

ID = 1
renet = cv2.dnn.readNet('person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.xml',
                            'person-reidentification-retail-0079/FP32/person-reidentification-retail-0079.bin')
net = cv2.dnn.readNet('person-detection-retail-0013/FP32/person-detection-retail-0013.xml',
                          'person-detection-retail-0013/FP32/person-detection-retail-0013.bin')
est_net = cv2.dnn.readNet('human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml',
                          'human-pose-estimation-0001/FP32/human-pose-estimation-0001.bin')                            
netsize = (544, 320)
renetsize = (64, 160) 
estnetsize = (456, 256)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
renet.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
renet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  
est_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
est_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 
#___________________________________VIDEO NAME/CHANGE TO 0 to get stream FROM WEBCAM
video = cv2.VideoCapture('2.mp4') 

distance = 0    
chips = []   
times = {}
data = {}
fps = 1
trigger = 0.8
dist_trigger = 0.3   
f_trigger = 0.5


def compare(data, chip, save=0):
    reblob = cv2.dnn.blobFromImage(chip, size=renetsize, ddepth=cv2.CV_8U)
    renet.setInput(reblob)
    reout = renet.forward()
    reout = reout.reshape(256)
    reout /= sqrt(np.dot(reout, reout))
    ide = 1
    distance = -1

    if len(data) != 0:
        for x in data:
            distance = np.dot(reout, data[x])
            ide += 1
            if distance > dist_trigger:
                ide = x
                break

    if distance < dist_trigger:
        data['id{}'.format(ide)] = reout
        if save:
            cv2.imwrite('photos/id{}.jpg'.format(ide), chip)

    return distance, ide
      
class VideoCamera():
    
    def get_frame(self):
        start = time()
        grab, frame = video.read()
        if not grab:
            raise Exception('Image not found!')

        blob = cv2.dnn.blobFromImage(frame, size=netsize, ddepth=cv2.CV_8U)
        net.setInput(blob)
        out = net.forward()

        objects = 0
        for detection in out.reshape(-1, 7):
            confidence = float(detection[2])
            xmin = int(detection[3] * frame.shape[1])
            ymin = int(detection[4] * frame.shape[0])
            xmax = int(detection[5] * frame.shape[1])
            ymax = int(detection[6] * frame.shape[0])

            if confidence > trigger:
                objects += 1
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                chip = frame[ymin:ymax, xmin:xmax]                
                try:               
                    distance, ID = compare(data, chip, 1)
                except:
                    continue

                cv2.putText(frame, '{}'.format(ID), (xmin, ymax - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
#_____________________________________________________________________________________
#_____________HUMAN_POSE_ESTIMATION___________________________________________________
#_____________________________________________________________________________________
#_____________________________________________________________________________________
        crop_frame = frame[ymin:ymax, xmin:xmax]
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        est_blob = cv2.dnn.blobFromImage(frame, size=estnetsize, ddepth=cv2.CV_8U)
        est_net.setInput(est_blob)
        est_out = est_net.forward()
        est_out = est_out[:, :19, :, :]
        assert(len(BODY_PARTS) == est_out.shape[1])
        points = []

        for i in range(len(BODY_PARTS)):
            heatMap = est_out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / est_out.shape[3]
            y = (frameHeight * point[1]) / est_out.shape[2]
            points.append((int(x), int(y)) if conf > 0.2 else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        t, _ = est_net.getPerfProfile()
        freq = cv2.getTickFrequency() / 1000
        cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
#_____________________________________________________________________________________
#_____________HUMAN_POSE_ESTIMATION(DONE)_____________________________________________
#_____________________________________________________________________________________
#_____________________________________________________________________________________

        end = time()
        fps = 1 / (end - start)

        cv2.putText(frame, 'fps:{:.2f}'.format(fps), (5, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'objects:{}'.format(objects), (5, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, 'IDs:{}'.format(len(data)), (5, frame.shape[0]-45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        jpeg = cv2.imencode('.jpg',frame)[1].tostring()
        '''print(jpeg)
        j = jpeg.tobytes()
        print(j)'''
        return jpeg


def gen(camera):
    while True:
        frame_1 = VideoCamera().get_frame()
        yield(b'--frame_1\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + frame_1 + b'\r\n\r\n')
        
'''
def indexscreen(request): 
    try:
        template = "screens.html"
        return render(request,template)

    except HttpResponseServerError:
        print("aborted")
'''


#___________________________________AUTH_UPDATE__________________________________

def indexscreen(request): 
   global myauth
   try:
      if myauth:
        template = "screens.html"
        return render(request,template)
      else:
        template = "auth.html"
        return render(request,template)
   except HttpResponseServerError:
        print("aborted")
       

      
def auth(request):
    global myauth
    print('-------------------------------------------')
    print(request)
    req = str(request).split('/')
    name = req[2]
    pas = req[3]
    pas = pas[:-2]
    print(str(req))
    print('-------------------------------------------')
    print(name)
    print(pas)
    if name == 'a' and pas =='1':
        myauth = True
        print('auth succesful')
        return HttpResponseRedirect('/stream/screen')
def del_auth(request):
    global myauth
    myauth = False
    return redirect('/stream/screen')


@gzip.gzip_page
def dynamic_stream(request,num=0,stream_path="2.mp4"):
    stream_path = 'add your camera stream here that can rtsp or http'
    return StreamingHttpResponse(gen(VideoCamera()),content_type="multipart/x-mixed-replace;boundary=frame")

