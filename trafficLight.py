from __future__ import print_function

from collections import deque
from bluetooth import *
import numpy as np
import argparse
import imutils
import cv2
import sys
import time
import threading

# 소켓 생성
MyCar_socket = BluetoothSocket(RFCOMM)

# 블루투스 연결
MyCar_socket.connect(("insert MAC address", 1))
print("MyCar_connect:: success!!")

go = "go"
stop = "stop"
 
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
    help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
    help="max buffer size")
args = vars(ap.parse_args())
 
# hsv 경계
lower = {'red':(-5, 100, 100), 'green':(76, 100, 100), 'yellow':(11, 100, 100)}
upper = {'red':(15,255,255), 'green':(86,255,255), 'yellow':(31,255,255)}
 
# BGR
colors = {'red':(1,42,235), 'green':(151,168,27), 'yellow':(53, 183, 236)}

def printTraffic(traffic):
    print(traffic)
    MyCar_socket.send(traffic)

# 웹캡 켜기
if not args.get("video", False):
    camera = cv2.VideoCapture(0)
    print("camera open")
    
 
# 웹캠 연결 안돼있으면 영상 재생
else:
    camera = cv2.VideoCapture(args["video"])

while True:
    # 프레임 잡기
    (grabbed, frame) = camera.read()
    frame = cv2.flip(frame, -1)
    
    # 영상 못잡았다면 종료
    if args.get("video") and not grabbed:
        break
 
    # 웹캠 보여질 화면 크기 조정
    frame = imutils.resize(frame, width=600)
    
    # hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 회색조
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 멀리 있는 픽셀로 인해 필터 퀄리티 낮아지는 것 보완
    blurred = cv2.GaussianBlur(gray, (0, 0), 1)
    
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 10, param1=60, param2=70, minRadius=0, maxRadius=0)
    
    #검출된 원이 있다면
    if circles is not None:

        circles = np.uint16(np.around(circles))
            
        for i in circles[0, :]:
    
            # 색 검출
            for key, value in upper.items():
                kernel = np.ones((9,9),np.uint8)
                
                #cv2.circle(frame, (i[0], i[1]), int(i[2]), (0, 0, 255), 3, cv2.LINE_AA)
                # 모든 hsv값이 lower, upper 범위 안에 있는지 체크 후 결과값 반환 
                mask = cv2.inRange(hsv, lower[key], upper[key])
                # 모폴로지 열기 > 노이즈 제거
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                # 모폴로지 닫기 > 침식, 팽창 적용 > 작음 흠 제거, 연결선 두꺼워짐
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
                # 윤곽선 검출, 원의 중심 검출
                cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE)[-2]
                
                center = None
                
                
                # 색이 검출되었을 때
                if len(cnts) > 0.5:
                    # 주어진 컨투어의 외접한 원 검출
                    c = max(cnts, key=cv2.contourArea)
                    ((x, y), radius) = cv2.minEnclosingCircle(c)
                    M = cv2.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    if radius > 0.5:
                        # 프레임에 원 그려줌
                        cv2.circle(frame, (int(x), int(y)), int(radius), colors[key], 2)
                        # 무슨 색인지 출력
                        cv2.putText(frame,key, (int(x-radius),int(y-radius)), cv2.FONT_HERSHEY_SIMPLEX, 0.6,colors[key],2)
                        
                        if key == 'red' or key == 'yellow':
                            # 빨간불/노란불일 때 아두이노로 stop 신호 전송
                            printTraffic(stop)
                        
                            #sendStop()
                            #sleep(1)
                            
                        elif key == 'green':
                            # 초록불일 때 아두이노로 go 신호 전송
                            printTraffic(go)
                            #sendGo()
                            #sleep(1)
                            
    # 화면에 보여주기
    cv2.imshow("Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # q 누르면 종료
    if key == ord("q"):
        break
 
# 카메라 종료, 블루투스 소켓 종료, 모든 창 닫기
camera.release()
#MyCar_socket.close()
cv2.destroyAllWindows()