import cv2 as cv 
import numpy as np
import math

fx = 1464.04236 # 초점거리 x
fy = 1459.47648 # 초점거리 y
h = 2300 # 카메라 높이 mm 단위로 입력
theta_tilt = -49 #카메라 각도 degree 단위
width = 2560 # 이미지 넓이
height = 1440 # 이미지 높이

def distance_angle_measure(x,y):
    cx = width/2
    cy = height/2
    u = (x-cx)/fx
    v = (y-cy)/fy
    c_p_ = h * math.tan(math.pi/2+math.radians(theta_tilt) - math.atan(v))
    cp_ = math.sqrt(h*h+c_p_*c_p_)
    cp = math.sqrt(1+v*v)
    pp_ = u*cp_/cp
    d = math.sqrt(c_p_*c_p_ + pp_*pp_)
    theta = -math.atan2(pp_,c_p_)
    theta = math.degrees(theta)
    d = round(d/1000,2)
    return d,theta


def get_traveled(prev_dis, prev_theta, x,y):
    dis, theta = distance_angle_measure(x,y)
    # 이동 각도가 작으면 이동하지 않은  것으로 처리
    if abs(theta - prev_theta) <= 2:
        angle = 0
        traveled_distance = 0
    # 그렇지 않으면 코사인 법칙을 이용해서 이동 거리 구하기
    else:
        # 거리의 차로 사이각 구하기
        angle = theta - prev_theta

        # 코사인 법칙
        traveled_distance = math.sqrt(prev_dis**2 + dis**2 - (2*prev_dis*dis*math.cos(math.radians(angle))))
        traveled_distance = round(abs(traveled_distance), 2)

    return dis, theta, traveled_distance