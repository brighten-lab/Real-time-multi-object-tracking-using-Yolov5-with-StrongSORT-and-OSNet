import datetime

import cv2

 

capture = cv2.VideoCapture(1)

 

width = int(capture.get(3))  # 가로

height = int(capture.get(4))  # 세로값 가져와서

num = 50

while (capture.isOpened):

    ret, frame = capture.read()

    if ret == False:

        break

    cv2.imshow("VideoFrame", frame)

 

    now = datetime.datetime.now().strftime("%d_%H-%M-%S")

    key = cv2.waitKey(33)  # 1) & 0xFF

 

    if key == 27:  # esc 종료

        break

    elif key == 26:  # ctrl + z
        num +=1         
        cv2.IMREAD_UNCHANGED

        # cv2.imwrite(f"C:\\internship\\capturing_cal\\{num}.png", frame)
        cv2.imwrite(f"./{num}.png", frame)
 

capture.release()

cv2.destroyAllWindows()