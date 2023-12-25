# import cv2
# import urllib.request
# import numpy as np

# url = 'http://192.168.1.26:4747/video'

# cap = cv2.VideoCapture(url)

# while(True):
#     ret, frame = cap.read()
#     cv2.imshow('frame', frame)
#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
    
import cv2
url = "http://192.168.1.36:4747/video"
cap = cv2.VideoCapture(url)

while(True):
    ret, frame = cap.read()
    if ret == False:
        print('Camera not found!')
        break
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    if cv2.waitKey(1) == ord('c'):
        # capture image
        cv2.imwrite('ok.jpg', frame)
        