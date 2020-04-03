import os
import argparse
import cv2
import LaneFindVideo_GUI as gui

filepath = "test_videos/challenge.mp4"
filename = os.path.basename(filepath)
print(filename)

cap = cv2.VideoCapture(filepath)

if cap.isOpened() == False:
    print("Error opening video")

frm_rt = cap.get(cv2.CAP_PROP_FPS)      # Get frame rate
frm_dur = int(1000 / frm_rt)

cv2.namedWindow(filename)
cv2.moveWindow(filename, 100,100)

gui = gui.LaneFindVideoGUI()

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        frame = gui.find_lanes(frame)
        cv2.imshow(filename, frame)
        if cv2.waitKey(frm_dur) & 0xFF == ord('q'):
            break
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)


cap.release()

cv2.destroyAllWindows()
