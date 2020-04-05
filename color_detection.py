import os
import argparse
import cv2
import numpy as np

def initTrackbars():
    cv2.namedWindow('Trackbars')
    cv2.createTrackbar('H lower W', 'Trackbars', 0, 180, callback)
    cv2.createTrackbar('H upper W', 'Trackbars', 180, 180, callback)
    cv2.createTrackbar('L lower W', 'Trackbars', 0, 255, callback)
    cv2.createTrackbar('L upper W', 'Trackbars', 255, 255, callback)
    cv2.createTrackbar('S lower W', 'Trackbars', 0, 255, callback)
    cv2.createTrackbar('S upper W', 'Trackbars', 255, 255, callback)
    cv2.createTrackbar('H lower Y', 'Trackbars', 0, 180, callback)
    cv2.createTrackbar('H upper Y', 'Trackbars', 180, 180, callback)
    cv2.createTrackbar('L lower Y', 'Trackbars', 0, 255, callback)
    cv2.createTrackbar('L upper Y', 'Trackbars', 255, 255, callback)
    cv2.createTrackbar('S lower Y', 'Trackbars', 0, 255, callback)
    cv2.createTrackbar('S upper Y', 'Trackbars', 255, 255, callback)

def callback(value):
    pass

def color_detection(img):
    w_h_lo = cv2.getTrackbarPos('H lower W', 'Trackbars')
    w_h_hi = cv2.getTrackbarPos('H upper W', 'Trackbars')
    w_l_lo = cv2.getTrackbarPos('L lower W', 'Trackbars')
    w_l_hi = cv2.getTrackbarPos('L upper W', 'Trackbars')
    w_s_lo = cv2.getTrackbarPos('S lower W', 'Trackbars')
    w_s_hi = cv2.getTrackbarPos('S upper W', 'Trackbars')
    y_h_lo = cv2.getTrackbarPos('H lower Y', 'Trackbars')
    y_h_hi = cv2.getTrackbarPos('H upper Y', 'Trackbars')
    y_l_lo = cv2.getTrackbarPos('L lower Y', 'Trackbars')
    y_l_hi = cv2.getTrackbarPos('L upper Y', 'Trackbars')
    y_s_lo = cv2.getTrackbarPos('S lower Y', 'Trackbars')
    y_s_hi = cv2.getTrackbarPos('S upper Y', 'Trackbars')

    w_lo_thr = np.array([w_h_lo, w_l_lo, w_s_lo])
    w_hi_thr = np.array([w_h_hi, w_l_hi, w_s_hi])

    y_lo_thr = np.array([y_h_lo, y_l_lo, y_s_lo])
    y_hi_thr = np.array([y_h_hi, y_l_hi, y_s_hi])

    img_hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img_msk_w = cv2.inRange(img_hls, w_lo_thr, w_hi_thr)
    img_msk_y = cv2.inRange(img_hls, y_lo_thr, y_hi_thr)
    img_msk = cv2.bitwise_or(img_msk_w, img_msk_y)

    img_colorMask = cv2.bitwise_and(img, img, mask = img_msk)
    return img_colorMask

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help = "file for lane finding")
    args = parser.parse_args()
    filename = os.path.basename(args.file)
    extension = os.path.splitext(args.file)[1]

    if extension == '.jpg':
        img = cv2.imread(args.file)
        if img is not None:
            initTrackbars()
            cv2.namedWindow(filename)
            cv2.moveWindow(filename, 100,100)
            while True:
                img_lanes = color_detection(img)
                cv2.imshow(filename, img_lanes)
                if cv2.waitKey(40) & 0xFF == ord('q'):
                    break
        else:
            print('Error opening image')
    elif extension == '.mp4':
        cap = cv2.VideoCapture(args.file)
        if cap.isOpened() == False:
            print("Error opening video")
        frm_rt = cap.get(cv2.CAP_PROP_FPS)      # Get frame rate
        frm_dur = int(1000 / frm_rt)
        initTrackbars()
        cv2.namedWindow(filename)
        cv2.moveWindow(filename, 100,100)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_lanes = color_detection(frame)
                cv2.imshow(filename, frame_lanes)
                if cv2.waitKey(frm_dur) & 0xFF == ord('q'):
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap.release()
    else:
        "Filetype must be *.jpg or *.mp4"

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
