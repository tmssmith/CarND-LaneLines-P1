import numpy as np
import cv2
import os
import argparse

left_lines = []
right_lines = []

def separate_lines(lines):
    for line in lines:
        for x1,y1,x2,y2 in line:
            grad = (y2-y1)/(x2-x1)
            if grad <= 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    separated = (left_lines, right_lines)
    return separated

def extrapolate_line(lines, img_height = 539, roi_height=335):
    """
    This function extrapolates a best fit / average line from multiple lines.
    The best fit line is derived from the average gradient and intercepts of
    all lines
    """

    m = []
    b = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            grad = (y2-y1)/(x2-x1)
            m.append(grad)
            b.append(y1 - (grad * x1))

    mu_m = sum(m) / len(m)
    mu_b = sum(b) / len(b)

    y1 = roi_height
    x1 = int((y1 - mu_b) / mu_m)
    y2 = img_height
    x2 = int((y2 - mu_b) / mu_m)

    return(x1,y1,x2,y2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help = "*.npy file for hough lines")
    args = parser.parse_args()
    hough_lines = np.load(args.file)
    separated_lanes = separate_lines(hough_lines)
    left_lane = extrapolate_line(separated_lanes[0])
    right_lane = extrapolate_line(separated_lanes[1])

    cv2.namedWindow("Lane Lines")
    canvas = np.zeros((540, 960, 3), dtype=np.uint8)
    cv2.line(canvas, (left_lane[0], left_lane[1]),(left_lane[2], left_lane[3]), (0,0,255),2)
    cv2.line(canvas, (right_lane[0], right_lane[1]),(right_lane[2], right_lane[3]), (0,255,0),2)
    img_orig = cv2.imread("test_images/solidWhiteRight.jpg")
    final = cv2.addWeighted(canvas, 0.5,  img_orig, 0.5, 0)
    cv2.imshow("Lane Lines", final)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
