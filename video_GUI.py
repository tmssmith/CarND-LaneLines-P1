import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from numpy.polynomial.polynomial import polyfit

class LaneFindVideoGUI:
    def __init__(self, size):

        self.size = size
        self.height = self.size[0]
        self.width = self.size[1]

        self.roi_top = 10
        self.roi_bottom = self.height - 10
        self.roi_topleft = 10
        self.roi_topright = self.width - 10
        self.roi_bottomleft = 10
        self.roi_bottomright = self.width-10

        self.kernel = 5
        self.low = 50
        self.ratio = 30
        self.high = self.low * self.ratio / 10

        self.rho = 2
        self.theta_bar = 1
        self.theta = self.theta_bar * (np.pi / 180)
        self.threshold = 10
        self.min_line_len = 30
        self.max_line_gap = 10

        self.image_selector = 0
        self.alpha = 0.5
        self.beta = 1 - self.alpha
        self.gamma = 0

        cv2.namedWindow("Parameters", flags = 2)
        cv2.moveWindow("Parameters",700,150)

        cv2.createTrackbar('Kernel Size', 'Parameters', self.kernel, 10, self.onChange_kernel)
        cv2.createTrackbar('Low Threshold', 'Parameters', self.low, 255, self.onChange_low)
        cv2.createTrackbar('Canny Ratio', 'Parameters', self.ratio, 50, self.onChange_ratio)

        cv2.createTrackbar('ROI top', 'Parameters',  self.roi_top, self.height, self.onChange_roitop)
        cv2.createTrackbar('ROI bottom', 'Parameters', self.roi_bottom, self.height, self.onChange_roibottom)
        cv2.createTrackbar('ROI top left', 'Parameters',  self.roi_topleft, self.width, self.onChange_roitopleft)
        cv2.createTrackbar('ROI top right', 'Parameters', self.roi_topright, self.width, self.onChange_roitopright)
        cv2.createTrackbar('ROI bottom left', 'Parameters', self.roi_bottomleft, self.width, self.onChange_roibottomleft)
        cv2.createTrackbar('ROI bottom right', 'Parameters', self.roi_bottomright, self.width, self.onChange_roibottomright)

        cv2.createTrackbar('Rho', 'Parameters', self.rho, 100, self.onChange_rho)
        cv2.createTrackbar('Theta', 'Parameters', self.theta_bar, 100, self.onChange_theta)
        cv2.createTrackbar('Threshold', 'Parameters', self.threshold, 100, self.onChange_threshold)
        cv2.createTrackbar('Min Line Len', 'Parameters', self.min_line_len, 100, self.onChange_minlinelen)
        cv2.createTrackbar('Max Line Gap', 'Parameters', self.max_line_gap, 100, self.onChange_maxlinegap)

        cv2.createTrackbar('Base Image', 'Parameters', self.image_selector, 3, self.onChange_baseimage)
        cv2.createTrackbar('Overlay Ratio', 'Parameters', int(self.alpha*100), 100, self.onChange_alpha)
        cv2.createTrackbar('Image Gamma', 'Parameters', int(self.gamma), 255, self.onChange_gamma)

    def onChange_kernel(self, value):
        value = max(1, value)
        self.kernel = 2 * value - 1

    def onChange_low(self, value):
        value = max(1, value)
        self.low = value
        self.high = self.low * self.ratio

    def onChange_ratio(self, value):
        if value == 0:
            value = 1
        self.ratio = value / 10
        self.high = self.low * self.ratio

    def onChange_rho(self,value):
        value = max(1, value)
        self.rho = value

    def onChange_theta(self,value):
        value = max(1, value)
        self.theta = value * (np.pi / 180)

    def onChange_threshold(self,value):
        value = max(1, value)
        self.threshold = value

    def onChange_minlinelen(self,value):
        value = max(1, value)
        self.min_line_len = value

    def onChange_maxlinegap(self,value):
        value = max(1, value)
        self.max_line_gap = value

    def onChange_baseimage(self,value):
        self.image_selector = value

    def onChange_alpha(self, value):
        value = max(1, value)
        self.alpha = value / 100
        self.beta = 1 - self.alpha

    def onChange_gamma(self, value):
        value = max(1, value)
        self.gamma = value

    def onChange_roitop(self, value):
        self.roi_top = value

    def onChange_roibottom(self, value):
        self.roi_bottom = value

    def onChange_roitopleft(self, value):
        self.roi_topleft = value

    def onChange_roitopright(self, value):
        self.roi_topright = value

    def onChange_roibottomleft(self, value):
        self.roi_bottomleft = value

    def onChange_roibottomright(self, value):
        self.roi_bottomright = value

    def find_lanes(self, img):
        self.img = img
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.img_gaus = cv2.GaussianBlur(self.img_gray, (self.kernel, self.kernel), 0)
        self.img_canny = cv2.Canny(self.img_gaus, self.low, self.high)

        self.mask(self.img_canny)
        self.img_mask = cv2.cvtColor(self.img_mask, cv2.COLOR_GRAY2BGR)

        self.hough_lines = cv2.HoughLinesP(self.img_roi, self.rho, self.theta, self.threshold, np.array ([ ]), self.min_line_len, self.max_line_gap)
        self.img_hough = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
        self.img_best_fit = np.zeros((self.img.shape[0], self.img.shape[1], 3), dtype=np.uint8)
        if self.hough_lines is not None:
            self.draw_lines(self.img_hough, self.hough_lines)
            (left_lines, right_lines) = self.separate_lines(self.hough_lines)
            self.best_fit_lines(self.img_best_fit, left_lines)
            self.best_fit_lines(self.img_best_fit, right_lines)
        else:
            print("No lines found")

        self.image_list = (self.img, self.img_gaus, self.img_canny, self.img_hough)
        self.base_image = self.image_list[self.image_selector]

        if len(self.base_image.shape) == 2:
            self.base_image = cv2.cvtColor(self.base_image, cv2.COLOR_GRAY2BGR)

        self.img_masked = cv2.addWeighted(self.base_image, self.alpha, self.img_mask, self.beta, self.gamma)
        self.img_final = cv2.addWeighted(self.img_masked, self.alpha, self.img_best_fit, self.beta, self.gamma)

        return self.img_final

    def draw_lines(self, img, lines, color=[0, 0, 255], thickness=2):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def mask(self, image):
        """
        Creates an image mask.

        Helps user define region of interest
        """
        #defining a blank mask to start with
        self.vertices = np.array([\
            [self.roi_bottomleft, self.roi_bottom],\
            [self.roi_topleft, self.roi_top],\
            [self.roi_topright, self.roi_top],\
            [self.roi_bottomright, self.roi_bottom]], np.int32)

        self.img_mask = np.zeros_like(image)
        if len(self.size) > 2:
            channel_count = self.size[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(self.img_mask, [self.vertices], ignore_mask_color)

        self.img_roi = cv2.bitwise_and(image, self.img_mask)

    def separate_lines(self, lines):
        left_lines = []
        right_lines = []

        for line in lines:
            for x1,y1,x2,y2 in line:
                grad = (y2-y1)/(x2-x1)
                if grad <= -0.5 and grad >= -1:
                    left_lines.append(line)
                elif grad <= 1 and grad > 0.5:
                    right_lines.append(line)

        return (left_lines, right_lines)

    def best_fit_lines(self, img, lines):
        x = []
        y = []

        points = self.get_points(lines)

        for point in points:
            x.append(point[0])
            y.append(point[1])

        if x and y:
            b, m = polyfit(x, y, 1)

            y1 = self.roi_top
            x1 = int((y1 - b) / m)
            y2 = self.roi_bottom
            x2 = int((y2 - b) / m)

            cv2.line(img, (x1,y1),(x2,y2), (0,0,255), 3)

    def get_points(self, lines):
        points = []
        for line in lines:
            for x1,y1,x2,y2 in line:
                points.append((x1,y1))
                points.append((x2,y2))

        return points


filepath = "test_videos/challenge.mp4"
filename = os.path.basename(filepath)
print(filename)

cap = cv2.VideoCapture(filepath)

if cap.isOpened() == False:
    print("Error opening video")

frm_rt = cap.get(cv2.CAP_PROP_FPS)      # Get frame rate
frm_dur = int(1000 / frm_rt)

ret, frame = cap.read()
size = frame.shape

cv2.namedWindow(filename)
cv2.moveWindow(filename, 100,100)

gui = LaneFindVideoGUI(size)
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
