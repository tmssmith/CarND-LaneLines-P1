import numpy as np
import numpy.polynomial.polynomial as poly
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
import os


class LaneFindVideoGUI:
    def __init__(self):

        self.flag = False
        self.kernel = 5
        self.low = 50
        self.high = 150

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

    def find_lanes(self, img):
        self.img_orig = img
        self.img_shape = self.img_orig.shape
        self.height = self.img_shape[0]
        self.width = self.img_shape[1]

        self.trackbars()
        self.canny_edges()
        self.apply_mask()
        self.hough_transform()
        self.fit_lane_lines()
        self.render()
        return self.img_final

    def canny_edges(self):
        self.img_gray = cv2.cvtColor(self.img_orig, cv2.COLOR_BGR2GRAY)
        self.img_gaus = cv2.GaussianBlur(self.img_gray, (self.kernel, self.kernel), 0)
        self.img_canny = cv2.Canny(self.img_gaus, self.low, self.high)

    def apply_mask(self):
        self.img_mask = self.mask(self.img_canny)
        self.img_roi = cv2.bitwise_and(self.img_canny, self.img_mask)
        self.img_mask = cv2.cvtColor(self.img_mask, cv2.COLOR_GRAY2BGR)

    def hough_transform(self):
        self.hough_lines = cv2.HoughLinesP(self.img_roi, self.rho, self.theta, self.threshold, np.array ([ ]), self.min_line_len, self.max_line_gap)
        self.img_hough = canvas(self.img_shape)
        if self.hough_lines is not None:
            self.img_hough = draw_lines(self.img_hough, self.hough_lines)
        else:
            print("No lines found")

    def fit_lane_lines(self):
        self.img_best_fit = canvas(self.img_shape)
        if self.hough_lines is not None:
            self.img_best_fit = self.best_fit_lines(self.img_best_fit, self.hough_lines)

    def render(self):
        self.image_list = (self.img_orig, self.img_gaus, self.img_canny, self.img_hough)
        self.base_image = self.image_list[self.image_selector]

        if len(self.base_image.shape) == 2:
            self.base_image = cv2.cvtColor(self.base_image, cv2.COLOR_GRAY2BGR)

        self.img_masked = cv2.addWeighted(self.base_image, self.alpha, self.img_mask, self.beta, self.gamma)
        self.img_final = cv2.addWeighted(self.img_masked, self.alpha, self.img_best_fit, self.beta, self.gamma)

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
            [self.roi_bottomright, self.roi_bottom]])

        self.img_mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(self.img_mask, [self.vertices], ignore_mask_color)

        return self.img_mask

    def best_fit_lines(self, img, lines):
        (left_lines, right_lines) = separate_lines(lines)

        left_points = get_points(left_lines)
        right_points = get_points(right_lines)
        left_best_fit = self.fit_line(left_points)
        right_best_fit = self.fit_line(right_points)
        img = draw_lines(img, [left_best_fit])
        img = draw_lines(img, [right_best_fit])

        # img = cv2.polylines(img, np.int32([left_best_fit]), False, (0,0,255), 2)
        # img = cv2.polylines(img, np.int32([right_best_fit]), False, (0,0,255), 2)
        return img

    def fit_line(self, points):
        x = points[0]
        y = points[1]

        if x and y:
            b, m = poly.polyfit(x, y, 1)
            y1 = self.roi_top
            x1 = int((y1 - b) / m)
            y2 = self.roi_bottom
            x2 = int((y2 - b) / m)
            return np.array([[x1,y1,x2,y2]], np.int32)
        else:
            return np.empty((1,4), np.int32)

    # def fit_line(self, points):
    #     x = points[0]
    #     y = points[1]
    #
    #     if x and y:
    #
    #         coefs = poly.polyfit(x, y, 2)
    #         x = np.arange(0, self.width, 5)
    #         print(x.shape)
    #         y = poly.polyval(x, coefs)
    #         print(y.shape)
    #         pts = np.column_stack((x,y))
    #         print(pts.shape)
    #         return pts
    #     else:
    #         return np.empty((1,2), np.int32)
    #         # Find x1, x2 at y1 and y2
    #         # Plot y for x in range x1 and x2
    #         # Use CV Polylines
    #
    #
        #     y1 = self.roi_top
        #     x1 = int((y1 - b) / m)
        #     y2 = self.roi_bottom
        #     x2 = int((y2 - b) / m)
        #     return np.array([[x1,y1,x2,y2]], np.int32)
        # else:
        #     return np.empty((1,4), np.int32)

    def trackbars(self):
        if self.flag != True:
            self.roi_top = 10
            self.roi_bottom = self.height - 10
            self.roi_topleft = 10
            self.roi_topright = self.width - 10
            self.roi_bottomleft = 10
            self.roi_bottomright = self.width-10

            cv2.namedWindow("Parameters", flags = 0)
            cv2.moveWindow("Parameters",700,150)
            cv2.createTrackbar('Kernel Size', 'Parameters', self.kernel, 10, self.onChange_kernel)
            cv2.createTrackbar('Low Threshold', 'Parameters', self.low, 255, self.onChange_low)
            cv2.createTrackbar('High Threshold', 'Parameters', self.high, 255, self.onChange_high)
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
            self.flag = True
    def onChange_kernel(self, value):
        value = max(1, value)
        self.kernel = 2 * value - 1
    def onChange_low(self, value):
        value = max(1, value)
        if value >= self.high:
            cv2.setTrackbarPos('Low Threshold', 'Parameters', self.high)
            self.low = self.high
        else:
            self.low = value
    def onChange_high(self, value):
        value = max(1, value)
        if value <= self.low:
            cv2.setTrackbarPos('High Threshold', 'Parameters', self.low)
            self.high = self.low
        else:
            self.high = value
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
        if value >= self.roi_bottom:
            cv2.setTrackbarPos('ROI top', 'Parameters', self.roi_bottom)
            self.roi_top = self.roi_bottom
        else:
            self.roi_top = value
    def onChange_roibottom(self, value):
        if value <= self.roi_top:
            cv2.setTrackbarPos('ROI bottom', 'Parameters', self.roi_top)
            self.roi_bottom = self.roi_top
        else:
            self.roi_bottom = value
    def onChange_roitopleft(self, value):
        if value >= self.roi_topright:
            cv2.setTrackbarPos('ROI top left', 'Parameters', self.roi_topright)
            self.roi_topleft = self.roi_topright
        else:
            self.roi_topleft = value
    def onChange_roitopright(self, value):
        if value <= self.roi_topleft:
            cv2.setTrackbarPos('ROI top right', 'Parameters', self.roi_topleft)
            self.roi_topright = self.roi_topleft
        else:
            self.roi_topright = value
    def onChange_roibottomleft(self, value):
        if value >= self.roi_bottomright:
            cv2.setTrackbarPos('ROI bottom left', 'Parameters', self.roi_bottomright)
            self.roi_bottomleft = self.roi_bottomright
        else:
            self.roi_bottomleft = value
    def onChange_roibottomright(self, value):
        if value <= self.roi_bottomleft:
            cv2.setTrackbarPos('ROI bottom right', 'Parameters', self.roi_bottomleft)
            self.roi_bottomright = self.roi_bottomleft
        else:
            self.roi_bottomright = value

def canvas(shape):
    canvas = np.zeros(shape, np.uint8)
    return canvas

def separate_lines(lines):
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

def get_points(lines):
    x = []
    y = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            x.extend([x1, x2])
            y.extend([y1, y2])

    return (x,y)

def draw_lines(img, lines, color=[0, 0, 255], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help = "file for lane finding")
    args = parser.parse_args()
    filename = os.path.basename(args.file)
    extension = os.path.splitext(args.file)[1]
    if extension == '.jpg':
        [imagename, fileext] = filename.split(".", 1)
        img = cv2.imread(args.file)
        cv2.namedWindow(filename)
        cv2.moveWindow(filename, 100,100)
        gui = LaneFindVideoGUI()

        while True:
            img_lanes = gui.find_lanes(img)
            cv2.imshow(filename, img_lanes)
            if cv2.waitKey(40) & 0xFF == ord('q'):
                break

    elif extension == '.mp4':
        cap = cv2.VideoCapture(args.file)

        if cap.isOpened() == False:
            print("Error opening video")

        frm_rt = cap.get(cv2.CAP_PROP_FPS)      # Get frame rate
        frm_dur = int(1000 / frm_rt)

        cv2.namedWindow(filename)
        cv2.moveWindow(filename, 100,100)

        gui = LaneFindVideoGUI()

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                frame_lanes = gui.find_lanes(frame)
                cv2.imshow(filename, frame_lanes)
                if cv2.waitKey(frm_dur) & 0xFF == ord('q'):
                    break
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        cap.release()

    else:
        "Filetype must be *.jpg or *.mp4"

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
