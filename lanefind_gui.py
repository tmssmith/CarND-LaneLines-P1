import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import argparse
import PolygonDrawer as pd

vertices = np.array([[100,530],[440,320],[520,320],[900,530]],np.int32)


class LaneFinderGUI:
    def __init__(self, image, kernel=1, low=1, ratio=1):
        self.image = image
        self.kernel = kernel
        self.low = low
        self.ratio = ratio
        self.high = self.low * self.ratio
        self.alpha = 0.5
        self.beta = 1 - self.alpha
        self.gamma = 0

        cv2.namedWindow("Edges")
        cv2.createTrackbar('Kernel Size', 'Edges', self.kernel, 10, self.onChange_kernel)
        cv2.createTrackbar('Low Threshold', 'Edges', self.low, 255, self.onChange_low)
        cv2.createTrackbar('Canny Ratio', 'Edges', self.ratio, 50, self.onChange_ratio)
        cv2.createTrackbar('Overlay Ratio', 'Edges', int(self.alpha*100), 100, self.onChange_alpha)
        self.render()
        cv2.waitKey(0)
        cv2.destroyWindow('Edges')

    def onChange_kernel(self, value):
        self.kernel = (value * 2) - 1
        self.render()

    def onChange_low(self, value):
        self.low = value
        self.high = self.low * self.ratio
        self.render()

    def onChange_ratio(self, value):
        self.ratio = value / 10
        self.high = self.low * self.ratio
        self.render()

    def onChange_alpha(self,value):
        self.alpha = value / 100
        self.beta = 1 - self.alpha
        self.render()

    def render(self):
        self.img_gaus = cv2.GaussianBlur(self.image, (self.kernel, self.kernel), 0)
        self.img_canny = cv2.Canny(self.img_gaus, self.low, self.high)
        self.dst = cv2.addWeighted(self.image, self.alpha, self.img_canny, self.beta, self.gamma)
        cv2.imshow("Edges", self.dst)


class HoughGUI:
    def __init__(self, image, roi, rho=1, theta=np.pi/180, threshold=5, min_line_len=10, max_line_gap=5):
        self.image = image
        self.roi = roi
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_len = min_line_len
        self.max_line_gap = max_line_gap
        self.alpha = 0.5
        self.beta = 1 - self.alpha

        cv2.namedWindow("Hough Transform")
        cv2.createTrackbar('Rho', "Hough Transform", self.rho, 10, self.onChange_rho)
        # cv2.createTrackbar('Theta', "Hough Transform", self.theta, 30, self.onChange_theta)

        self.render()
        cv2.waitKey(0)
        cv2.destroyWindow("Hough Transform")

    def onChange_rho(self,value):
        self.rho = value
        self.render()

    def draw_lines(self, img, lines, color=[0, 0, 255], thickness=2):
        """
        NOTE: this is the function you might want to use as a starting point once you want to
        average/extrapolate the line segments you detect to map out the full
        extent of the lane (going from the result shown in raw-lines-example.mp4
        to that shown in P1_example.mp4).

        Think about things like separating line segments by their
        slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
        line vs. the right line.  Then, you can average the position of each of
        the lines and extrapolate to the top and bottom of the lane.

        This function draws `lines` with `color` and `thickness`.
        Lines are drawn on the image inplace (mutates the image).
        If you want to make the lines semi-transparent, think about combining
        this function with the weighted_img() function below
        """
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)

    def render(self):
        lines = cv2.HoughLinesP(self.roi, self.rho, self.theta, self.threshold, self.min_line_len, self.max_line_gap)
        line_img = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        self.dst = cv2.addWeighted(self.image, self.alpha, line_img, self.beta, 0)
        cv2.imshow("Hough Transform", self.dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help = "image filename for lane finding")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    lanes = LaneFinderGUI(img_gray)

    roi = pd.PolygonDrawer(lanes.img_canny)
    roi.drawPoly()

    img_mask = roi.region_of_interest()
    cv2.namedWindow("Masked Image")
    cv2.imshow("Masked Image", img_mask)
    cv2.waitKey()

    print(roi.vertices)
    hough = HoughGUI(img, img_mask)


if __name__ == "__main__":
    main()
