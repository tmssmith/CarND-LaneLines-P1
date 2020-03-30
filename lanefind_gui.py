import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import argparse

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
        img_gaus = cv2.GaussianBlur(self.image, (self.kernel, self.kernel), 0)
        img_canny = cv2.Canny(img_gaus, self.low, self.high)
        dst = cv2.addWeighted(self.image, self.alpha, img_canny, self.beta, self.gamma)
        cv2.imshow("Edges", dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help = "image filename for lane finding")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    img_gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)

    lanes = LaneFinderGUI(img_gray)

    cv2.destroyAllWindows()

    print("Kernel: {} \nLow: {} \nHigh: {}".format(lanes.kernel, lanes.low, lanes.high))


if __name__ == "__main__":
    main()
