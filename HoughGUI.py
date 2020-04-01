import numpy as np
import cv2
import os
import argparse

class GUI:
    def __init__(self, image, roi,):
        self.image = image
        self.roi = roi
        self.rho = 1
        self.theta_bar = 1
        self.theta = self.theta_bar * (np.pi / 180)
        self.threshold = 1
        self.min_line_len = 1
        self.max_line_gap = 1
        self.alpha = 0.3
        self.beta = 1 - self.alpha

        cv2.namedWindow("Hough Transform")
        cv2.moveWindow("Hough Transform", 100,100)
        cv2.createTrackbar('Rho', "Hough Transform", self.rho, 10, self.onChange_rho)
        cv2.createTrackbar('Theta', "Hough Transform", self.theta_bar, 10, self.onChange_theta)
        cv2.createTrackbar('Threshold', "Hough Transform", self.threshold, 100, self.onChange_threshold)
        cv2.createTrackbar('Min line length', "Hough Transform", self.min_line_len, 100, self.onChange_minlinelen)
        cv2.createTrackbar('Max line gap', "Hough Transform", self.max_line_gap, 100, self.onChange_maxlinegap)

        self.render()
        cv2.waitKey(0)
        cv2.destroyWindow("Hough Transform")

    def onChange_rho(self,value):
        value = max(1, value)
        self.rho = value
        self.render()

    def onChange_theta(self,value):
        value = max(1, value)
        self.theta = value * (np.pi / 180)
        self.render()

    def onChange_threshold(self,value):
        value = max(1, value)
        self.threshold = value
        self.render()

    def onChange_minlinelen(self,value):
        value = max(1, value)
        self.min_line_len = value
        self.render()

    def onChange_maxlinegap(self,value):
        value = max(1, value)
        self.max_line_gap = value
        self.render()

    def render(self):
        self.lines = cv2.HoughLinesP(self.roi, self.rho, self.theta, self.threshold, np.array ([ ]), self.min_line_len, self.max_line_gap)
        self.img_lines = np.zeros((self.image.shape[0], self.image.shape[1], 3), dtype=np.uint8)
        self.draw_lines(self.img_lines, self.lines)
        self.img_merged = cv2.addWeighted(self.image, self.alpha, self.img_lines, self.beta, 0)
        
        cv2.putText(self.img_merged, "Lines detected: {:.0f}".format(len(self.lines)), (10,20), \
            cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(self.img_merged, "Rho: {}".format(self.rho), (10,40), \
            cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(self.img_merged, "Theta: {:.0f} x pi / 180".format(self.theta * (180/np.pi)), (10,60), \
            cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(self.img_merged, "Threshold: {}".format(self.threshold), (10,80), \
            cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(self.img_merged, "Min line length: {}".format(self.min_line_len), (10,100), \
            cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(self.img_merged, "Max line gap: {}".format(self.max_line_gap), (10,120), \
            cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.imshow("Hough Transform", self.img_merged)

    def draw_lines(self, img, lines, color=[0, 0, 255], thickness=2):
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help = "image filename for hough lines")
    args = parser.parse_args()
    filename = os.path.basename(args.image)
    [imagename, fileext] = filename.split(".", 1)
    image = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2GRAY)
    canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    hough = GUI(canvas, image)

if __name__ == "__main__":
    main()
