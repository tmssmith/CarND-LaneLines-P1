import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os
import argparse
import PolygonDrawer as pd
import HoughGUI
import extrapolate_lines as el

vertices = np.array([[100,530],[440,320],[520,320],[900,530]],np.int32)


class LaneFinderGUI:
    def __init__(self, image, kernel=1, low=1, ratio=1):
        self.image = image
        self.kernel = kernel
        self.low = low
        self.ratio = ratio
        self.high = self.low * self.ratio / 10
        self.alpha = 0.5
        self.beta = 1 - self.alpha
        self.gamma = 0

        cv2.namedWindow("Edges")
        cv2.moveWindow("Edges", 100,100)
        cv2.createTrackbar('Kernel Size', 'Edges', self.kernel, 10, self.onChange_kernel)
        cv2.createTrackbar('Low Threshold', 'Edges', self.low, 255, self.onChange_low)
        cv2.createTrackbar('Canny Ratio', 'Edges', self.ratio, 50, self.onChange_ratio)
        cv2.createTrackbar('Overlay Ratio', 'Edges', int(self.alpha*100), 100, self.onChange_alpha)
        self.render()
        cv2.waitKey(0)
        cv2.destroyWindow('Edges')

    def onChange_kernel(self, value):
        value = max(1, value)
        self.kernel = (value * 2) - 1
        self.render()

    def onChange_low(self, value):
        value = max(1, value)
        self.low = value
        self.high = self.low * self.ratio
        self.render()

    def onChange_ratio(self, value):
        value = max(1, value)
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
        cv2.putText(self.dst, "Kernel size: {:.0f}".format(self.kernel), (10,20), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(self.dst, "Low Threshold: {:.0f}".format(self.low), (10,40), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.putText(self.dst, "High Threshold: {:.0f}".format(self.high), (10,60), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.imshow("Edges", self.dst)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help = "image filename for lane finding")
    args = parser.parse_args()
    filename = os.path.basename(args.image)
    [imagename, fileext] = filename.split(".", 1)
    img_orig = cv2.imread(args.image)
    img_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)

### RUN CANNY EDGE DETECTION ON IMAGE WITH GUI FOR PARAMETERISATION
    canny = LaneFinderGUI(img_gray)
    print("Kernel size: {:.0f}\nLow Threshold: {:.0f}\nHigh Threshold: {:.0f}" \
        .format(canny.kernel, canny.low, canny.high))
### Save canny edges image to test_images_output folder
    canny_save_img = canny.img_canny.copy()
    cv2.putText(canny_save_img, "Image: {}".format(imagename), (10,20), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(canny_save_img, "Kernel size: {:.0f}".format(canny.kernel), (10,40), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(canny_save_img, "Low Threshold: {:.0f}".format(canny.low), (10,60), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(canny_save_img, "High Threshold: {:.0f}".format(canny.high), (10,80), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.imwrite("test_images_output/" + imagename + "_canny.jpg", canny_save_img)

### DETERMINE REGION OF INTEREST WITH GUI FOR USER SELECTION
    roi = pd.PolygonDrawer(canny.img_canny)
    roi.drawPoly()
    img_roi = roi.region_of_interest()
### Save region of interest of canny image to test_images_output folder
    roi_save_img = img_roi.copy()
    cv2.putText(roi_save_img, "Image: {}".format(imagename), (10,20), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.imwrite("test_images_output/" + imagename + "_roi.jpg", roi_save_img)
    cv2.namedWindow("Masked Image")
    cv2.moveWindow("Masked Image", 100, 100)
    cv2.imshow("Masked Image", img_roi)
    cv2.waitKey()
    cv2.destroyWindow("Masked Image")

    print(roi.vertices)

### RUN HOUGH TRANSFORM ON ROI WITH GUI FOR PARAMETERISATION
    hough = HoughGUI.GUI(img_orig, img_roi)
    print("Rho: {}\nTheta: {:.0f} x pi / 180\nThreshold: {}\nMin line length: {}\nMax line gap: {}" \
        .format(hough.rho, hough.theta * (180 / np.pi), hough.threshold, hough.min_line_len, hough.max_line_gap))
### Save hough lines image to test_images_output folder
    hough_save_img = hough.img_lines.copy()
    cv2.putText(hough_save_img, "Image: {}".format(imagename), (10,20), \
        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(hough_save_img, "Lines detected: {:.0f}".format(len(hough.lines)), (10,40), \
        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(hough_save_img, "Rho: {}".format(hough.rho), (10,60), \
        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(hough_save_img, "Theta: {:.0f} x pi / 180".format(hough.theta * (180/np.pi)), (10,80), \
        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(hough_save_img, "Threshold: {}".format(hough.threshold), (10,100), \
        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(hough_save_img, "Min line length: {}".format(hough.min_line_len), (10,120), \
        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.putText(hough_save_img, "Max line gap: {}".format(hough.max_line_gap), (10,140), \
        cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.imwrite("test_images_output/" + imagename + "_hough.jpg", hough_save_img)
    hough_lines = np.array(hough.lines)
    np.save("test_images_output/" + imagename + "_houghlines.npy", hough_lines)

### EXTRAPOLATE HOUGH LINES TO RUN LINE ALONG FULL LENGTH OF VISIBLE LANES
    separated_lanes = el.separate_lines(hough_lines)
    left_lane = el.extrapolate_line(separated_lanes[0],img_orig.shape[0])
    right_lane = el.extrapolate_line(separated_lanes[1],img_orig.shape[0])
    img_laneline = np.zeros((img_orig.shape[0], img_orig.shape[1], 3), dtype=np.uint8)
    cv2.line(img_laneline, (left_lane[0], left_lane[1]),(left_lane[2], left_lane[3]), (0,0,255),2)
    cv2.line(img_laneline, (right_lane[0], right_lane[1]),(right_lane[2], right_lane[3]), (0,255,0),2)

### Merge lane line with original image

    img_final = cv2.addWeighted(img_orig, 0.5,  img_laneline, 0.5, 0)
    cv2.namedWindow("Final")
    cv2.imshow("Final", img_final)
    cv2.waitKey(0)
    final_save_img = img_final.copy()
    cv2.putText(final_save_img, "Image: {}".format(imagename), (10,20), cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
    cv2.imwrite("test_images_output/" + imagename + "_final.jpg", final_save_img)


if __name__ == "__main__":
    main()
