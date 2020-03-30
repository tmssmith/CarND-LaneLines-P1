import cv2
import numpy as np

class PolygonDrawer:
    def __init__(self, image):
        self.image = image
        self.mouse_pos = (0, 0)
        self.vertices = []
        self.done = False

    def on_mouse(self, event, x, y , buttons, user_param):
        if event == cv2.EVENT_MOUSEMOVE:
            # We want to be able to draw the line-in-progress, so update current mouse position
            self.mouse_pos = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            # Left click means adding a point at current position to the list of points
            print("Adding point #%d with position(%d,%d)" % (len(self.vertices), x, y))
            self.vertices.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click to reset polygon
            self.vertices = []

    def drawPoly(self):
        cv2.namedWindow("Polygon Mask")
        cv2.setMouseCallback("Polygon Mask", self.on_mouse)

        while not self.done:
            canvas = np.copy(self.image)
            if len(self.vertices) > 0:
                cv2.polylines(canvas, np.array([self.vertices]), False, (255,255,255), 2)
                cv2.line(canvas, self.vertices[-1], self.mouse_pos, (125,125,125), 2)
            cv2.imshow("Polygon Mask", canvas)
            if cv2.waitKey(50) == 27:
                self.done = True
            if len(self.vertices) == 4:
                cv2.fillPoly(canvas, np.array([self.vertices]), (255,255,255))
                dst = cv2.addWeighted(self.image, 0.3, canvas, 0.7, 0)
                cv2.imshow("Polygon Mask", dst)
                self.done = True

        cv2.waitKey()
        cv2.destroyAllWindows()

    def region_of_interest(self):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        `vertices` should be a numpy array of integer points.
        """
        #defining a blank mask to start with
        mask = np.zeros_like(self.image)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(self.image.shape) > 2:
            channel_count = self.image.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        #filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, np.array([self.vertices]), ignore_mask_color)

        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(self.image, mask)
        return masked_image

if __name__ == "__main__":
    app = PolygonDrawer(np.zeros((480,480), np.uint8))
    app.drawPoly()
    print("Polygon = %s" % app.vertices)
