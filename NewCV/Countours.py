import sys
import numpy as np
import cv2 as cv

hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)

if __name__ == '__main__':
    fn = 'pic2.jpg'
    img = cv.imread(fn)

    hsv = cv.cvtColor( img, cv.COLOR_BGR2HSV )
    thresh = cv.inRange( hsv, hsv_min, hsv_max )
    _, contours0, hierarchy = cv.findContours( thresh.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    index = 0
    layer = 0

    def update():
        vis = img.copy()
        cv.drawContours( vis, contours0, index, (255,0,0), 2, cv.LINE_AA, hierarchy, layer )
        cv.imshow('contours', vis)

    def update_index(v):
        global index
        index = v-1
        update()

    def update_layer(v):
        global layer
        layer = v
        update()

    update_index(0)
    update_layer(0)
    cv.createTrackbar( "contour", "contours", 0, 7, update_index )
    cv.createTrackbar( "layers", "contours", 0, 7, update_layer )

    cv.waitKey()
    cv.destroyAllWindows()