import cv2
import matplotlib.pyplot as plt

# read image as grayscale
image = cv2.imread("simple-shapes.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# threshold image to binary
ret, thresh = cv2.threshold(gray, 127, 255, 1)

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    polygon_type = ""
    # get perimeter of closed contour, needed for approximation accuracy
    perimeter = cv2.arcLength(contour, True)
    # get approximated polygon with 1% of perimeter accuracy
    polygon = cv2.approxPolyDP(contour, 0.01*perimeter, True)
    vertices_cnt = len(polygon)

    if vertices_cnt == 3:
        polygon_type = "triangle"
    elif vertices_cnt == 4:
        polygon_type = "rectangle"
    elif vertices_cnt == 5:
        polygon_type = "pentagon"
    elif vertices_cnt == 6:
        polygon_type = "hexagon"
    elif vertices_cnt == 7:
        polygon_type = "heptagon"
    elif vertices_cnt == 8:
        polygon_type = "octagon"
    elif vertices_cnt > 15:
        polygon_type = "circle"
   

    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/      py_contour_features/py_contour_features.html?highlight=approxpolydp
    # get the center of object
    M = cv2.moments(contour)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    cv2.drawContours(image, [contour], 0, (0, 125, 0), 4)
    cv2.putText(image, polygon_type, (cx-50, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

cv2.imwrite("result.png", image)
cv2.imshow('Identifying Shapes', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
