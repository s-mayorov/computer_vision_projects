import cv2
import matplotlib.pyplot as plt
import numpy as np

coins = cv2.imread("images/coins.jpeg")
gray_coins = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)
blurred_coins = cv2.GaussianBlur(gray_coins, (5,5), 1)

farms = cv2.imread("images/satellite-fields.jpg")
gray_farms = cv2.cvtColor(farms, cv2.COLOR_BGR2GRAY)
blurred_farms = cv2.GaussianBlur(gray_farms, (5,5), 1)

# coins image
coin_circles = cv2.HoughCircles(blurred_coins, cv2.HOUGH_GRADIENT, 1, 
                           minDist=150,
                           param1=50,
                           param2=25,
                           minRadius=28,
                           maxRadius=110)
# round farms image
farm_circles = cv2.HoughCircles(blurred_farms, cv2.HOUGH_GRADIENT, 1, 
                           minDist=150,
                           param1=68,
                           param2=48,
                           minRadius=80,
                           maxRadius=150)

coin_circles_im = np.copy(coins)
# convert circles into expected type
coin_circles = np.uint16(np.around(coin_circles))
# draw each one
for i in coin_circles[0,:]:
    # draw the outer circle
    cv2.circle(coin_circles_im,(i[0],i[1]),i[2],(255,0,0),5)
    # draw the center of the circle
    cv2.circle(coin_circles_im,(i[0],i[1]),2,(0,0,255),3)

farms_circles_im = np.copy(farms)
# convert circles into expected type
farm_circles = np.uint16(np.around(farm_circles))
# draw each one
for i in farm_circles[0,:]:
    # draw the outer circle
    cv2.circle(farms_circles_im,(i[0],i[1]),i[2],(255,0,0),5)
    # draw the center of the circle
    cv2.circle(farms_circles_im,(i[0],i[1]),2,(0,0,255),3)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.set_title("Coins")
ax1.imshow(coin_circles_im)
ax2.set_title("Farms")
ax2.imshow(farms_circles_im)
plt.show()

