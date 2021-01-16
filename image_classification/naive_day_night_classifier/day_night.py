import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os
import cv2

IMAGE_SIZE = (800, 600)

def load_dataset(im_dir):
    img_list = []
    img_type = ['day', 'night']

    for tp in img_type:
        for image_path in glob.glob(os.path.join(im_dir, tp, "*")):
            image = mpimg.imread(image_path)
            image = cv2.resize(image, IMAGE_SIZE)
            binary_type = 0 if tp == "night" else 1
            img_list.append((image, binary_type))

    return img_list

def average_channel(image, channel=2):
    # brightness is default channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    brightness = hsv[:,:,channel]
    average = np.sum(brightness) / (IMAGE_SIZE[0]*IMAGE_SIZE[1])
    return average

def classify(image):
    brightness = average_channel(image, 2)
    saturation = average_channel(image, 1)
    # day if average brightness > 100
    if brightness > 100 and saturation < 90:
        return 1
    return 0

def accuracy(correct, total):
    return round(correct/total*100, 2)

ds = load_dataset("./day_night_images/")
correct_count = 0
incorrect_list = []
for image, label in ds:
    predicted = classify(image)
    if predicted == label:
        correct_count+=1
    else:
        incorrect_list.append(image)

total = len(ds)
incorrect_count = len(incorrect_list)

accuracy = accuracy(correct_count, len(ds))

print("Correctly classified {} of {} images, {} mistakes. Total accuracy is {}".format(
    correct_count, total, incorrect_count, accuracy
))

