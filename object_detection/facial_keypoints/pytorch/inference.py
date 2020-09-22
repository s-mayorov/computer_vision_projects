import cv2
import torch
import numpy as np

from models import Net


net = Net()
net.load_state_dict(torch.load('v2_model.pt'))
net.eval()


def preprocess_roi(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    normalized = gray/255.
    resized = cv2.resize(normalized, (224,224))
    return resized


def add_sunglasses(keypoints, image):
    # top-left location for sunglasses to go
    # 17 = edge of left eyebrow
    x = int(keypoints[17, 0])
    y = int(keypoints[17, 1])

    # height and width of sunglasses
    # h = length of nose
    h = int(abs(keypoints[27,1] - keypoints[34,1]))
    # w = left to right eyebrow edges
    w = int(abs(keypoints[17,0] - keypoints[26,0]))

    # read in sunglasses
    sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
    # resize sunglasses
    new_sunglasses =  cv2.resize(sunglasses, (w, h), interpolation = cv2.INTER_CUBIC)

    # get region of interest on the face to change
    roi_color = image[y:y+h,x:x+w]

    # find all non-transparent pts
    ind = np.argwhere(new_sunglasses[:,:,3] > 0)

    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    for j in range(3):
        roi_color[ind[:,0],ind[:,1],j] = new_sunglasses[ind[:,0],ind[:,1],j]    
    # set the area of the image to the changed region with sunglasses
    image[y:y+h,x:x+w] = roi_color
    return image

def add_moustaches(keypoints, image):
    # 33 = bottom of nose 
    x = int(keypoints[33, 0])
    y = int((keypoints[33,1] + keypoints[51,1])/2)

    # height and width of moustache
    # h = length of mouth
    h = int(abs(keypoints[51,1] - keypoints[57,1]))
    # w = width of mouth
    w = int(abs(keypoints[48,0] - keypoints[54,0]))

    h += int(h*.3)
    w += int(w*.3)

    # read in sunglasses
    moustache = cv2.imread('images/moustache.png', cv2.IMREAD_UNCHANGED)
    # resize sunglasses
    new_moustache =  cv2.resize(moustache, (w, h), interpolation = cv2.INTER_CUBIC)

    # get region of interest on the face to change
    roi_color = image[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)]

    # find all non-transparent pts
    ind = np.argwhere(new_moustache[:,:,3] > 0)

    # for each non-transparent point, replace the original image pixel with that of the new_sunglasses
    for k in range(3):
        roi_color[ind[:,0],ind[:,1],k] = new_moustache[ind[:,0],ind[:,1],k]    
    # set the area of the image to the changed region with sunglasses
    image[int(y-h/2):int(y+h/2),int(x-w/2):int(x+w/2)] = roi_color
    return image


def full_image_process(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_cascade = cv2.CascadeClassifier('detector_architectures/haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, 1.2, 2)

    image_copy = np.copy(image)

    # haar cascade find very tight face region, it's better to add some padding
    padding = 30

    for (x,y,w,h) in faces:
        roi = image_copy[y-padding:y+h+padding, x-padding:x+w+padding]
        prep_roi = preprocess_roi(roi)    
        
        # add 1 dim as batch dim and 1 dim as color channel
        batch = torch.tensor(prep_roi, dtype=torch.float32).reshape([1,1,prep_roi.shape[0], prep_roi.shape[1]])
        
        preds = net(batch)
        preds = preds.data.reshape(-1, 2).numpy()*50.+100

        modified = add_sunglasses(preds, roi)
        modified = add_moustaches(preds, modified)
        resized_back = cv2.resize(modified, (roi.shape[0], roi.shape[1]))
        image[y-padding:y+h+padding, x-padding:x+w+padding] = resized_back

    return image


if __name__ == "__main__":
    image = cv2.imread('images/jeremy-howard.jpg')
    # it's in BGR, but it's fine to check facial keypoints
    cv2.imshow("result", full_image_process(image))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
