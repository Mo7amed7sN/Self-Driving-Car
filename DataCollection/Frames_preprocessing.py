import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from skimage.measure import label, regionprops, regionprops_table
import math
import numpy as np

def return_rotated(red , angle):

    (h, w) = red.shape[:2]
    # calculate the center of the image
    center = (w / 2, h / 2)
    scale = 1.0
    # Perform the counter clockwise rotation holding at the center
    # 30 degrees
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(red, M, (h, w))
    return rotated


def return_with_y_x(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # apply threshold
    thresh = threshold_otsu(img)
    bw = closing(img > thresh, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label(cleared)
    image_label_overlay = label2rgb(label_image, image=img)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image_label_overlay)
    for region in regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 100:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            y0 , x0 = region.centroid
            ax.add_patch(rect)

    return x0,y0


imaga = cv2.imread('InkedInkedsteeringwheel2_LI.jpg')
dim = (640,640)
imaga = cv2.resize(imaga , dim,interpolation = cv2.INTER_AREA)
cv2.imshow("steering wheel",imaga)

# Red color
hsv_frame = cv2.cvtColor(imaga, cv2.COLOR_BGR2HSV)
low_red = np.array([161, 155, 84])
high_red = np.array([179, 255, 255])
red_mask = cv2.inRange(hsv_frame, low_red, high_red)
red = cv2.bitwise_and(imaga, imaga, mask=red_mask)
cv2.imshow("Red", red)

rotation_output = return_rotated(red,-70)
cv2.imshow("rotated img", rotation_output)

# to calc radious
rotation_180_output = return_rotated(red,-180)
cv2.imshow("rotated img 2", rotation_180_output)

v11 , v12  = return_with_y_x(red)
v21 , v22  = return_with_y_x(rotation_output)
v31 , v32  = return_with_y_x(rotation_180_output)


#check if to right or to left
bol = False
''' x1 < x2 '''
if(v11 < v21):
    bol = True




ECU_distance = math.sqrt(((v11-v21)**2)+((v12-v22)**2))
rad = math.sqrt(((v11-v31)**2)+((v12-v32)**2)) / 2
cos_theta = (2*(rad**2) - (ECU_distance**2)) / (2 *(rad**2))
print(cos_theta)

theta_in_radian = np.arccos(cos_theta)
print(theta_in_radian)

theta_in_degree = (theta_in_radian * 180 / math.pi)

if (bol):
    theta_in_degree *= -1
print(theta_in_degree)

key = cv2.waitKey(0)
cv2.destroyAllWindows()

