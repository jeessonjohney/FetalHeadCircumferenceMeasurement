import pandas as pd
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def get_ellipse(im):    
    contour_list = []
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(imgray,127,255,0)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cont in contours:
        area = cv2.contourArea(cont)
        contour_list.append(area)
    contour = sorted(contour_list,reverse=False)
    ellipse = cv2.fitEllipse(contours[-1])
    return ellipse

def circumference(predicted_image,pixel_size):

    # image = cv2.imread(str('./predicted_mask/'+predicted_image))
    ellipse = get_ellipse(predicted_image)
    center_x_mm,center_y_mm = ellipse[0]
    axes_a ,axes_b = ellipse[1]
    angle = ellipse[2]
    
    if axes_a < axes_b:
        axes_a, axes_b = axes_b, axes_a

    axes_a,axes_b = axes_a/2,axes_b/2

    if angle < 90:
        angle+=90
    else:
        angle-=90

    angle = np.deg2rad(angle)

    return center_x_mm*pixel_size, center_y_mm*pixel_size, axes_a*pixel_size, axes_b*pixel_size, angle

def test_image_pixel_size(file_name):
    file_list = pd.read_csv('./Data Augmentation/raw_data/test_pixel_size.csv')
    for i in range(0,335):
        if file_list['filename'][i] == file_name:
            pixel_size = file_list['pixel size(mm)'][i]
            break
    return pixel_size

def create_submission():
    
    file_name = pd.read_csv('./Data Augmentation/raw_data/test_pixel_size.csv')
    sub = "filename,center_x_mm,center_y_mm,semi_axes_a_mm,semi_axes_b_mm,angle_rad \n"
    for i in range(0,335):
        c_x,c_y,ax_a,ax_b,angle = circumference(file_name['filename'][i],file_name['pixel size(mm)'][i])
        sub+=str(file_name['filename'][i])+','+str(c_x)+','+str(c_y)+','+str(ax_a)+','+str(ax_b)+','+str(angle)+'\n'
    with open('submission.csv','w') as csv_file:
        csv_file.write(sub)