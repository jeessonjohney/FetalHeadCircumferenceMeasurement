from keras.models import load_model
import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt
import pandas as pd
import imghdr
import calculate_size
import os
import sys
import random
from math import sqrt

model = load_model('model/head_circum_model.h5')

def load_file_names():
    file_names = pd.read_csv('./Data Augmentation/raw_data/test_pixel_size.csv')
    rand_id = random.randint(0,334)
    file_name = './Data Augmentation/raw_data/test_set/'+str(file_names['filename'][rand_id])
    pixel_size = calculate_size.test_image_pixel_size(file_names['filename'][rand_id])
    predicted_mask = predict(file_name)
    x,y,r1,r2,angle = calculate_size.circumference(predicted_mask,pixel_size)
    circum = 3.14*(3*(r1+r2)-sqrt((3*r1+r2)*(r1+3*r2)))
    display_predicted_image(predicted_mask,file_names['filename'][rand_id],file_name,circum)

def read_image(file_name):
    x_shape = np.zeros((192,192,1))
    resolution = cv2.imread(file_name)
    height = len(resolution)
    width = len(resolution[0])
    image = cv2.resize(cv2.imread(file_name,cv2.IMREAD_GRAYSCALE),dsize=(192,192))
    x_shape[:,:,:] = np.reshape(image,(192,192,1))
    x_shape = x_shape.astype(np.float)
    x_shape = np.multiply(x_shape, 1.0/255)
    return x_shape,width,height

def predict(image):
    x,h,w = read_image(image)
    pred = model.predict(np.expand_dims(x, axis=0))[0]
    pred = pred *255
    pred = pred.astype(np.uint8)
    pred = np.concatenate([pred, pred, pred], axis=2)
    pred = cv2.resize(pred,(h,w))
    return pred

def write_predicted_image(file_name,predicted_image):
    file_name = './results/'+str(file_name)
    cv2.imwrite(file_name,predicted_image)

def display_predicted_image(predicted_mask,file_name,file_dir,circum,diameter):
    bpd = "BPD: "+str(diameter[0])+str(' mm')
    ofd = "OFD: "+str(diameter[1])+str(' mm')
    predicted_mask = predicted_mask*[0,0,255]
    image = cv2.imread(file_dir)
    image = image*[1,1,1]
    image = cv2.addWeighted(image,1,predicted_mask,1,0)
    image = cv2.putText(image,str(circum),(590,480),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2,cv2.LINE_AA)
    image = cv2.putText(image,str(bpd),(30,480),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2,cv2.LINE_AA)
    image = cv2.putText(image,str(ofd),(30,520),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0), 2,cv2.LINE_AA)
    image = cv2.putText(image,str(file_name),(30,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255), 2,cv2.LINE_AA)
    plt.imshow(image)
    plt.show()

def custom_image_measurement(image,pixel_size):
    predicted_mask = predict(image)
    x,y,r1,r2,angle = calculate_size.circumference(predicted_mask,pixel_size)
    bpd = r2*2
    ofd = r1*2
    circum = 3.14*(3*(r1+r2)-sqrt((3*r1+r2)*(r1+3*r2)))
    display_predicted_image(predicted_mask,image.split('/')[-1],image,str(round(circum,2))+' mm',(str(round(bpd,2)),str(round(ofd,2))))


if __name__ == '__main__':
    mast_head = "\n\nAutomated fetal head circumference Measurement \nWritten By: Jeesson Johney\n\n "
    usage = "Usage:\npredict.py opt1 opt2 \nopt1 : -t for importing random image from test set\n     : -c for custom Image and directory must be specified\n\nopt2 : Location of image file if '-c' is opted\n\nExample : \nFor Predicting Random Test image : predict.py -t\n\nFor Predicting Custom image : predict.py -c image_location pixel_size\n\nSupported Image Formats : \n JPG,JPEG,PNG,BMP \n\n"

    print(mast_head)
    if len(sys.argv) > 1:
        
        file_format = ['png','jpg','jpeg','bmp']
        
        
        if sys.argv[1]=='t':
            load_file_names()            
        elif sys.argv[1] == 'c':            
            if len(sys.argv) > 3 :                
                if os.path.isfile(sys.argv[2]) and (imghdr.what(sys.argv[2]) in file_format) :                                       
                    custom_image_measurement(sys.argv[2],float(sys.argv[3]))                    
            else:
                print("File location not specified / file not supported / Pixel size not specified\n")
                print(usage)
        else:
            print(usage)
    else:
        print(usage)


# write_predicted_image('007_HC.png', predict('./Data Augmentation/raw_data/training_set/007_HC.png'))
# new_mask = cv2.cvtColor(p,cv2.COLOR_RGB2GRAY)
# none ,new_mask = cv2.threshold(new_mask,220,255,cv2.THRESH_BINARY)
# edge = cv2.Canny(new_mask,30, 200)

# edge = cv2.cvtColor(edge,cv2.COLOR_GRAY2RGB)
# edge = edge*[0,0,255]
# image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
# image = image*[1,1,1]
# image = cv2.addWeighted(image,1,edge,1,0)