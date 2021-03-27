
import data_process
import matplotlib.pyplot as plt
import cv2 as cv2
import numpy as np
from pandas import DataFrame
from tqdm import tqdm
import u_net 
from keras.callbacks import EarlyStopping, ModelCheckpoint

height = 192
width = 192


def reshape(x,y):
    x_shape = np.zeros((len(x),width,height,1))
    y_shape = np.zeros((len(y),width,height,1))

    for images in tqdm(range(0,len(x)-1)):
        image = cv2.resize(cv2.imread(x[images][0],cv2.IMREAD_GRAYSCALE),dsize=(192,192))
        mask = cv2.resize(cv2.imread(y[images][0],cv2.IMREAD_GRAYSCALE),dsize=(192,192))
        x_shape[images,:,:,:] = np.reshape(image,(width,height,1))
        y_shape[images,:,:,:] = np.reshape(mask,(width,height,1))
    
    # fig = plt.figure(figsize = (8,8))

    # fig.add_subplot(2,2,1)
    # plt.imshow(x_shape[6].squeeze())

    # fig.add_subplot(2,2,2)
    # plt.imshow(y_shape[6].squeeze())

    # plt.show()

    x_shape = x_shape.astype(np.float)
    y_shape = y_shape.astype(np.float)

    x_shape = np.multiply(x_shape, 1.0/255)
    y_shape = np.multiply(y_shape, 1.0/255)
    
    return x_shape,y_shape
    


def prepare_train_data():

    train_x ,train_y ,test_x ,test_y = data_process.split_test_from_train() # raw image matrix 1497 test and 8483 train
    pd_train_x = DataFrame(train_x).iloc[:,:].values
    pd_train_y = DataFrame(train_y).iloc[:,:].values

    permutation = np.arange(len(train_x))
    np.random.shuffle(permutation)
    train_x = pd_train_x[permutation]
    train_y = pd_train_y[permutation]
    
    return reshape(train_x,train_y)



if __name__ == '__main__':

    train_x,train_y = prepare_train_data()
    model = u_net.build_unet((192,192,1))
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])
    model.summary() 
    model_path = "model/head_circum_model.h5"
    checkpoint = ModelCheckpoint(model_path,
                                monitor="val_loss",
                                mode="min",
                                save_best_only = True,
                                verbose=1)

    earlystop = EarlyStopping(monitor = 'val_loss', 
                            min_delta = 0, 
                            patience = 5,
                            verbose = 1,
                            restore_best_weights = True)

    results = model.fit(train_x, train_y, validation_split=0.15,
                        batch_size=8, epochs=10, 
                        callbacks=[earlystop, checkpoint])