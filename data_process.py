import random
import matplotlib.pyplot as plt
import cv2 as cv2

h = 192
w = 288

def split_test_from_train():
    test_list = []
    test_mask_list = []
    train_list = []
    train_mask_list = []
      
    for i in range(0,998):
        rand_num_1 ,rand_num_2 = random.sample(range(1,11),2)
        path = './Data Augmentation/Augmented_data/src/'+str(i)+'_'+str(rand_num_1)+'.bmp'
        test_mask_path = './Data Augmentation/Augmented_data/mask/'+str(i)+'_'+str(rand_num_1)+'.bmp'

        if i%2==0:
            path2 = './Data Augmentation/Augmented_data/src/'+str(i)+'_'+str(rand_num_2)+'.bmp'
            test_mask_path2 = './Data Augmentation/Augmented_data/mask/'+str(i)+'_'+str(rand_num_2)+'.bmp'
            test_mask_list.append(test_mask_path2)
            test_list.append(path2)

        test_mask_list.append(test_mask_path)
        test_list.append(path)  

        for j in range(1,11)  :  
            path = './Data Augmentation/Augmented_data/src/'+str(i)+'_'+str(j)+'.bmp'
            if path not in test_list:
                train_list.append(path)
                train_mask_list.append('./Data Augmentation/Augmented_data/mask/'+str(i)+'_'+str(j)+'.bmp') 
       

    return train_list,train_mask_list,test_list,test_mask_list

