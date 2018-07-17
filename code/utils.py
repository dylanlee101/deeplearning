# -*- coding: utf-8 -*-
# @Time    : 2018/7/9 16:58
# @Author  : Dylan
# @File    : utils.py
# @Email   : wenyili@buaa.edu.cn
import os
import cv2

import numpy as np
import resnet
from mymodel import build
# import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator


def rotate(image, angle, center=None, scale=1.0): #1
    (h, w) = image.shape[:2] #2
    if center is None: #3
        center = (w // 2, h // 2) #4

    M = cv2.getRotationMatrix2D(center, angle, scale) #5

    rotated = cv2.warpAffine(image, M, (w, h)) #6
    return rotated 
def equal_hist(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst = cv2.equalizeHist(gray)
    dst = cv2.cvtColor(dst,cv2.COLOR_GRAY2BGR)
    return dst
def h(image):
    size = (image.shape[1],image.shape[0])
    iLR  = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    h = image.shape[1]
    w = image.shape[0]
    for i in range(h):
        for j in range(w):
            # iUD[h-1-i,j] = image[i,j]
            iLR[i,w-1-j] = image[i,j]
            # iAcross[h-1-i,w-1-j] = image[i,j]
    return iLR
def load_data(params):
    path = '/data2/dockspace_lwy/cloth_data/data/train2/'
    path2 = '/data2/dockspace_lwy/cloth_data/data/train3/'
    data_folder_list1 = list(map(lambda x:path + x,os.listdir(path)))
    data_folder_list2 = list(map(lambda x:path2 + x,os.listdir(path2)))
    data_folder_list = data_folder_list1 + data_folder_list2
    images = []
    labels = []
    file_name = []
    for files in data_folder_list:
        print("loading data :",files)
        files_list = os.listdir(files)
        for file in files_list:
            name = files + "/" + file
#             print(name.split("/")[-2])
            if name[-3:] == "jpg":
                if name.split("/")[-2] == 'zc':
                    label = 0
                else:
                    label = 1
                labels.append(label)
                file_name.append(name)
#                 print(name)
                image = cv2.imread(name)
                image1 = cv2.resize(image,(params["norm_size"],params["norm_size"]))
                image = img_to_array(image1)
                images.append(image)
                image2 = equal_hist(image1)
                image2 = img_to_array(image2)
                label2 = label
                images.append(image2)
                labels.append(label2)
                image3 = h(image1)
                image3 = img_to_array(image3)
                label3 = label
                images.append(image3)
                labels.append(label3) 
                
#                 if not name.split("/")[-2] == 'zc':
#                     img1 = rotate(image1,90)
#                     img1 = img_to_array(img1)
#                     label1 = 1
#                     images.append(img1)
#                     labels.append(label1)
#                     file_name.append(name)
                    
#                     img2 = rotate(image1,180)
#                     img2 = img_to_array(img2)
#                     label2 = 1
#                     images.append(img2)
#                     labels.append(label2)
#                     file_name.append(name)
                    
#                     img3 = rotate(image1,270)
#                     img3 = img_to_array(img3)
#                     label3 = 1
#                     images.append(img3)
#                     labels.append(label3)
#                     file_name.append(name)
#                 if name.split("/")[-2] == 'zc':
#                     img1 = rotate(image1,90)
#                     img1 = img_to_array(img1)
#                     label1 = 0
#                     images.append(img1)
#                     labels.append(label1)
#                     file_name.append(name)
                    
#                     img2 = rotate(image1,180)
#                     img2 = img_to_array(img2)
#                     label2 = 0
#                     images.append(img2)
#                     labels.append(label2)
#                     file_name.append(name)
                    
#                     img3 = rotate(image1,270)
#                     img3 = img_to_array(img3)
#                     label3 = 0
#                     images.append(img3)
#                     labels.append(label3)
#                     file_name.append(name)
                    
                
            else:
                continue
#     images = np.array(images,dtype="float") / 255.0
#     labels = np.array(labels)
#     labels = labels.reshape(len(labels),1)
#     print(labels.shape)
    images = np.array(images, dtype="float") / 255.0
    labels = np.array(labels)
    labels = labels.reshape(labels.shape[0],1)
    return images,labels,file_name

def model_train(x_train,y_train,x_test,y_test,params):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc',patience=10,verbose=1)
    early_stop = EarlyStopping(monitor='val_acc',patience=20,verbose=1)
    csv_log = CSVLogger(params["log"])
    checkpoint = ModelCheckpoint(params["save_model"],monitor='val_acc',verbose=1,save_best_only=True)
#     model = resnet.ResnetBuilder.build_resnet_101((3, params["norm_size"], params["norm_size"]), 1)
    
    model = resnet.ResnetBuilder.build_resnet_101((3,params["norm_size"], params["norm_size"]), 1)
    
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy"])
    if not params["aug"]:
        print("Not using data augmentation.")
        H = model.fit(x_train, y_train,
                  batch_size=params["batch_size"],
                  nb_epoch=params["epoch"],
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=[lr_reduce, early_stop,csv_log,checkpoint])
    else:
        print("Using real-time data augmentation.")
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        datagen.fit(x_train)

        H = model.fit_generator(datagen.flow(x_train, y_train, batch_size=params["batch_size"]),
                                steps_per_epoch=x_train.shape[0] // params["batch_size"],
                                validation_data=(x_test, y_test),class_weight = {0:0.9,1:0.1},
                                epochs=params["epoch"], verbose=1, max_q_size=100,
                                callbacks=[lr_reduce, early_stop,csv_log,checkpoint])
    print("model training is finished...")
    model.save("../models/mymodel2.h5")

#     plt.style.use("ggplot")
#     plt.figure()
#     N = params["epoch"]
#     plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
#     plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
#     plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
#     plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
#     plt.title("Training Loss and Accuracy on Audio Test classifier")
#     plt.xlabel("Epoch ")
#     plt.ylabel("Loss/Accuracy")
#     plt.legend(loc="lower left")
#     plt.savefig(params["plot"])




