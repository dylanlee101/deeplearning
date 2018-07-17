# -*- coding: utf-8 -*-
# @Time    : 2018/7/10 11:00
# @Author  : Dylan
# @File    : predict.py
# @Email   : wenyili@buaa.edu.cn

import os
from utils import load_data
from sklearn.metrics import accuracy_score
from keras.models import load_model
import numpy as np
import pandas as pd
import cv2
from keras.preprocessing.image import img_to_array
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# data = "../data/train2/"
# data_list = os.listdir(data)
# params = {"norm_size":224}
# probability = []
# filename = []
# X,Y,name = load_data(params)
# for n in name:
#     fn = n.split("/")[-1]
#     filename.append(fn)
# model1 = load_model("../models/best_model.h5")
# probability = np.array(model1.predict(X))
# pd.DataFrame(pd.concat(filename,probability), columns=["filename","probability"]).to_csv(
#     "../result/result.csv",
#     index=False
# )
params = {
    "norm_size":224
}
def predict():
    weights_path = "../models/best_model2.h5"
#     weights_path = "gap_ResNet50.h5"
    data_path = "/data2/dockspace_lwy/cloth_data/data/test/"
    data_list = os.listdir(data_path)
    data = []
    for file in data_list:
        file_name = data_path + file
        image = cv2.imread(file_name)
        image = cv2.resize(image, (224, 224))
        image = img_to_array(image)
        data.append(image)
    data = np.array(data, dtype="float") / 255.0

    
    model = load_model(weights_path)
    pred = model.predict(data)
    
    filename = pd.DataFrame(data_list,columns=["filename"])
    probability = pd.DataFrame(pred,columns=["probability"])

    pd.DataFrame(pd.concat([filename,probability],axis=1),columns=["filename","probability"]).to_csv(
        "../result/baseline_res.csv",
        index=False
    )
if __name__ == "__main__":
    predict()
# score = accuracy_score(Y,pred)
# def to_ca(x):
#     if x >0.9:
#         x = 1
#     else:
#         x = 0
#     return x
# predicts = np.array(list((map(to_ca,pred[:,0]))))
# score1 = accuracy_score(Y,predicts.reshape(predicts.shape[0],1))
# print("best model on val score :",score1)

# model2 = load_model("../models/resnet_model_101.h5")
# pred2 = np.array(model2.predict(X))
# predicts2 = np.array(list((map(to_ca,pred2[:,0]))))
# score2 = accuracy_score(Y,predicts2.reshape(predicts2.shape[0],1))
# print("best model on test score :",score2)

