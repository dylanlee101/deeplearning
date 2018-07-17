# -*- coding: utf-8 -*-
# @Time    : 2018/7/9 19:50
# @Author  : Dylan
# @File    : main.py.py
# @Email   : wenyili@buaa.edu.cn
import argparse
import os
import numpy as np
from utils import load_data,model_train
from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
np.random.seed(42)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_train",         default=True,               action = 'store_true')
    parser.add_argument("--predict",              default=False,              action = 'store_true')
    parser.add_argument('--save_model',           default='../models/best_model2.h5')
    parser.add_argument("--epoch",                default=100,                  type = int)
    parser.add_argument("--lr",                    default=0.001,               type = float)
    parser.add_argument("--batch_size",           default=30,                  type = int)
    parser.add_argument("--norm_size",            default=224,                  type = int)
    parser.add_argument("--log",                   default="../analyse/log.csv")
    parser.add_argument("--plot",                  default="../analyse/plot.png")
    parser.add_argument("--data_augmentation",    default=True)
    args = parser.parse_args()

    params = {
        "model_train":args.model_train,
        "save_model":args.save_model,
        "epoch":args.epoch,
        "lr":args.lr,
        "batch_size":args.batch_size,
        "norm_size":args.norm_size,
        "plot":args.plot,
        "aug":args.data_augmentation,
        "log":args.log
    }
    x,y,name = load_data(params)
    print(len(x),len(y))
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4)
    model_train(x_train,y_train,x_test,y_test,params)
