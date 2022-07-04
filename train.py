import numpy as np
import pandas as pd

from modules import conv_nn as cnn

#Load in training and testing datasets
train = pd.read_csv(r'C:\Users\SullyIsCool\Downloads\train.csv\train.csv').to_numpy()
test = pd.read_csv(r'C:\Users\SullyIsCool\Downloads\test.csv\test.csv').to_numpy()

#Since each pixel is a value between 0 : 255, divide the pixels by 255 to be a value between 0 : 1 
test = test / 255
x = train[:, 1:]
y = train[:, 0:1]
x = x / 255 #Don't divide the labels (y), they are not pixels