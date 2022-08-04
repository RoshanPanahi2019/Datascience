import matplotlib.pyplot as plt
from os import listdir
import pandas as pd
import numpy as np
from keras.preprocessing import sequence
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

path="/media/ms/OS/Users/rosha/Desktop/DS_USB_07302022/Data/TimeSeries/Train_Data/Data_Input/11039_Numeric.csv"

# Read time series data
df=pd.read_csv(path)

# Pre-processing
df=df.rename(columns={"Index":"Time","Root Cause for Slip / Push":"Delay_Cause"})
train=df.to_numpy()
label = 0
m,n=df.shape
target_shape=2
# Train & test on a single scope -- Add other scopes -- Add other features. 

model=Sequential()

model.add(LSTM(300, input_shape=(m, n), return_sequences=True))

#model.add(Dropout(0.2))
model.add(LSTM(300))
#model.add(Dropout(0.2))
model.add(Dense(target_shape, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
print("===========")

# fitting the model . 
model.fit(train, label, epochs=1, batch_size=30)
