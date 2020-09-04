import numpy as np
import pandas as pd
#import keras
import glob
#from keras.preprocessing.image import array_to_img,img_to_array,load_img
#import matplotlib.pyplot as plt

#from keras.utils import np_utils
from sklearn.model_selection import train_test_split

#import keras.backend as K
#K.clear_session()
#from keras.models import Model,Sequential
#from keras.layers import Input,BatchNormalization,Conv2D,MaxPooling2D,Flatten,Dense,Dropout
#from keras.optimizers import SGD, Adam
import tensorflow as tf

root_dir = "./capture"
categories = ['bolt_gin','bolt_kuro','desk']
nb_classes = len(categories)
#image_size = 224
image_size = 64
X = []
Y = []
for idx, cat in enumerate(categories):
    files = glob.glob(root_dir + "/" + cat + "/*")
    for i, f in enumerate(files):
        img = tf.keras.preprocessing.image.load_img(f, target_size=(image_size,image_size))
        data = tf.keras.preprocessing.image.img_to_array(img)
        X.append(data)
        Y.append(idx)
X = np.array(X)
Y = np.array(Y)

X = X.astype('float32')/255
Y = tf.keras.utils.to_categorical(Y)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=0,stratify=Y)
#print(x_train.shape,x_test.shape,y_train.shape,y_test.shape)

#学習
model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=x_train.shape[1:]),
tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Dropout(0.25),

tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
tf.keras.layers.Dropout(0.25),

tf.keras.layers.Flatten(),
tf.keras.layers.Dense(256,activation='relu'),
tf.keras.layers.Dropout(0.25),
tf.keras.layers.Dense(nb_classes,activation='softmax')])

#sgd = SGD(lr=0.001,decay=1e-4,momentum=0.9,nesterov=True)
#adam = Adam(lr=1e-4,decay=1e-4)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

batch_size=8
epochs=100
history=model.fit(x_train,y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=1,
                  validation_data=(x_test,y_test))

result=pd.DataFrame(history.history)
#plt=result[['acc','val_acc']].plot()
print(result[['acc','val_acc']])
#model.save('./model/model.h5',include_optimizer=False)
model.save('./model/t-model.h5')
#plt.plot(list(range(epochs)),result[['acc']])
#plt.plot(list(range(epochs)),result[['val_acc']])
#plt.show()

