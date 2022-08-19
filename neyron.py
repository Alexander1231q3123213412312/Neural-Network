import tensorflow.python.keras.models
import numpy
import keras.saving.saved_model_experimental
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.utils import np_utils
import os
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='4'

train=False

if train:
    
    model=None 

    Russian_tanks_ready=os.listdir('Russian_tanks_ready/')
    Other_nation_tanks_ready=os.listdir('Other_nation_tanks_ready/')

    Russian_tanks_data=[]
    Other_nation_tanks_data=[]

    for image in Russian_tanks_ready:
        im=Image.open("Russian_tanks_ready/"+"/"+image)
        im_arr=np.array(im)
        Russian_tanks_data.append((im_arr,"Russian"))

    for image in Other_nation_tanks_ready:
        im=Image.open('Other_nation_tanks_ready/'+"/"+image)
        im_arr=np.array(im)
        Other_nation_tanks_data.append((im_arr,"Other"))

    x_train=Russian_tanks_data+Other_nation_tanks_data
    random.shuffle(x_train)

    y_train=[[1,0] if _[1]=='Russian' else [0,1] for _ in x_train]
    x_train=[_[0] for _ in x_train]

    x_train=np.array(x_train)
    y_train=np.array(y_train)

    model=Sequential()

    model.add(Conv2D(256, (3, 3), input_shape=(256, 256, 1), padding="same"))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(256,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(256,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Conv2D(256,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.2))

    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(256, kernel_constraint=maxnorm(3)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(2))
    model.add(Activation("softmax"))

    optimizer='SGD'

    model.compile(loss="categorical_crossentropy", optimizer=optimizer,metrics=['accuracy'])
    
    model.fit(x_train, y_train,epochs=15, batch_size=16)

    model.save("ai.h5")

else:
    model=tensorflow.keras.models.load_model("ai.h5")

    Russian_tanks_test_ready=os.listdir('Russian_tanks_test_ready/')
    Other_nation_tanks_test_ready=os.listdir('Other_nation_tanks_test_ready/')

    Russian_tanks_test_ready_data=[]
    Other_nation_tanks_test_ready_data=[]

    for image in Russian_tanks_test_ready:
        im=Image.open('Russian_tanks_test_ready/'+"/"+image)
        im_arr=np.array(im)
        Russian_tanks_test_ready_data.append((im_arr,"Russian"))

    for image in Other_nation_tanks_test_ready:
        im=Image.open('Other_nation_tanks_test_ready/'+"/"+image)
        im_arr=np.array(im)
        Other_nation_tanks_test_ready_data.append((im_arr,"Other"))

    x_test=Russian_tanks_test_ready_data+Other_nation_tanks_test_ready_data
    random.shuffle(x_test)

    y_test=[[1,0] if _[1]=='Russian' else [0,1] for _ in x_test]
    x_test=[_[0] for _ in x_test]

    x_test=np.array(x_test)
    y_test=np.array(y_test)

    error, acc=model.evaluate(x_test, y_test)
    print(format(100*acc))

    answers=model.predict(x_test)
    answers=['Russian' if max(_)==_[0] else 'Other' for _ in answers]

    w=3
    h=3
    
    fig, axes=plt.subplots(w, h)
    fig.set_size_inches(10, 10)
    fig.subplots_adjust(0.1, 0.1)

    cur=0

    for y in range(h):
        for x in range(w):
            axes[y, x].get_xaxis().set_visible(False)
            axes[y, x].get_yaxis().set_visible(False)

            axes[y, x].imshow(x_test[y*w+x], cmap='gray')
            axes[y, x].set_title(answers[y*w+x])
    plt.show()










