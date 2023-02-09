import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential,Model,initializers,layers,Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential,Model,initializers,layers,Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from Y_Dense_Block import *

class Y_Block(Y_Dense_Block):
    def __init__(self, num_classes, input_shape=(224,224,3)):
        self.num_classes=num_classes
        self.input_shape=input_shape

    def Y_Module(self,x,filter_size):
        x1,x2=layers.Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=3))(x)

        x1=layers.Conv2D(filters=filter_size/4,kernel_size=1,strides=1,padding='same',use_bias=False)(x1)
        x1=layers.BatchNormalization()(x1)
        x1=layers.ReLU(max_value=6)(x1)
        x1=layers.DepthwiseConv2D(kernel_size=3,strides=1,padding='same',use_bias=False)(x1)
        x1=layers.BatchNormalization()(x1)
        x1=layers.Conv2D(filters=filter_size,kernel_size=1,strides=1,padding='same',use_bias=False)(x1)
        x1=layers.BatchNormalization()(x1)
        x1=layers.ReLU(max_value=6)(x1)
        
        x3=layers.Concatenate()([x1,x2])
        x3=layers.Lambda(lambda x:self.Channel_Shuffle(x,groups=2))(x3)
        return x3


    def Y_Transition(self,x,filter_size):
        x2=layers.Conv2D(filters=filter_size/4,kernel_size=1,strides=1,padding='same',use_bias=False)(x)
        x2=layers.BatchNormalization()(x2)
        x2=layers.ReLU(max_value=6)(x2)
        x2=layers.DepthwiseConv2D(kernel_size=3,strides=2,padding='same',use_bias=False)(x2)
        x2=layers.BatchNormalization()(x2)
        x2=layers.Conv2D(filters=filter_size,kernel_size=1,strides=1,padding='same',use_bias=False)(x2)
        x2=layers.BatchNormalization()(x2)
        x2=layers.ReLU(max_value=6)(x2)

        x1=layers.AveragePooling2D(pool_size=2,strides=2,padding='same')(x)
        x1=layers.Conv2D(filters=filter_size,kernel_size=1,strides=1,padding='same',use_bias=False)(x1)

        x3=layers.Concatenate()([x1,x2])
        x3=layers.Lambda(lambda x:self.Channel_Shuffle(x,groups=2))(x3)

        return x3

    def Y_Block(self, x, filter_size,repeat):
        x = self.Y_Module(x,filter_size)
        for i in range(repeat):
            x_temp=self.Y_Module(x=x, filter_size=filter_size)
            x=x_temp
        return x

    def forward(self):
        input=Input(shape=self.input_shape)
        x=layers.Conv2D(filters=24,kernel_size=3,strides=2,padding='same')(input)
        x=layers.MaxPool2D(pool_size=3,strides=2,padding='same')(x)
        
        x=self.Y_Transition(x,116)

        x=self.Y_Block(x,116,3)

        x=self.Y_Transition(x,232)
    
        x=self.Y_Block(x,232,7)
   
        x=self.Y_Transition(x,464)
    
        x=self.Y_Block(x,464,3)
    
        x=self.Classifier(x)    
        model = tf.keras.models.Model(input,x)
        
        return model
