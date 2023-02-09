import tensorflow as tf
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential,Model,initializers,layers,Input
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split

class ShuffleNet_v2(Y_Dense_Block):
    def __init__(self, num_classes, input_shape=(224,224,3)):
        self.num_classes=num_classes
        self.input_shape=input_shape

    def Basic_Unit(self,x,f,groups):
        x1,x2=layers.Lambda(lambda x: tf.split(x,num_or_size_splits=2,axis=3))(x)

        x2=layers.Conv2D(filters=f/4,kernel_size=1,strides=1,padding='same')(x2)
        x2=layers.BatchNormalization()(x2)
        x2=layers.ReLU()(x2)
        x2=layers.DepthwiseConv2D(kernel_size=3,strides=1,padding='same')(x2)
        x2=layers.BatchNormalization()(x2)
        x2=layers.Conv2D(filters=f,kernel_size=1,strides=1,padding='same')(x2)
        x2=layers.BatchNormalization()(x2)
        x2=layers.ReLU()(x2)

        x3=layers.Concatenate()([x1,x2])
        x3=layers.Lambda(lambda x:self.Channel_Shuffle(x,groups=2) )(x3)
        return x3

    def Downsampling_Unit(self,x,f):
        x1=layers.DepthwiseConv2D(kernel_size=3,strides=2,padding='same')(x)
        x1=layers.BatchNormalization()(x1)
        x1=layers.Conv2D(filters=f,kernel_size=1,strides=1,padding='same')(x1)
        x1=layers.BatchNormalization()(x1)
        x1=layers.ReLU()(x1)

        x2=layers.Conv2D(filters=f/4,kernel_size=1,strides=1,padding='same')(x)
        x2=layers.BatchNormalization()(x2)
        x2=layers.ReLU()(x2)
        x2=layers.DepthwiseConv2D(kernel_size=3,strides=2,padding='same')(x2)
        x2=layers.BatchNormalization()(x2)
        x2=layers.Conv2D(filters=f,kernel_size=1,strides=1,padding='same')(x2)
        x2=layers.BatchNormalization()(x2)
        x2=layers.ReLU()(x2)

        x3=layers.Concatenate()([x1,x2])
        x3=layers.Lambda(lambda x:self.Channel_Shuffle(x,4))(x3)
        return x3

    def forward(self):
        input=Input(shape=self.input_shape)
        x=layers.Conv2D(filters=24,kernel_size=3,strides=2,padding='same')(input)
        x=layers.MaxPool2D(pool_size=3,strides=2,padding='same')(x)

        x=self.Downsampling_Unit(x,116)

        x=self.Basic_Unit(x,116,4)
        x=self.Basic_Unit(x,116,4)
        x=self.Basic_Unit(x,116,4)

        x=self.Downsampling_Unit(x,232)
 
        x=self.Basic_Unit(x,232,4)
        x=self.Basic_Unit(x,232,4)
        x=self.Basic_Unit(x,232,4)
        x=self.Basic_Unit(x,232,4)
        x=self.Basic_Unit(x,232,4)
        x=self.Basic_Unit(x,232,4)
        x=self.Basic_Unit(x,232,4)
     
        x=self.Downsampling_Unit(x,464)

        x=self.Basic_Unit(x,464,4)
        x=self.Basic_Unit(x,464,4)
        x=self.Basic_Unit(x,464,4)

        x=layers.Conv2D(filters=1024,kernel_size=1,strides=1,padding='same')(x)
   
        x=layers.GlobalAveragePooling2D()(x)
        x=layers.Dense(units=self.num_classes,activation='softmax')(x)

        model = tf.keras.models.Model(input,x)

        return model