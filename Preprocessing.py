
import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential,Model,initializers,layers,Input
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
from keras_flops import get_flops
from EffShuff_Dense_Block import*
from EffShuff_Block import *
from ShuffleNet_v2 import*

class Preprocessing():
    def __init__(self,dir,batch_size,validation_split,num_classes,seed,subset,
                    shuffle=True,img_size=(224, 224)):
        self.dir=dir
        self.batch_size=batch_size
        self.validation_split=validation_split
        # self.epochs=epochs ### Why need epochs?
        self.num_classes=num_classes
        self.subset=subset
        self.shuffle=shuffle
        self.seed=seed
        self.img_size=img_size


    def Load_Data(self):
        '''
        dir                 : [str] directory path
        validation_split    : [float] or None
        subset              : [boolean] Train / Validation
        shuffle             : [boolean]
        seed                : [int] or None
        batch_size          : [int] mini batch size
        img_size            : [tuple] batch image size
        '''
        ds=tf.keras.preprocessing.image_dataset_from_directory(
        self.dir,
        labels="inferred",
        label_mode="categorical",
        class_names=None,
        color_mode="rgb",
        batch_size=self.batch_size,
        image_size=self.img_size,
        validation_split=self.validation_split, 
        subset=self.subset,
        shuffle=self.shuffle,
        seed=self.seed,
        interpolation="gaussian",
        follow_links=False,
        crop_to_aspect_ratio=False,)

        return ds

    def Normalization(self,data):    
        '''
        data    : [string] dataset
        '''
        normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255.)
        
        ds=data.map(lambda data,y: (normalization_layer(data), y))

        return ds

    def Get_Preprosessed_Dataset(self):
    
        ds=self.Load_Data()
        ds=self.Normalization(ds)

        return ds


class Get_Model(Model):
    def __init__(self,model_name,num_classes,activation='softmax'):
        self.model_name=model_name
        self.num_classes=num_classes
        self.activation=activation

    def Get_Train_Model(self):
        '''
        pretrained model    : [keras model] pretrained model
        num_classes         : [int] number of classes
        activation          : [str] classifier activation function of model
        '''
        model_list=['densenet201','densenet201_scratch','resnet50','resnet50_scratch','mobilnetv2','mobilnetv2_scratch','efficientnet','efficientnet_scratch','EffShuff_Block', 'EffShuff_Dense_Block','shuffleNetv2']
        if self.model_name in model_list[0:8]:
            if self.model_name==model_list[0]:
                model=tf.keras.applications.DenseNet201(input_shape=(224,224,3), include_top=False, weights='imagenet')
            elif self.model_name==model_list[1]:
                model=tf.keras.applications.DenseNet201(input_shape=(224,224,3), include_top=False, weights=None)
            elif self.model_name==model_list[2]:
                model=tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights='imagenet')
            elif self.model_name==model_list[3]:
                model=tf.keras.applications.ResNet50(input_shape=(224,224,3), include_top=False, weights=None)
            elif self.model_name==model_list[4]:
                model=tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
            elif self.model_name==model_list[5]:
                model=tf.keras.applications.MobileNetV2(input_shape=(224,224,3), include_top=False, weights=None)
            elif self.model_name==model_list[6]:
                model=tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights='imagenet')
            elif self.model_name==model_list[7]:
                model=tf.keras.applications.EfficientNetB0(input_shape=(224,224,3), include_top=False, weights=None)
            model=Sequential([
                            model,
                            layers.Flatten(),
                            layers.Dense(units=self.num_classes, activation=self.activation)
                            ])
        elif self.model_name in model_list[8:11]:
            if self.model_name==model_list[8]:
                model=EffShuff_Block(num_classes=self.num_classes).forward()

            if self.model_name==model_list[9]:
                model=EffShuff_Dense_Block(num_classes=self.num_classes).forward()
                
            if self.model_name==model_list[10]:
                model=ShuffleNet_v2(num_classes=self.num_classes).forward()

        else:
            print('error: no model')
        
        return model





    # def Get_My_Model(self,model):
    #     model_list=['densenet201','densenet201_scratch','resnet50','resnet50_scratch','mobilnetv2','mobilnetv2_scratch','efficientnet','efficientnet_scratch','Y_Block', 'Y_Dense_Block','shuffleNetv2']
    #     if self.model_name in model_list[8:10]:
    #         if self.model_name==model_list[8]:
    #             model=Y_Block(num_classes=self.num_classes).forward()

    #         if self.model_name==model_list[9]:
    #             model=Y_Dense_Block(num_classes=self.num_classes).forward()
                
    #         if self.model_name==model_list[10]:
    #             model=ShuffleNet_v2(num_classes=self.num_classes).forward()

    #     else:
    #         print('error: no model')

    #     return model