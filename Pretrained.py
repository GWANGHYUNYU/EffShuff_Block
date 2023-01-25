import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential,Model,initializers,layers,Input
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import matplotlib.pyplot as plt
from keras_flops import get_flops
from Y_Dense_Block import*
from Y_Block import *
from Preprocessing import *
from Traning_Testing import *
from ShuffleNet_v2 import*

if __name__ == "__main__":

    # Hyperparameters for Function
    lr_rate=1e-3
    #tf.keras.optimizers.schedules.ExponentialDecay(1e-3, 10000, 0.97, staircase=False, name=None)
    #opt=tf.keras.optimizers.Adam(learning_rate=lr_rate)
    opt=tf.optimizers.Adam(learning_rate=lr_rate)
    #tf.keras.optimizers.Adam(learning_rate=lr_rate)
    loss='categorical_crossentropy'
    epochs=20
    batch_size=64
    num_classes=4
    metrics=['acc']
    img_shape=(224,224,3)

    # Directory Path

    # 나비 데이터셋 경로
    dir_butterfly=r'D:\GH\DeepLearning\dataset\butterfly_moth\train'
    dir_test_butterfly=r'D:\GH\DeepLearning\dataset\butterfly_moth\test'

    # 성별 데이터셋 경로
    dir_gender=r'D:\GH\DeepLearning\dataset\age_gender\gender'

    # 나이 데이터셋 경로
    dir_age=r'D:\GH\DeepLearning\dataset\age_gender\age'

    dir=dir_butterfly

    model_list=['densenet201','densenet201_scratch','resnet50','resnet50_scratch','mobilnetv2','mobilnetv2_scratch','efficientnet','efficientnet_scratch','Y_Block', 'Y_Dense_Block','shuffleNetv2']
    
    model_current_name ='efficientnet'

    model=Get_Model(model_name=model_current_name,num_classes=num_classes).Get_Train_Model()

    print(model_current_name)

    
    # 가중치 저장 경로
    dir_weights_save=r'D:\GitHub_repo\yBlock\save_weights\y_block'
    dir_weights_save=os.path.join(dir_weights_save, model_current_name,'_gender')  
    print("save dir:  ",dir_weights_save)

    # 데이터 전처리
    training_ds=Preprocessing(dir=dir,batch_size=batch_size,validation_split=0.1,num_classes=num_classes,subset='training',
                              shuffle=True,seed=123,img_size=(224, 224)).Get_Preprosessed_Dataset()
                        
    validation_ds=Preprocessing(dir=dir,batch_size=batch_size,validation_split=0.1,num_classes=num_classes,subset='validation',
                                shuffle=True,seed=123,img_size=(224, 224)).Get_Preprosessed_Dataset()
     
    # Train
    history=Train_Model(model_name=model,training_ds=training_ds,validation_ds=validation_ds,num_classes=num_classes,
                        opt=opt,loss=loss,metrics=metrics,epochs=epochs,batch_size=batch_size,dir_weights_save=dir_weights_save,
                        activation='softmax').Train()


    # Test
    test=Test_Model(model_name=model,testing_data=validation_ds,batch_size=batch_size,dir_weights_save=dir_weights_save)
    test.Evaluate_Model(validation_ds)
    test.Predict_Model(validation_ds)
    test.Accuracy_Result_Visual(history, model_current_name)
    test.Loss_Result_Visual(history, model_current_name)