'''
Made by Lee Jin, GwangHyun Yu, Dang Thanh Vu
Initial version: 2023.01.16
Modified version: 2023.01.23
'''

import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential,Model,initializers,layers,Input
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping

import os
from EffShuff_Dense_Block import*
from EffShuff_Block import *
from Preprocessing import *

from ShuffleNet_v2 import*

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

if __name__ == "__main__":

    run = neptune.init_run(
            project="vuyuanh/gender-classification1",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYWYxMDUxNy1hOGUzLTRmN2YtOWQzNi0xZGYzNGYwMmQ0NTcifQ==",
        )  # gender classification : Male / Female / Baby

    model_list=['densenet201','densenet201_scratch','resnet50','resnet50_scratch','mobilnetv2','mobilnetv2_scratch',
                'efficientnet','efficientnet_scratch','Y_Block', 'Y_Dense_Block','shuffleNetv2']

    # Hyperparameters for Function
    params = {
            'model_name': 'efficientnet',
        
            #Training
            'opt': "SGD",
            "lr": 1e-5,
            "lr_schedule": None,
            "momentum": 0.9, 
            "epochs": 100, 
            "batch_size": 64,
            "loss" : 'categorical_crossentropy',
            "metrics": ['acc'],

            #dataset
            "num_classes": 3,   # butterfly:100 / gender:4 - 3(No none)/ age:30 (with none) - 29 (No none) - 10 (unite)
            "img_shape": (224,224),
            
            #directory
            "dir": r'D:\Coding\DeepLearning\dataset\age_gender\gender_none',
            'dir_weights_save': r'D:\Coding\DeepLearning\y_block\save_weights\gender_none'
           }
    run["parameters"] = params

    if(params['lr_schedule'] == "ExponentialDecay"):
        lr = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=params['lr'], decay_steps=10000, decay_rate=0.97, staircase=False, name=None)
        # tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=10000, decay_rate=0.97, staircase=False, name=None)
    elif(params['lr_schedule'] == "CosineDecayRestarts"):
        lr = tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=params['lr'], first_decay_steps=10, t_mul=1, m_mul=0.9, alpha=0)
        # tf.keras.experimental.CosineDecayRestarts(initial_learning_rate=0.1, first_decay_steps=10, t_mul=1, m_mul=0.9, alpha=0)
    elif(params['lr_schedule'] == "lr_warmup_cosine_decay"):
        lr = tf.keras.experimental.CosineDecay(initial_learning_rate=params['lr'], decay_steps=50, alpha=0.001)
        # tf.keras.experimental.CosineDecay(initial_learning_rate=0.001, decay_steps=50, alpha=0.001)
    elif(params['lr_schedule'] == None):
        lr = params['lr']

    
    if(params['opt'] == "Adam"):
        opt = tf.optimizers.Adam(learning_rate=lr)
    elif(params['opt'] == "SGD"):
        opt = tf.keras.optimizers.SGD(learning_rate=lr, momentum=params['momentum'])
   

    model=Get_Model(model_name=params['model_name'],num_classes=params['num_classes']).Get_Train_Model()
    print("Train model:  ", params['model_name'])

    # 가중치 저장 경로
    dir_weights_save=os.path.join(params['dir_weights_save'], params['model_name'], params['model_name']+'_gender')
    print("Save dir:  ", params['dir_weights_save'])

    # 데이터 전처리
    print("Load Data:  ", params['dir'])
    training_ds=Preprocessing(dir=params['dir'],batch_size=params['batch_size'],validation_split=0.1,
                            num_classes=params['num_classes'],subset='training',
                            shuffle=True,seed=123,img_size=params['img_shape']).Get_Preprosessed_Dataset()
                        
    validation_ds=Preprocessing(dir=params['dir'],batch_size=params['batch_size'],validation_split=0.1,
                            num_classes=params['num_classes'],subset='validation',
                            shuffle=False,seed=123,img_size=params['img_shape']).Get_Preprosessed_Dataset()
     
    # Train
    neptune_cbk = NeptuneCallback(run=run, base_namespace="training")

    early_stopping=EarlyStopping(monitor='val_acc',patience=15,min_delta=1e-4)
    model.compile(optimizer=opt,loss=params['loss'],metrics=params['metrics'])
    history= model.fit(training_ds,validation_data=validation_ds,
            epochs=params['epochs'],batch_size=params['batch_size'],verbose=1,
            callbacks=[neptune_cbk,
            ModelCheckpoint(filepath=dir_weights_save,
            monitor='val_acc',save_best_only=True,save_weights_only=True,save_freq='epoch'),
            early_stopping
            ])

    # Test
    print("Eval model:  ", params['model_name'])
    # Load best_save_weights
    model.load_weights(dir_weights_save)
    eval_metrics = model.evaluate(validation_ds, verbose=0)
    for j, metric in enumerate(eval_metrics):
        run["eval/{}".format(model.metrics_names[j])] = metric

    # Calcurate Flops
    flops = get_flops(model, batch_size=params['batch_size'])
    print(f"FLOPS: {flops / 10 ** 9:.03} G")

    run.stop()
