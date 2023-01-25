import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential,Model,initializers,layers,Input
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
import os
import matplotlib.pyplot as plt
from keras_flops import get_flops


class Train_Model():
        def __init__(self,model_name,training_ds,validation_ds,num_classes,opt,loss,metrics,epochs,batch_size,dir_weights_save,activation='softmax'):
            self.model_name=model_name
            self.num_classes=num_classes
            self.opt=opt
            self.loss=loss
            self.metrics=metrics
            self.epochs=epochs
            self.batch_size=batch_size
            self.dir_weights_save=dir_weights_save
            self.training_ds=training_ds
            self.validation_ds=validation_ds
            self.activation=activation
            

        def Train(self):
            '''
            pretrained model    : [keras model object] pretrained model
            train_ds            : [map data object] training data object
            validation_ds       : [map data object] validation data object
            dir_weights_save    : [str] directory of weights to save
            epochs              : [int] epochs of training
            batch_size          : [int] mini batch of traning model
            loss                : [str] loss function mse/categorical_crossentropy/sparse_categorical_crossentropy
            optimizer           : [str] training optimizer of model
            metrics             : [list] training metrics of model 
            '''
            early_stopping=EarlyStopping(monitor='val_acc',patience=20,min_delta=1e-4)
            self.model_name.compile(optimizer=self.opt,loss=self.loss,metrics=self.metrics)
            history=self.model_name.fit(self.training_ds,validation_data=self.validation_ds,epochs=self.epochs,batch_size=self.batch_size,verbose=1,callbacks=[ModelCheckpoint(filepath=self.dir_weights_save,
            monitor='val_acc',verbose=1,save_best_only=True,save_weights_only=True,save_freq='epoch'),early_stopping])

            return history

class Test_Model():
    def __init__(self,model_name,testing_data,batch_size,dir_weights_save):
        self.model_name=model_name
        self.testing_data=testing_data
        self.batch_size=batch_size
        self.dir_weights_save=dir_weights_save
            
    def Evaluate_Model(self,data):
        '''
        pretrained model    : [keras model object] pretrained model
        dir_weights_save    : [str] directory of weights to save
        data                : [map data object] training or validation or test dataset for test
        '''
        self.model_name.load_weights(self.dir_weights_save),
        loss,acc=self.model_name.evaluate(data,verbose=1)

    def Predict_Model(self,data):
        '''
        pretrained model    : [keras model object] pretrained model
        dir_weights_save    : [str] directory of weights to save
        data                : [map data object] training or validation or test dataset for test
        '''
        tf.random.set_seed(123)

        image_batch, labels_batch = next(iter(data))
        print(image_batch[0:1].shape)
        print(labels_batch[0:1].shape)

        prediction = np.argmax(self.model_name.predict(image_batch[0:1]), axis=1)
        print(prediction)
        print(np.argmax(labels_batch[0:1]))

    def Get_Flops(self):
        '''
        pretrained model    : [keras model object] pretrained model
        batch_size          : [int] mini batch of traning modele
        '''  
        flops = get_flops(self.model_name,batch_size=self.batch_size)
        print(f"FLOPS: {flops / 10 ** 9:.03} G")
    

    def Accuracy_Result_Visual(self,history,name):
        '''
        history    : [history object] return value of function Trian_Model
        '''
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('pretrained model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.savefig('{0} acc.png'.format(name), dpi=300)

    def Loss_Result_Visual(self,history,name):
        '''
        history    : [history object] return value of function Trian_Model
        '''
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('pretrained model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig('{0} loss.png'.format(name), dpi=300)


