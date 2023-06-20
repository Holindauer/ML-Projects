import pathlib
import os
from keras.preprocessing.image import ImageDataGenerator

class Load_ImageData:
    def Load_ImgData(path):
        '''
        This function will load in image data using ImageDataGenerator. It will
        do some basic preproccessing such as scaling rgb intensities to [0,1]

        The function assumes the directory being accessed contains just three folders.
        Them being : test, train, validation. (and that they are writen as such in alphabetical
        order). as well that within those directories are subdirectories containing titled after
        the different classes.

        It takes argument path which is the directory that contains these subdirectories.
        '''

        
        path = pathlib.Path(path)
            
        train_test_val_folders = os.listdir(path)
        data_path = []

        for set_folder in train_test_val_folders:
            set_ = os.path.join(path, set_folder)
            data_path.append(set_)

        print(data_path)

        #initialize image data generator
        train_gen=ImageDataGenerator(rescale=1./255,validation_split=0.2)

        #load train
        train_data=train_gen.flow_from_directory(data_path[1],
                                                target_size=(224,224),
                                                batch_size=32,
                                                class_mode='categorical',
                                                shuffle=True)

        #load test
        test_data=train_gen.flow_from_directory(data_path[0],
                                                target_size=(224,224),
                                                batch_size=1,
                                                shuffle=False)   

        #load val
        val_data=train_gen.flow_from_directory(data_path[2],
                                                target_size=(224,224),
                                                batch_size=1,
                                                shuffle=False)   
        
        return train_data, test_data, val_data
    
