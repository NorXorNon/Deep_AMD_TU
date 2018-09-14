#K-Fold Cross Validation Densenet

from keras.applications.densenet import DenseNet121
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Flatten,Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint,EarlyStopping,CSVLogger
import numpy as np
import Dataset
import math
import random
#generate ref number for each round
ref = random.sample(range(10000, 40000),1)[0]
# initial parameter tuning
numepoch = 100
classNum = 5
batchsize = 32
numberOfFold = 10
# Load Dataset to parameters
_ , Train, Test = Dataset.TestTrainList('/workspace/DeepLearning_AMD/DataSet/flowers/',test_size=0.2,random_state=24,shuffle=True)
Dataset.data_to_txt('WeightLog/Train_ref_'+str(ref)+'_.txt',Train)
Dataset.data_to_txt('WeightLog/Test_ref_'+str(ref)+'_.txt',Test)
#initial parameters for training model
fold = 1


for dataTf , dataVf in Dataset._Kfold(data=Train,nFold=numberOfFold):

    base_model = DenseNet121(include_top=False,weights='imagenet',input_shape=(224, 224, 3))

    x = base_model.output
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(classNum, activation='softmax')(x)


    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = SGD(lr=0.0001)

    model.compile(optimizer=optimizer,
             loss = 'categorical_crossentropy',
             metrics=['accuracy']) 
    
    # set callbacks
    callbacks = [
        ModelCheckpoint(filepath=('WeightLog/fold_'+str(fold)+'_reference_'+str(ref)+'_.hdf5'), monitor='val_loss', 
                        verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1),
        EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto'),
        CSVLogger(filename='WeightLog/fold_'+str(fold)+'_reference_'+str(ref)+'_.csv', separator=',', append=False)
    ]

    print('Fold: ',fold)
    
    model.fit_generator(Dataset.batch_generator(
    data = dataTf,
    batch_size = batchsize,
    shape= (224,224),
    shuffleIm = True,
    num_class = classNum
    ),
                    steps_per_epoch=math.ceil(dataTf.shape[0]/batchsize), 
                    epochs=numepoch,
                    validation_data=Dataset.batch_generator(
    data = dataVf,
    batch_size = batchsize,
    shape=(224,224),
    shuffleIm = True,
    num_class = classNum
    ),
                       validation_steps=math.ceil(dataVf.shape[0]/batchsize),
                        callbacks=callbacks
                       )

    fold += 1

print('-----Trainning Successfully-----')
    
    