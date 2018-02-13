
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.models import load_model
from keras import backend as K
from keras.layers import Activation
from keras.layers import Input
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import Cropping3D
from keras.layers.core import Permute
from keras.layers.core import Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU 
from keras.preprocessing.image import ImageDataGenerator

# dimensions of our images.
img_width, img_height =150, 150

train_data_dir = '/home/tigerz/cad/rot/train/'
validation_data_dir = '/home/tigerz/cad/rot/val/'
model_filename = 'outrun_step_{}.h5'
csv_filename = 'outrun_step_{}.cvs'
nb_train_samples = 4072
nb_validation_samples = 2704
epochs = 500
batch_size = 256
num_classes=2
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)



def generate_model(num_classes) :
    # input layer
    visible = Input(shape=(227, 227, 3))
    

    conv1 = Conv2D(24, (5,5), activation='relu')(visible)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    batch1 = BatchNormalization(axis=-1)(pool1)
    conv2 = Conv2D(32, (3,3), activation='relu')(batch1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    batch2 = BatchNormalization(axis=-1)(pool2)
    conv3 = Conv2D(48, (3,3), activation='relu')(batch2)
    flat2 = Flatten()(conv3)
    batch3 = BatchNormalization()(flat2)
    hidden2 = Dense(256, activation='relu')(batch3)
    batch4 =  BatchNormalization()(hidden2)
    drop1 =  Dropout(0.2)(batch4)

    #merge = concatenate([flat1, flat2])
    # interpretation layer
   # hidden1 = Dense(128, activation='relu')(merge)
#    hidden2 = Dense(128, activation='relu')(flat2)

    # prediction output
    output = Dense(2, activation='softmax')(drop1)
    model = Model(inputs=visible, outputs=output)


    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy'])
    return model


stopper = EarlyStopping(patience=1)

# Model checkpoint to save the training results
checkpointer = ModelCheckpoint(
    filepath=model_filename.format(1),
    verbose=1,
    save_best_only=True,
    save_weights_only=True)

# CSVLogger to save the training results in a csv file
csv_logger = CSVLogger(csv_filename.format(1), separator=';')

callbacks = [checkpointer, csv_logger, stopper]


# Build model
model = generate_model(num_classes)

# data augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range = 360,
    zoom_range=0.5,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling because we dont want to augment testing image
test_datagen = ImageDataGenerator(rescale=1. / 255)

# resizing and producing the mini batches 
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')



model.fit_generator(
      train_generator,
      steps_per_epoch=nb_train_samples // batch_size,
      epochs=epochs,
      verbose=2,
      validation_data=validation_generator,
      validation_steps=nb_validation_samples // batch_size)
      #callbacks=callbacks)

