import tensorflow as tf
from tensorflow.keras import models,layers
import matplotlib.pyplot as plt
import numpy as np
import pickle

IMAGE_SIZE = 256
BATCH_SIZE = 32 # each batch contain 32 images
EPOOCH = 5
CHANNELS = 3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "PlantVillage",
    shuffle = True,
    image_size = (IMAGE_SIZE, IMAGE_SIZE),
    batch_size = BATCH_SIZE )


class_names = dataset.class_names
print(class_names)


train_size = 0.8
train_ds = dataset.take(54)
test_ds = dataset.skip(54) # remaining datset after 54
valid_size = 0.1
test_size = test_ds.skip(6)
valid_ds = test_ds.take(6)
test_ds = test_ds.skip(6)

def splitting_data(dataset, train_split = 0.8, test_split = 0.1, val_split = 0.1, shuffle = True, shuffle_size = 100000):
    ds_size = len(dataset)
    
    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed = 12)
        
        
    train_size = int(ds_size * train_split)
    val_size = int(ds_size * val_split)
    
    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = splitting_data(dataset)

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)

resize_and_rescale = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.Rescaling(1.0/255)
    
])

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip('horizontal'),
    layers.experimental.preprocessing.RandomRotation(0.2)
    
])

# neural network architecture

input_shape = (BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
n_classes = 3
# model building with set of layers  like rescale, resize, data augmentation, cnn layer
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32, (3,3),activation = 'relu', input_shape = input_shape), # no. of layers(trail and error), actual filter size,activation layer
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size = (3,3),activation = 'relu'), # no. of layers(trail and error), actual filter size,activation layer
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, kernel_size = (3,3),activation = 'relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3),activation = 'relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3),activation = 'relu', input_shape = input_shape), 
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3,3),activation = 'relu'), 
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dense(n_classes, activation = 'softmax'),
    
])

model.build(input_shape = input_shape)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

# history of every epoch
history = model.fit(
    train_ds,
    epochs = EPOOCH,
    batch_size = BATCH_SIZE,
    verbose = 1,
    validation_data = val_ds
)

scores = model.evaluate(test_ds) #

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0) # create a batch
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

pickle.dump(model, open('iri.pkl','wb'))

