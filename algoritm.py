import matplotlib.pyplot as plt
import numpy as np
# import PIL#Python Imaging Library.
import tensorflow as tf
# import pathlib
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.metrics import classification_report
from tensorflow.keras.activations import relu #Rectified Linear Unit


dataset_url = "dataset/train"

img_height= 100
img_width = 100
batch_size = 32

train_ds = tf.keras.utils.image_dataset_from_directory( 
    dataset_url, 
    validation_split=0.2, 
    subset= 'training', #, so the function will return a dataset for the training subset
    seed = 256, #sets the seed for shuffling the data
    image_size=(img_height,img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=0.2,
  subset="validation",
  seed=256,
  image_size=(img_height,img_width),
  batch_size=batch_size
)

# Print labels

class_names = train_ds.class_names
print(class_names)


plt.figure(figsize=(12, 12))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
    
    AUTOTUNE = tf.data.AUTOTUNE
    #tf.data.AUTOTUNE is a special value that allows TensorFlow to automatically tune the parameters of the input pipeline dynamically at runtime

train_ds = train_ds.cache().shuffle(buffer_size=batch_size).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().shuffle(buffer_size=batch_size).prefetch(buffer_size=AUTOTUNE)
# to introduce randomness and reduce any bias during each epoch of training
#Caching keeps the data in memory after it's loaded from disk

num_classes = len(class_names)

data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),# adds a layer for randomly flipping the input images horizontally and vertically
  layers.RandomRotation(0.5),# adds a layer for randomly rotating the input images
])

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)), #resacle to be between 0 and 1
  data_augmentation,
  layers.Conv2D(16, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),  
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),  
  layers.Flatten(),
   layers.Dense(256, activation='relu'),
  layers.Dense(num_classes)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

epochs=2
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

test_url = "dataset/test"

test_ds = tf.keras.utils.image_dataset_from_directory( 
    test_url, 
    seed = 256, 
    image_size=(img_height,img_width),
    shuffle=False #No shuffling for classification report
)

test_images, test_labels = tuple(zip(*test_ds))

predictions = model.predict(test_ds)
score = tf.nn.softmax(predictions)#This converts the predicted logits into probabilities

results = model.evaluate(test_ds)
print("Test loss, test acc:", results)

y_test = np.concatenate(test_labels) 
y_pred = np.array([np.argmax(s) for s in score])

import pickle #module is imported to enable object serialization and deserialization
filename="dataset/train"
pickle_out=open('model.pkl','wb')
pickle.dump(model,pickle_out)
pickle_out.close()
