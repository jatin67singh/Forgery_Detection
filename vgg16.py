from PIL import Image, ImageChops, ImageEnhance
import random
#import necessary libraries
import numpy as np
np.random.seed(2)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout,Input
from keras.src.optimizers.adam import Adam
import os
from keras.callbacks import EarlyStopping
import keras
from keras.models import load_model, Sequential
from keras.layers import Dense, Flatten
from keras.applications.resnet50 import ResNet50
# def get_imlist(path):
#     return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.jpg') or f.endswith('.png')]

def convert_to_ela_image(path, quality):
    im = Image.open(path).convert('RGB')
    resaved_filename = path.split('.')[0] + '.resaved.jpg'
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)
    ela_im = ImageChops.difference(im, resaved_im)
    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    
    # Prevent division by zero
    if max_diff == 0:
        max_diff = 1

    scale = 255.0 / max_diff 
    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)
    
    return ela_im

def prepare_image(image_path):
    return np.array(convert_to_ela_image(image_path, 90).resize(image_size)).flatten() / 255.0

# Example usage for an original image
# original_image_path = 'test_ela/Au_ani_10104.jpg'
# original_image = Image.open(original_image_path)
# original_image.show()

# ela_image_original = convert_to_ela_image(original_image_path, 90)
# ela_image_original.show()

# # Example usage for a forged image
# forged_image_path = 'test_ela/Tp_D_CRN_M_N_ani10104_ani00100_10093.tif'
# forged_image = Image.open(forged_image_path)
# forged_image.show()

# ela_image_forged = convert_to_ela_image(forged_image_path, 90)
# ela_image_forged.show()

image_size = (128, 128)
X = [] 
Y = [] 


path = 'data/AU'
for dirname, _, filenames in os.walk(path):
    print("dirname",dirname)
    print(filenames)
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(1)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

random.shuffle(X)
X = X[:2100]
Y = Y[:2100]
print(len(X), len(Y))

path = 'data/TP'
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        if filename.endswith('jpg') or filename.endswith('png'):
            full_path = os.path.join(dirname, filename)
            X.append(prepare_image(full_path))
            Y.append(0)
            if len(Y) % 500 == 0:
                print(f'Processing {len(Y)} images')

print(len(X), len(Y))

X = np.array(X)
Y = to_categorical(Y, 2)
X = X.reshape(-1, 128, 128, 3)

X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.2, random_state=5)

print(len(X_train), len(Y_train))
print(len(X_val), len(Y_val))

# Shape of X_train: (16, 128, 128, 3)
# Shape of X_val: (4, 128, 128, 3)
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape) 


vgg_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Define top dense layers
top_model = Sequential()
top_model.add(Flatten(input_shape=vgg_model.output_shape[1:]))
top_model.add(Dense(128, activation='relu'))
top_model.add(Dense(2, activation='softmax'))

# Combine VGG16 and top model
model_aug = Sequential([
    Input(shape=(128, 128, 3)),
    vgg_model,
    top_model
])

# Freeze layers of VGG16 for fine-tuning
for layer in model_aug.layers[1].layers[:15]:
    layer.trainable = False

# Load weights for VGG16 base model
vgg_model.load_weights('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)

# Compile the model
model_aug.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=1e-6), metrics=['accuracy'])

# Train the model
history = model_aug.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=32)

# Evaluate the model on validation data
loss, accuracy = model_aug.evaluate(X_val, Y_val)
print("Validation Loss:", loss)
print("Validation Accuracy:", accuracy)
