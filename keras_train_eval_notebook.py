
# %%
from importlib import import_module
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pathlib 
import os
import PIL
import glob
import json
import tensorflow as tf
from  tensorflow import keras
from  tensorflow.keras import layers
from keras.preprocessing import image
import tensorflow_datasets as tfds
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from tensorflow.python.keras.layers import Input, Dense, Flatten

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam



# ======================= VARIABLES =======================
IMAGES_PATH = "./data/perfectly_detected_ears/test"
ANNOTATIONS_PATH = "./data/perfectly_detected_ears/annotations/recognition/ids.csv"
DATA_DIR = "./data/perfectly_detected_ears/subfoldered_train"

IMG_HEIGHT,IMG_WIDTH = 128, 128
N_CLASSES = 100
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# check for GPU
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
data_dir = pathlib.Path(DATA_DIR)
# ======================= VARIABLES =======================
# %% 
# ======================= DATA AUGMENTATION WITH GENERATOR =======================
# create data generator
VALIDATION_SPLIT = 0.25
datagen = ImageDataGenerator(
	#featurewise_center=True,
	#featurewise_std_normalization=True,
	rescale=1. / 255,
	rotation_range=20,
	width_shift_range=0.1,
	height_shift_range=0.1,
	zoom_range=0.5,
	horizontal_flip=True,
	vertical_flip=True,
	validation_split = VALIDATION_SPLIT,
)

valid_datagen=ImageDataGenerator(rescale=1./255,validation_split=VALIDATION_SPLIT)

## generators 
train_generator = datagen.flow_from_directory(
    directory=DATA_DIR,
    subset="training",
    batch_size=BATCH_SIZE,
    seed=123,
    shuffle=False,
    class_mode="categorical",
    target_size=(IMG_HEIGHT,IMG_WIDTH))

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

valid_generator = valid_datagen.flow_from_directory(
    directory=DATA_DIR,
    subset="validation",
    batch_size=BATCH_SIZE,
    seed=123,
    shuffle=False,
    class_mode="categorical",
    target_size=(IMG_HEIGHT,IMG_WIDTH))

STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
# ======================= DATA AUGMENTATION WITH GENERATOR =======================
# %%
# ======================= DATASET FROM DIRECTORY =======================
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="training",
  seed=123,
  label_mode='categorical',
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  DATA_DIR,
  validation_split=0.2,
  subset="validation",
  seed=123,
  label_mode='categorical',
  image_size=(IMG_HEIGHT, IMG_WIDTH),
  batch_size=BATCH_SIZE)

print(np.array(train_ds.class_names))
# =======================/ DATASET FROM DIRECTORY =======================

# %%
# ======================= DATA AUGMENTATION FOR ABOVE DATASETS =======================
train_ds2 = train_ds.map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)
).cache().map(
    lambda image, label: (tf.image.random_flip_left_right(image), label)
).map(
    lambda image, label: (tf.image.random_contrast(image, lower=0.8, upper=1.2), label)
).shuffle(
    100
).repeat()


val_ds2 = val_ds.map(
    lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)
).cache()
# =======================/ DATA AUGMENTATION FOR ABOVE DATASETS =======================
# %%
data_augmentation = keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1),])
# Augment training images.
augmented_train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

print(np.array(augmented_train_ds.class_names))
# ======================= DATA AUGMENTATION NO GENERATOR =======================


################################### SELECT PRETRAINED MODEL ###################################
# %%
# ======================= Resnet50 =======================
pretrained_model = tf.keras.applications.ResNet50(include_top=False,
				   input_shape=(IMG_HEIGHT,IMG_WIDTH,3),
				   pooling='avg',
				   classes=N_CLASSES,
				   weights='imagenet')
for layer in pretrained_model.layers:
		layer.trainable=False
BASE_MODEL = "Resnet50"
# =======================/ Resnet50 =======================
# %%
# ======================= DenseNet121 =======================
pretrained_model = tf.keras.applications.DenseNet121(include_top=False,
				   input_shape=(IMG_HEIGHT,IMG_WIDTH,3),
				   pooling='avg',
				   classes=N_CLASSES,
				   weights='imagenet')
for layer in pretrained_model.layers:
		layer.trainable=False
BASE_MODEL = "DenseNet121"
# =======================/ DenseNet121 =======================
# %%
# ======================= EfficientNetB0 =======================
pretrained_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_HEIGHT,IMG_WIDTH,3),
    pooling=None,
	include_top= False,
    classes=N_CLASSES,
    classifier_activation="softmax",
    weights="imagenet"
)
for layer in pretrained_model.layers:
		layer.trainable=False
BASE_MODEL = "EfficientNetB0"
# ======================= EfficientNetB0 =======================
###################################/ SELECT PRETRAINED MODEL ###################################
# %%
# ======================= COMPILE MODEL =======================
model = Sequential([
	pretrained_model,
	Flatten(),
	Dense(512, activation='relu'),
	Dense(N_CLASSES, activation='softmax')
]) 
model.compile(optimizer=Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()
# =======================/ COMPILE MODEL =======================
# %%
# ======================= TRAIN MODEL ======================
epochs_ = 30
history = model.fit(
	train_ds,
	epochs=epochs_,
	#steps_per_epoch = 100,
	validation_data=val_ds
)
history.history['base_model'] = BASE_MODEL
# =======================/ TRAIN MODEL =======================
# %%
# =======================/ PLOT HISTORY =======================
fig, (pltAcc, pltLoss) = plt.subplots(2)
fig.suptitle(history.history['base_model'])

fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)

pltAcc.set_title('Model accuracy')
pltAcc.plot(history.history['accuracy'])
pltAcc.plot(history.history['val_accuracy'])
pltAcc.set_ylabel('accuracy')
pltAcc.set_xlabel('epoch')
pltAcc.legend(['train', 'val'], loc='center right')

pltLoss.set_title('Model loss')
pltLoss.plot(history.history['loss'])
pltLoss.plot(history.history['val_loss'])
pltLoss.set_ylabel('loss')
pltLoss.set_xlabel('epoch')
pltLoss.legend(['train', 'val'], loc='center right')
plt.show()
# =======================/ PLOT HISTORY =======================
# %%
# ======================= SAVE MODEL =======================
# Save the trained model
model.save("./models/densenet121_no_augmentation_10e")
# =======================/ SAVE MODEL =======================

# %%
# =======================/ LOAD MODELS =======================
model_resnet = keras.models.load_model("./models/resnet50_no_augmentation_20e")
model_resnet.compile()
model_resnet.summary()

model_densenet = keras.models.load_model("./models/densenet121_no_augmentation_20e")
model_densenet.compile()
model_densenet.summary()

model_efficientnet = keras.models.load_model("./models/efficientNetB0_no_augmentation_20e")
model_efficientnet.compile()
model_efficientnet.summary()
# =======================/ LOAD MODEL =======================



# %%
def get_annotations(annot_f):
	print(annot_f)
	d = {}
	with open(annot_f) as f:
		lines = f.readlines()
		for line in lines:
			(key, val) = line.split(',')
			# keynum = int(self.clean_file_name(key))
			d[key] = int(val)
	return d


# %%
im_list = sorted(glob.glob(IMAGES_PATH + '/*.png', recursive=True))
cla_d = get_annotations(ANNOTATIONS_PATH)
# Actual classes
y = []
Y = []
# Predicted classes
Y_resnet = []
Y_densenet = []
Y_efficientnet = []
for im_name in im_list[:100]:
			# Read an image
			img = cv2.imread(im_name)
			y.append(cla_d["test/" + '/'.join(im_name.split('\\')[-1:])])

			img = image.load_img(im_name, target_size=(IMG_HEIGHT, IMG_WIDTH))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			images = np.vstack([x])
			
			# Current model
			predict_x = model.predict(images) 
			Y.append(predict_x[0])

			# Loaded models
			""" predict_x = model_resnet.predict(images)
			Y_resnet.append(predict_x[0])

			predict_x = model_densenet.predict(images)
			Y_densenet.append(predict_x[0])

			predict_x = model_efficientnet.predict(images)
			Y_efficientnet.append(predict_x[0]) """

			#print("predicted: " + str(np.argmax(predict_x[0],axis=0)) + ', (actual:' + str(cla_d["test/" + '/'.join(im_name.split('\\')[-1:])]) + ')')

# =======================/ EVALUATE USING DIFFERENT METRICS =======================
# %%
import importlib
import metrics.evaluation_recognition as eval_
importlib.reload(eval_)
eval =  eval_.Evaluation()


cmc = eval.compute_CMC_ranks_nn(Y, y, cmc_max_rank)
print("rank1: ", cmc[0], "%")

cmcs = [cmc]
""" cmcs =  []
cmc_max_rank = 100
cmcs.append(eval.compute_CMC_ranks_nn(Y_resnet, y, cmc_max_rank))
cmcs.append(eval.compute_CMC_ranks_nn(Y_densenet, y, cmc_max_rank))
cmcs.append(eval.compute_CMC_ranks_nn(Y_efficientnet, y, cmc_max_rank)) """


# ======================= EVALUATE USING DIFFERENT METRICS =======================
# %%
# ======================= PLOT CMC ======================= 
ranks = list(range(cmc_max_rank))
colors = ['or', 'ob', 'og', 'oy', 'oo', 'og']
labels = ["ResNet50", "DenseNet121", "EfficientNetB0"]

fig, ax = plt.subplots()

ax.set(xlabel='Rank',
       ylabel='Rank accuracy %',
       title='CMC graph')

for idx, cmc in enumerate(cmcs):
	ax.plot(ranks, cmc, label = labels[idx])

plt.legend(loc="lower right")
plt.show()
# ======================= PLOT CMC =======================
#%%
# ======================= PLOT rank1 and rank5 =======================
x = np.arange(len(labels))
width = 0.35
rank1s = [cmcs[0][0], cmcs[1][0], cmcs[2][0]]
rank5s = [cmcs[0][4], cmcs[1][4], cmcs[2][4]]

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, rank1s, width, label='Rank 1')
rects2 = ax.bar(x + width/2, rank5s, width, label='Rank 5')

ax.set_ylabel('Rank accuracy %')
ax.set_title('Rank accuracy by model')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()
# =======================/ PLOT rank1 and rank5 =======================
# %%
