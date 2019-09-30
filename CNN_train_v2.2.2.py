###############################################################################
################################# Modules #####################################
###############################################################################



# Modules of Ross
import numpy as np
import glob
import os
from PIL import Image
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Other modules
import math
from copy import copy
import sys

# Plot Modules
import matplotlib.pyplot as plt
from matplotlib import pyplot  

# Augmentation
from keras.preprocessing.image import ImageDataGenerator

# File operations 
import shutil



###############################################################################
########################### Editable Setting ##################################
###############################################################################



# Size of images (resolution 64 or 128) (switch2)
###############################################################################

resolution = 128



# Classes of training : For not use (presence = False) or use (presence = True) and definition of 'classes_select'  
##################################################################################################################

presence= False
classes_select = []



# Create groups (switch1='on' -> activated mapping )
###############################################################################

switch1='on'   # 'on' or 'off'
# mapping = {'name of group 1' : [classe1, classe2,...],'name of group 2' : [...)  
mapping = {'amar': ['aa', 'aj', 'al', 'ca', 'co', 'dm'], 'mono':['c','p']}    



# Callbacks RateScheduler (choose of the learning rate)
###############################################################################

# "manual" or  "automatic" -> manual=predefined function (cf : def schedule2) ; automatic=in function of val_loss 
switch3='manual'       
base_learning_rate = 0.001

# Setting for 'automatic' only
DropRate=0.8



# Training
###############################################################################

# number of epoch
epoques=2

# batch size
size_of_batch = 101

# augmentation ('on' or 'off')
switch4 = 'on'

# if switch4 = 'on', select one type of augmentation (choose one on the two "datagen" below : add or remove the "#")
# For each image , random combination of parameters will be used to create images for training  (cf. model.fit_generator)
# The image creation is relaunched at each epoch, the image x will therefore be modified at each epoch.

#datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True) # base
datagen = ImageDataGenerator(rotation_range=90, horizontal_flip=True, width_shift_range=0.15, height_shift_range=0.15, vertical_flip=True) # full

# transfer learning ('on' or 'off')
switch5 = 'on'              
save_file = 'auto_save(c+other_mono, 128, off, off)'                  # file path                                                         
checkpoint = '0044'                                                   # save number




###############################################################################
####################### Memo for the loops variables ##########################
###############################################################################

# i -> individual counter 
# t -> time counter (loop counter)
# x -> variables for equations 



###############################################################################
################################ Definition ###################################
###############################################################################

# Definition of process_image
def process_image(filename, width):
    im = Image.open(filename).convert('L')                       # open and convert to greyscale
    im = np.asarray(im, dtype=np.float)                          # numpy array
    im_shape = im.shape
    im = resize(im, [width, width], order=1 , mode='reflect')    # resize using linear interpolation, replace mode='reflect' by 'constant' otherwise error message 
    im = np.divide(im, 255)                                      # divide to put in range [0 - 1]
    return im, im_shape

# Definition of image loading
def load_from_class_dirs(directory, extension, width, norm, min_count=20):
    print(" ")
    print("Loading images from the directory './" + directory + "'.")
    print(" ")
    # Init lists
    images = []
    labels = []
    cls = []
    filenames = []
    # Alphabetically sorted classes
    class_dirs = sorted(glob.glob(directory + "/*"))
    # Load images from each class
    idx = 0
    for class_dir in class_dirs:
        # Class name
        class_name = os.path.basename(class_dir)
        if (class_name in classes_select) == presence:       
            num_files = len(os.listdir(class_dir))
            print("%s - %d" % (class_name, num_files))
            if num_files < min_count: continue
            class_idx = idx
            idx += 1
            labels.append(class_name)
            # Get the files
            files = sorted(glob.glob(class_dir + "/*." + extension))
            for file in files:
                im, sz = process_image(file, width)
                if norm:
                    im = rescale_intensity(im, in_range='image', out_range=(0.0, 1.0))
                images.append(im)
                cls.append(class_idx)
                filenames.append(os.path.basename(os.path.dirname(file)) + os.path.sep + os.path.basename(file))
    # Final clean up
    images = np.asarray(images)
    cls = np.asarray(cls)
    num_classes = len(labels)

    return images, cls, labels, num_classes, filenames

# Definition of val_acc history in real time 
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = [20,20,20,20,19,19,19,19]

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))
        del self.losses[0] 

# Definition of function for the Learning Rate
def schedule(epoch):
  if t1[0]>0:
    t1[0]=t1[0]-1
  if epoch>4 and sum(history_val_loss.losses[4:]) > sum(history_val_loss.losses[:4]) and t1[0]==0 :
    new_lr[0] = base_lr[0]*math.pow(DropRate,x1[0])
    x1[0]=x1[0]+1
    print()
    print('###    Update of  Learning Rate : lr = {0}    ###'.format(new_lr))
    print()
    t1[0]=4
  history_lr.append(new_lr[0])
  print('lr = {0}     timer = {1}'.format(new_lr[0], t1[0]))
  print(history_val_loss.losses)
  return new_lr[0]

def schedule2(epoch, current_learning_rate):
    base_learning_rate = 0.001
    drop_amount = 0.6
    if epoch < 200 :
        x2=0
    elif epoch < 300 :
        x2=1
    elif epoch < 400 :
        x2=2
    elif epoch < 500:
        x2=3
    elif epoch < 600:
        x2=4
    elif epoch < 700:
        x2=5
    elif epoch < 800:
        x2=6
    elif epoch < 900:
        x2=7    
    else :
        x2=8  
    new_lr[0] = base_learning_rate * math.pow(drop_amount,x2)
    history_lr.append(new_lr[0])
    print('lr = {0}     timer = {1}'.format(new_lr[0], t1[0]))
    print(history_val_loss.losses)
    return new_lr[0]

    

###############################################################################
########################### Main scripte ######################################
###############################################################################
    


# Image extraction and conversion from the file "intact"
###############################################################################
    
# load_from_class_dirs("name of file", "extention type of images", "image size (in pixel)",...)
images, cls, labels, num_classes, filenames = load_from_class_dirs("intact", "tif", resolution, False, min_count=0)  



# Adding one dimension to images 
###############################################################################

# image dimensions must be [batch, width, height, channels]. But the images are "single channel" because "numpy" don't include the dimension "channel", we add this dimention with "np.newaxis"
images = images[:,:,:,np.newaxis] 



# Group creation with the 'mapping option'
###############################################################################

#*************************** switch1_start ************************************
if switch1=='on':
  print(" ")
  print("Activated mapping:")
  print(mapping)
  print(" ") 
  labels_mapping=[]
  mapping_security=[]
  cls_mapping = copy(cls)
  for key in mapping.keys():
    labels_mapping.append(key)
    for val in mapping[key]:
      mapping_security.append(val)  
      for i1 in range(0,len(cls)):
        if cls[i1]==labels.index(val):
          cls_mapping[i1]=(labels_mapping.index(key))
  for security_1 in labels :
    if (security_1 in mapping_security) == False :
      sys.exit('Error : "' + security_1 + '" is not found in "mapping"! Test stop.')           
else:
  print(" ")
  print("Desactivated mapping.")
  print(" ") 
  labels_mapping = copy(labels) 
  cls_mapping = copy(cls)  
#*************************** switch1_end  ************************************* 
  
  
  
# Splite of images(=images) and images lables (=cls) in two sets (train set and test set)
##########################################################################################

train_images, test_images, train_labels, test_labels, train_labels_mapping, test_labels_mapping = train_test_split(images, cls, cls_mapping, test_size=0.20, random_state=42)            # test_size=0.25 = pourcentage de test_size



# Augmentation 
###############################################################################

# Application of ImageDataGenerator for training set
datagen.fit(train_images)  



# Configuration of auto save at each epoch
###############################################################################

os.mkdir('CNN_solo/auto_save')    # Creation of auto-save file 
print('Auto-save folder operational.') 
print(' ')
checkpoint_path = "./CNN_solo/auto_save/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1,period=1) 


# Configuration of val_acc history in real time
###############################################################################

history_val_loss = LossHistory()


# Configuration of change of learning rate during the training  
###############################################################################

history_lr=[]
t1=[0]
x1=[1]
base_lr=[base_learning_rate]
new_lr=[base_learning_rate]        

#*************************** switch3_start ************************************
if switch3=="automatic" :
  RateScheduler=tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
else :
  RateScheduler=tf.keras.callbacks.LearningRateScheduler(schedule2, verbose=0)
#*************************** switch3_end  ************************************* 


# Optimizer and Network consctruction  
###############################################################################  

# Optimizer
opt = tf.keras.optimizers.Adam(lr=base_learning_rate)

# Network
#*************************** switch2_start ************************************
if resolution == 64 :
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dropout(0.5,seed=7))
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dense(len(labels_mapping), activation='softmax'))
else :
  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
  model.add(tf.keras.layers.MaxPooling2D())
  model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for image by 128 pixel 
  model.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for image by 128 pixel
  model.add(tf.keras.layers.MaxPooling2D())                                         # additional layer for image by 128 pixel 
  model.add(tf.keras.layers.Flatten())
  model.add(tf.keras.layers.Dropout(0.5,seed=7))
  model.add(tf.keras.layers.Dense(512, activation='relu'))
  model.add(tf.keras.layers.Dense(len(labels_mapping), activation='softmax'))  
#*************************** switch2_end  *************************************   
model.count_params()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.summary()



# Loading of save of model parameter without augmentation (if transfert lerning is on)
###############################################################################

#*************************** switch5_start ************************************
if switch5 == 'on' :
  if os.path.isfile('./CNN_solo/' + save_file + '/cp-' + checkpoint + '.ckpt') == False :
    sys.exit('Error : "' + './CNN_solo/' + save_file + '/cp-' + checkpoint + '.ckpt' + 'is not found! Test stop.')
  model.load_weights('./CNN_solo/' + save_file + '/cp-' + checkpoint + '.ckpt')
#*************************** switch5_end  *************************************   


# Configuration of training paremeter
###############################################################################
  
#*************************** switch4_start ************************************
if switch4 == 'off' :
  history = model.fit(train_images, train_labels_mapping, epochs=epoques , verbose=1, batch_size = size_of_batch ,validation_data=(test_images, test_labels_mapping), callbacks = [cp_callback, history_val_loss, RateScheduler]) 
else :
  history = model.fit_generator(datagen.flow(train_images, train_labels_mapping, batch_size=size_of_batch), epochs=epoques, verbose=1, validation_data=(test_images, test_labels_mapping), callbacks = [cp_callback, history_val_loss, RateScheduler])
#*************************** switch4_end  *************************************   



# Histograms of accuacy, loss and lr 
###############################################################################

# Basic parameters  
epoch_index = range(1,len(history.history['acc'])+1,1)
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
figure = pyplot.figure(figsize = (10, 10))
pyplot.gcf().subplots_adjust(left = 0.1, bottom = 0.1,right = 0.9, top = 0.9, wspace = 0, hspace = 0.1)

# Summarize history for accuracy
axes = figure.add_subplot(3, 1, 1)
plt.plot(epoch_index, acc)
plt.plot(epoch_index, val_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel(' ')
plt.legend(['train', 'test'], loc='upper left')

# Summarize history for loss
axes = figure.add_subplot(3, 1, 2)
plt.plot(epoch_index,loss)
plt.plot(epoch_index, val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# Summarize history for Ir
axes = figure.add_subplot(3, 1, 3)
plt.plot(epoch_index,history_lr)
plt.title('history_lr')
plt.ylabel('lr')
plt.xlabel('epoch')
plt.legend(['lr'], loc='upper left')



# Final save
###############################################################################

# Save of Histograms
plt.savefig('./CNN_solo/auto_save/historique.jpg')
plt.show()

# Save history of accuacy, loss and lr in .csv
import csv
with open('./CNN_solo/auto_save/historique.csv','w') as csvfile:
  writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
  writer.writerow(('epoch', 'acc', 'val_acc', 'loss', 'val_loss','learning_rate'))
  for i2 in range (0, len(history.history['acc'])):
    writer.writerow([epoch_index[i2],acc[i2], val_acc[i2], loss[i2], val_loss[i2],history_lr[i2]])

# Save one copy of the script used (with current parameter)
shutil.copyfile('1_CNN_train_v2.2.2.py', 'CNN_solo/auto_save/script_used.txt')    
    
    
    