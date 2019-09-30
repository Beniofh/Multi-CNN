###############################################################################
################################# Modules #####################################
###############################################################################



# Module de Ross
import numpy as np
import glob
import os
from PIL import Image
from skimage.transform import resize
from skimage.exposure import rescale_intensity
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Plots
import matplotlib.pyplot as plt

# Confusion_matrix
import itertools
from sklearn.metrics import confusion_matrix

# Autre
from copy import copy
import sys



###############################################################################
########################### Editable Setting ##################################
###############################################################################



# Disable message of runtime warning for invalid value encountered in divide
###############################################################################

# Our code is trying to "divide by 0" in plot_confusion_matrix_synthese for normalized matrix
# But is not one problem because if the divide is not possible, we can see "nan" in the plot of matrix
np.seterr(divide='ignore', invalid='ignore') 



# Select our dataset : 'intact' or 'damaged' or 'fossil'
###############################################################################

# 'intact'-> is the test set of the modern intact dataset (part of dataset not used in training)
# 'damaged'-> modern damaged dataset 
# 'fossil" -> fossil dataset 
switch2='intact' 



# Size of images (resolution 64 or 128) (switch3)
###############################################################################
resolution = 128



# Classes of training : For not use (presence = False) or use (presence = True) and definition of 'classes_select'  
##################################################################################################################

presence= False
classes_select = []



# Create groups (switch1='on' -> activated mapping )
###############################################################################

switch1='off'   # 'on' or 'off'
# mapping = {'name of group 1' : [classe1, classe2,...],'name of group 2' : [...)  
mapping = {'aa': ['aa'], 'other' : ['aj', 'al', 'ca', 'co', 'dm']}  



# Localisation of model saves : indicate the number of the epoch 
dossier_save = 'auto_save(all, 128, off, off)'    
checkpoint = '0158'    

### Parametre of image plot
num_rows_plot = 200                                                              
num_cols_plot = 3                                                             
condition = 'test_labels_mapping[i] != sonic[i]'    



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
        if (class_name in classes_select) == presence:     # Pour ne pas prendre (si presence = False)/prendre (si presence = True) les groupe seclectionner                        
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

# Definition of image plots
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])  
  plt.imshow(img, cmap=plt.cm.binary)
  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'  
  plt.xlabel("{} {:2.0f}% ({})".format(labels_mapping[predicted_label],
                                100*np.max(predictions_array),
                                labels_mapping[true_label]),
                                color=color)
  plt.ylabel(test_filenames[i][0:len(test_filenames[i])-4])

# Definition of histogram plots 
def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks(range(num_class), labels_mapping, rotation=45)       # range(nomber of classe)
  plt.yticks([])
  thisplot = plt.bar(range(num_class), predictions_array, color="#777777")      # range(nomber of classe)
  plt.ylim([0, 1]) 
  predicted_label = np.argmax(predictions_array) 
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()



###############################################################################
########################### Main scripte ######################################
###############################################################################
    
    
    
# Extration and converstion of images in folder "images"
###############################################################################

#*************************** switch2_start ************************************
if switch2=='intact':
  path_folder_image = "intact"
else : 
  path_folder_image = switch2
#*************************** switch2_end **************************************
images, cls, labels, num_classes, filenames = load_from_class_dirs(path_folder_image, "tif", resolution, False, min_count=0)  ### load_from_class_dirs("nom du dossier", "extention des images", "taille des images en pixel",...)



# image dimensions must be [batch, width, height, channels]. But the images are "single channel" because "numpy" don't include the dimension "channel", we add this dimention with "np.newaxis"
###############################################################################

images = images[:,:,:,np.newaxis]



#Group creation with the 'mapping option'
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
#*************************** switch1_end ************************************** 
  


# Variable attribution
###############################################################################  
  
#*************************** switch2_start ************************************
if switch2=='damaged'or switch2=='fossil':
  # Attribution des variables 'test'
  test_labels_mapping = cls_mapping
  test_images = images
  test_labels = cls
  test_filenames = filenames
else : 
  # Splite of images(=images) and images lables (=cls) in two sets (train set and test set)
  train_images, test_images, train_labels, test_labels, train_labels_mapping, test_labels_mapping, train_filenames, test_filenames = train_test_split(images, cls, cls_mapping, filenames, test_size=0.20, random_state=42)            # test_size=0.25 = pourcentage de test_size
#*************************** switch2_end **************************************



# Optimizer
###############################################################################
  
opt = tf.keras.optimizers.Adam()



### Network
#*************************** switch3_start ************************************
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
#*************************** switch3_end **************************************
model.count_params()
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.summary()



# Load save of CNN models 
###############################################################################

if os.path.isfile('./CNN_solo/' + dossier_save + '/cp-' + checkpoint + '.ckpt') == False :
  sys.exit('Error : "' + './CNN_solo/' + dossier_save + '/cp-' + checkpoint + '.ckpt' + 'is not found! Test stop.') 

model.load_weights('./CNN_solo/' + dossier_save + '/cp-' + checkpoint + '.ckpt') 



# Summary
###############################################################################

loss,acc = model.evaluate(test_images, test_labels_mapping)
# "sonic" is the list of test_lables predicted by the model => if you do not save the labels in this list, the script is slow because it recalculates them several times 
sonic=(model.predict_classes(test_images))[0:test_images.shape[0]] 
success_images  = 0
fail_images = 0  
for b in range (test_images.shape[0]):
  if test_labels_mapping[b] == sonic[b]:
    success_images =  success_images  +1
  else:
    fail_images = fail_images +1
    
p_success= round(success_images/len(test_labels)*100,1)
p_fail=round(fail_images/len(test_labels)*100,1)
    
print(" ")  
print('Pollen successuffully classified  : {0}% ({1}/{2})'.format(p_success, success_images,len(test_labels)))
print('Pollen misclassified : {0}% ({1}/{2})'.format(p_fail, fail_images,len(test_labels)))



# Plot of images and histogrames
###############################################################################

predictions = model.predict(test_images)
num_class = len(labels_mapping)

# Configuration of plot
num_rows = num_rows_plot                                   
num_cols = num_cols_plot                                    
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
test_images2d=test_images[:,:,:,0]               # Removal of the 3rd dimension "channels" of the images because not to be managed by the plot
a=0 
i=0
while a < num_images:
  if i< test_images.shape[0]:                    # here "if" function is security for the "loob for" if there are too many test images
    if eval(condition):
      plt.subplot(num_rows, 2*num_cols, 2*a+1)
      plot_image(i, predictions, test_labels_mapping, test_images2d)
      plt.subplot(num_rows, 2*num_cols, 2*a+2)
      plot_value_array(i, predictions, test_labels_mapping)
    else :
      a=a-1   
  else:
    break
  i=i+1
  a=a+1
plt.show()



# Confusion matrix
###############################################################################

y_pred = sonic   # y perd = predit labels 
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_labels_mapping, y_pred, labels=range(num_classes))   #"lables=" List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in y_true or y_pred are used in sorted order.
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, labels_mapping, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, labels_mapping, normalize=True, title='Normalized confusion matrix')
plt.show()