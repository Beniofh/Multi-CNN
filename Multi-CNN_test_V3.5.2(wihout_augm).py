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
from sklearn import svm, datasets       
from sklearn.model_selection import train_test_split
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



# Size of images (resolution 64 or 128) (switch_0)
###############################################################################

resolution = 128



# Select our dataset : 'intact' or 'damaged' or 'fossil'
###############################################################################

# 'intact'-> is the test set of the modern intact dataset (part of dataset not used in training)
# 'damaged'-> modern damaged dataset 
# 'fossil" -> fossil dataset 
switch_1='fossil'                                                                 

# Setting for 'train_test_split'                            
# For use the same test set for the modern intact dataset as in the training  
size_test_size = 0.20       
seed_for_random = 42        



# Setting for Step 1 and Step 2 (simple CNN)
###############################################################################

# Nomber of CNN
sum_cnn_trie_1 = 3

# (not used in our paper) Classes of training : For not use (presence = False) or use (presence = True) and definition of 'classes_select'  
presence= False
classes_select = []

# localisation of model saves
# is for 1.1 
dossier_save_1_1 = 'auto_save(amar+mono, 128, off, off)'                                                                
checkpoint_1_1 = '0204'                                                 
# is for 2.1 
dossier_save_2_1 = 'auto_save(amar+p+c, 128, off, base)'                                                      
checkpoint_2_1 = '0212'                                                 
# is for 2.2 
dossier_save_2_2 = 'auto_save(amar+p+c, 128, on, full)'                                                             
checkpoint_2_2 = '0385' 

# Mapping
# mapping = {'nom du groupe 1' : [class1, classe2,...],'nom du groupe 2' : [...)  
mapping_1_1 =  {'amar': ['aa', 'aj', 'al', 'ca', 'co', 'dm'], 'mono': ['c' , 'p']}  
mapping_2_1 =  {'amar': ['aa', 'aj', 'al', 'ca', 'co', 'dm'], 'c': ['c'], 'p' : ['p']}
mapping_2_2 =  {'amar': ['aa', 'aj', 'al', 'ca', 'co', 'dm'], 'c': ['c'], 'p' : ['p']} 

                                                
 
# Setting for Step 3A and Step 3B (composite CNN) 
###############################################################################

# Localisation of model saves : indicate the number of the epoch for each CNN            
# is for 3A.1    
dossier_save_aa = "auto_save(aa+other_amar, 128, off, off)"                                                                 
checkpoint_aa = "0064" 
# is for 3A.2 
dossier_save_aj ="auto_save(aj+other_amar, 128, off, off)"                                                                 
checkpoint_aj = "0079"     
# is for 3A.3 
dossier_save_al ="auto_save(al+other_amar, 128, off, off)"                     
checkpoint_al = "0122"     
# is for 3B.1  
dossier_save_c ="auto_save(c+other_mono, 128, off, off)"                    
checkpoint_c  = "0044" 
# is for 3A.4 
dossier_save_ca ="auto_save(ca+other_amar, 128, off, off)"                     
checkpoint_ca = "0190"     
# is for 3A.5
dossier_save_co ="auto_save(co+other_amar, 128, off, off)"                    
checkpoint_co = "0093"      
# is for 3A.6 
dossier_save_dm ="auto_save(dm+other_amar, 128, off, off)"                     
checkpoint_dm = "0056"     
# is for 3B.2 
dossier_save_p ="auto_save(p+other_mono, 128, on, full)"                     
checkpoint_p  = "0433" 

# Use or not (on/off) the plot with les polle images with weight and labels  
switch_2_1='on'                                             #'on' ou 'off'
condition1 = 'predict_val_x[a4] != test_true_val_x[a4]'                    
num_rows_b = 50                                             # Number of lines in the figure
num_cols_b = 3                                              # Number of columns in the figure

# Use confision matrix without normalization
switch_2_2a='on'
# Use confision matrix with normalization
switch_2_2b='on'


# Setting for Step 4 (Synthesis) 
###############################################################################

# Use or not (on/off) the plot with les polle images with labels  
switch_3_1='on'                                                   #'on' ou 'off'
condition2 = 'predict_val_final[a2] != test_true_val_final[a2]'                       
num_rows_a = 50                                                   # Number of lines in the figure               
num_cols_a = 6                                                    # Number of columns in the figure

# Use confision matrix without normalization
switch_3_2a='on'
# Use confision matrix with normalization
switch_3_2b='on'



###############################################################################
################################ Definition ###################################
###############################################################################



# Definition of process_image
def process_image(filename, width):
  im = Image.open(filename).convert('L')                         # open and convert to greyscale
  im = np.asarray(im, dtype=np.float)                            # numpy array
  im_shape = im.shape
  im = resize(im, [width, width], order=1 , mode='reflect')      # resize using linear interpolation, replace mode='reflect' by 'constant' otherwise error message 
  im = np.divide(im, 255)                                        # divide to put in range [0 - 1]
  return im, im_shape

# Definition of image loading
def load_from_class_dirs(directory, extension, width, norm, min_count=20):
  print(" ")
  print("Loading images from the directory './" + directory + "'.")
  print("########################################################")
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

# Definition of CNN model for image size = 64 pixels
def CNN_model_64(i19, i22) :
    i19.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
    i19.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    i19.add(tf.keras.layers.MaxPooling2D())
    i19.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    i19.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    i19.add(tf.keras.layers.MaxPooling2D())
    i19.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    i19.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    i19.add(tf.keras.layers.MaxPooling2D())
    i19.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    i19.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    i19.add(tf.keras.layers.MaxPooling2D())
    i19.add(tf.keras.layers.Flatten())
    i19.add(tf.keras.layers.Dropout(0.5,seed=7))
    i19.add(tf.keras.layers.Dense(512, activation='relu'))
    i19.add(tf.keras.layers.Dense(i22, activation='softmax'))
    i19.count_params()
    i19.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #i19.summary()
  
# Definition of CNN model for image size = 128 pixels
def CNN_model_128(i20,i21) :
    i20.add(tf.keras.layers.Conv2D(16, (3,3), input_shape=(resolution, resolution, 1), activation='relu', padding='same'))
    i20.add(tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same'))
    i20.add(tf.keras.layers.MaxPooling2D())
    i20.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    i20.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
    i20.add(tf.keras.layers.MaxPooling2D())
    i20.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    i20.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
    i20.add(tf.keras.layers.MaxPooling2D())
    i20.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    i20.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
    i20.add(tf.keras.layers.MaxPooling2D())
    i20.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for image by 128 pixel
    i20.add(tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'))  # additional layer for image by 128 pixel
    i20.add(tf.keras.layers.MaxPooling2D())                                         # additional layer for image by 128 pixel
    i20.add(tf.keras.layers.Flatten())
    i20.add(tf.keras.layers.Dropout(0.5,seed=7))
    i20.add(tf.keras.layers.Dense(512, activation='relu'))
    i20.add(tf.keras.layers.Dense(i21, activation='softmax'))
    i20.count_params()
    i20.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    #i20.summary()

# Definition of image plots for step 3A and 3B  (/!\ used for definition of plot_value_array)
def plot_image(i13):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(test_image_x_2d[i13], cmap=plt.cm.binary)
  if predict_val_x[i13] == test_true_val_x[i13]:
    color = 'blue'
  elif predict_val_x[i13] == 'undiff':
    color = 'green'
  elif predict_val_x[i13] == 'uncertain':
    color = 'darkviolet'
  else :
    color = 'red'
  plt.xlabel("{} ({})".format(predict_val_x[i13], test_true_val_x[i13]), color=color)
  plt.ylabel(test_filename_x[i13][0:len(test_filename_x[i13])-4])

# Definition of histogram plots for step 3A and 3B
def plot_value_array(i14, classes, certitude_predictions):
  plt.grid(False)
  plt.xticks(range(len(classes)), classes, rotation=0)   # range(nomber of classe)
  plt.yticks([])
  thisplot = plt.bar(range(len(classes)), certitude_predictions[i14], color="#777777")      # range(nomber of classe)
  plt.ylim([0, 1])  
  maxi = attributed_classes_certitude_x[i14].index(np.max(attributed_classes_certitude_x[i14]))  # gives for an image, the number of the class with the most important certainty value
  thisplot[maxi].set_color('blue')

# Definition of confusion matrix for step 1 and 2
#This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
def plot_confusion_matrix_solo(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks_x = np.arange(len(classes))
  tick_marks_y = np.arange(len(classes))
  plt.xticks(tick_marks_x, classes, rotation=45)
  plt.yticks(tick_marks_y, classes)
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i15, i16 in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(i16, i15, format(cm[i15, i16], fmt),horizontalalignment="center", color="white" if cm[i15, i16] > thresh else "black")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

# Definition of confusion matrix for step 3A and 3B
#This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
def plot_confusion_matrix_serie(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks_x = np.arange(len(classes)-a6)
  tick_marks_y = np.arange(len(classes)-2)
  if (-1 in test_true_num_x_pure) :                                             # add the column for 'mono' if there are any in the 'amar' or for 'amar' if there are any in the 'mono'
    plt.xticks(tick_marks_x, classes[1:], rotation=45)                          # classes[1:] -> all values except the first one
  else : 
    plt.xticks(tick_marks_x, classes, rotation=45)   
  plt.yticks(tick_marks_y, classes)                                        
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i17, i18 in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(i18, i17, format(cm[i17, i18], fmt),horizontalalignment="center", color="white" if cm[i17, i18] > thresh else "black")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()

# Definition of image plots for step 4 
def plot_image_synthese(i13):
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  plt.imshow(test_image_final_2d[i13], cmap=plt.cm.binary)
  if predict_val_final[i13] == test_true_val_final[i13]:
    color = 'blue'
  elif predict_val_final[i13] == 'indet' :
    color = 'green'
  elif predict_val_final[i13] == 'amar.indet' and test_true_val_final[i13] in mapping['amar']:
    color = 'green'    
  elif predict_val_final[i13] == 'mono.indet' and test_true_val_final[i13] in mapping['mono']:
    color = 'green'
  else :
    color = 'red'
  plt.xlabel("{} ({})".format(predict_val_final[i13], test_true_val_final[i13]), color=color)
  plt.ylabel(test_filename_final[i13][0:len(test_filename_final[i13])-4])

# Definition of confusion matrix for step 4 
# This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`
def plot_confusion_matrix_synthese (cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks_x = np.arange(len(classes))
  tick_marks_y = np.arange(len(classes)-3)
  plt.xticks(tick_marks_x, classes, rotation=90)   
  plt.yticks(tick_marks_y, classes)                                        
  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i17, i18 in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(i18, i17, format(cm[i17, i18], fmt),horizontalalignment="center", color="white" if cm[i17, i18] > thresh else "black")
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.tight_layout()  



###############################################################################
########################### Main scripte ######################################
###############################################################################



# Extration and converstion of images in folder "images"
###############################################################################

#*************************** switch_1_start ***********************************
if switch_1=='intact':
  path_folder_image = "intact"
else : 
  path_folder_image = switch_1
#*************************** switch_1_end  ************************************
# load_from_class_dirs("name of file", "extention type of images", "image size (in pixel)",...)
images, cls, labels, num_classes, filenames = load_from_class_dirs(path_folder_image, "tif", resolution, False, min_count=0)  
# image dimensions must be [batch, width, height, channels]. But the images are "single channel" because "numpy" don't include the dimension "channel", we add this dimention with "np.newaxis"
images = images[:,:,:,np.newaxis]
# true_labels -> convertion of cls in text with labels
true_labels=[]                        
for j1 in cls :
  true_labels.append(labels[j1])



# Step 1 and 2 
###############################################################################

print("")
print("")
print("Initialization phases 1 and 2")
print("#############################")

for j2 in range(sum_cnn_trie_1):
  if j2 == 0 :
    mapping = mapping_1_1
    dossier_save = dossier_save_1_1
    checkpoint= checkpoint_1_1
  elif j2==1 :
    mapping = mapping_2_1
    dossier_save = dossier_save_2_1
    checkpoint= checkpoint_2_1 
  else :
    mapping = mapping_2_2
    dossier_save = dossier_save_2_2
    checkpoint= checkpoint_2_2       

    
  # Group creation with the 'mapping option'
  print("")
  print("# Phases 1 et 2 : {0}/{1}".format(j2+1,sum_cnn_trie_1))
  print("")
  print("Mapping : {0}".format(mapping))
  labels_mapping=[]
  mapping_security=[]
  cls_mapping = copy(cls)
  for key in mapping.keys():
    labels_mapping.append(key)
    for val in mapping[key]:
      mapping_security.append(val)  
      for j3 in range(len(cls)):
        if cls[j3]==labels.index(val):
          cls_mapping[j3]=(labels_mapping.index(key))
  for security_1 in labels :
    if (security_1 in mapping_security) == False :
      sys.exit('Error : "' + security_1 + '" is not found in "mapping"! Test stop.')           


  # Variable attribution 
  #*************************** switch_1_start *********************************
  if switch_1=='damaged' or switch_1=='fossil':
    # Attribution of 'test' variables 
    test_images = images
    test_labels = cls_mapping
    test_true_num = cls
    test_true_val = true_labels
    test_filenames = filenames
  else : 
    # Splite of images(=images) and images lables (=cls) in two sets (train set and test set)
    train_images, test_images, train_labels, test_labels, train_true_num, test_true_num, train_true_val, test_true_val, train_filenames, test_filenames = train_test_split(images, cls_mapping, cls, true_labels, filenames, test_size=0.20, random_state=42)            # test_size=0.25 = pourcentage de test_size
  #*************************** switch_1_end ***********************************
  
  # Optimizer
  opt = tf.keras.optimizers.Adam()

  # CNN Model
  #*************************** switch_0_start *********************************
  if resolution == 64 :
    model = tf.keras.models.Sequential()
    CNN_model_64(model, len(labels_mapping))
  else :
    model = tf.keras.models.Sequential()
    CNN_model_128(model, len(labels_mapping))
  #*************************** switch_0_end *********************************** 
  # Checked that the file to be loaded exists 
  if os.path.isfile('./CNN_solo/' + dossier_save + '/cp-' + checkpoint + '.ckpt') == False :
    sys.exit('Error : "' + './CNN_solo/' + dossier_save + '/cp-' + checkpoint + '.ckpt' + 'is not found! Test stop.') 
  # Load save of CNN models 
  model.load_weights('./CNN_solo/' + dossier_save + '/cp-' + checkpoint + '.ckpt')

  # Summary of CNN 1.1, 2.1 and 2.2
  loss,acc = model.evaluate(test_images, test_labels)
  # "predict_num_1" is the list of test_lables predicted by the model => if you do not save the labels in this list, the script is slow because it recalculates them several times 
  predict_num_1=(model.predict_classes(test_images))[0:test_images.shape[0]] 
  success_images  = 0
  fail_images = 0  
  for j4 in range (test_images.shape[0]):
    if test_labels[j4] == predict_num_1[j4]:
      success_images =  success_images  +1
    else:
      fail_images = fail_images +1
  
  p_success= round(success_images/len(test_labels)*100,1)
  p_fail=round(fail_images/len(test_labels)*100,1)
    
  print(" ")  
  print('Pollen successuffully classified  : {0}% ({1}/{2})'.format(p_success, success_images,len(test_labels)))
  print('Pollen misclassified : {0}% ({1}/{2})'.format(p_fail, fail_images,len(test_labels)))

  # Compute confusion matrix
  cnf_matrix = confusion_matrix(test_labels, predict_num_1)
  np.set_printoptions(precision=2)
  # Plot non-normalized confusion matrix
  plt.figure()
  plot_confusion_matrix_solo(cnf_matrix, labels_mapping, title='Confusion matrix, without normalization')
  plt.show()

  if j2 == 0 :
    labels_mapping_1_1 = copy(labels_mapping)
    test_labels_1_1 = copy(test_labels)            # -> 0=amar; 1=mono
    predict_num_1_1 = copy(predict_num_1)          # -> 0=amar; 1=mono
  elif j2== 1 :
    labels_mapping_2_1 = copy(labels_mapping)
    test_labels_2_1 = copy(test_labels)            # -> 0=amar; 1=Cyperaceae; 2=Poaceae
    predict_num_2_1 = copy(predict_num_1)          # -> 0=amar; 1=Cyperaceae; 2=Poaceae
  else :
    labels_mapping_2_2 = copy(labels_mapping)
    test_labels_2_2 = copy(test_labels)            # -> 0=amar; 1=Cyperaceae; 2=Poaceae
    predict_num_2_2 = copy(predict_num_1)          # -> 0=amar; 1=Cyperaceae; 2=Poaceae  



#Creation of groups for steps 3A, 3B and 4 
###############################################################################

test_image_amar=[]
test_true_num_amar=[]
test_true_val_amar=[]
test_filename_amar=[]
predict_num_1_1_amar=[]
predict_num_2_1_amar=[]
predict_num_2_2_amar=[]
test_image_mono=[]
test_true_num_mono=[]
test_true_val_mono=[]
test_filename_mono=[]
predict_num_1_1_mono=[]
predict_num_2_1_mono=[]
predict_num_2_2_mono=[]

for j5 in range (test_images.shape[0]):
  if (predict_num_1_1[j5] == 0) or (predict_num_2_1[j5] == 0) or (predict_num_2_2[j5] == 0):
    test_image_amar.append(test_images[j5])
    test_true_num_amar.append(test_true_num[j5])
    test_true_val_amar.append(test_true_val[j5])
    test_filename_amar.append(test_filenames[j5])
    predict_num_1_1_amar.append(predict_num_1_1[j5])
    predict_num_2_1_amar.append(predict_num_2_1[j5])
    predict_num_2_2_amar.append(predict_num_2_2[j5])
  else:
    test_image_mono.append(test_images[j5])
    test_true_num_mono.append(test_true_num[j5])
    test_true_val_mono.append(test_true_val[j5])
    test_filename_mono.append(test_filenames[j5])
    predict_num_1_1_mono.append(predict_num_1_1[j5])      
    predict_num_2_1_mono.append(predict_num_2_1[j5])  
    predict_num_2_2_mono.append(predict_num_2_2[j5])  
    
test_image_amar=np.asarray(test_image_amar)    
test_image_mono=np.asarray(test_image_mono)   



# Steps 3A and 3B
###############################################################################

print("")
print("")
print("Initialization phase 3A")
print("#######################")
print("")

mapping = mapping_1_1
labels_mapping = copy(labels_mapping_1_1)

for j6 in range(2):
  if j6==0:
    test_image_x = test_image_amar
    test_true_num_x = test_true_num_amar
    test_true_val_x = test_true_val_amar
    test_filename_x = test_filename_amar
    key_mapping_actif='amar'
    key_mapping_passif='mono'
  else :
    print("")
    print("")
    print("Initialization phase 3B")
    print("#######################")
    print("")
    test_image_x = test_image_mono
    test_true_num_x = test_true_num_mono
    test_true_val_x = test_true_val_mono
    test_filename_x = test_filename_mono
    key_mapping_actif='mono'
    key_mapping_passif='amar'

  # Setting up memories                
  attributed_classes_certitude_x = []        # gives for each image, for all labels, the  % of certainty of model if one basic classe is allocated (otherwise it is 0) 
  attributed_classes_val_x = []              # gives for each image, for all labels, the class assigned to it atribue (='Indet' if it is not one of the basic classes) 
  predict_val_x = []                         # gives the label assigned to each image (in text lable) : len(labels)=undiff et len(labels)+1=uncertain)
  predict_num_x = []                         # gives the label assigned to each image (in number lable)
  attributed_success_x = []                  # dit si le label attribue est fail, succes, undiff ou uncertain
  for j7 in test_true_num_x :
    attributed_classes_certitude_x.append([])
    attributed_classes_val_x.append([]) 
    predict_val_x.append('NA')
    predict_num_x.append('NA')
    attributed_success_x.append('NA') 

  # Loop for CNN recognition in step 3A
  t1=1
  for j8 in labels:
    if (j8 in mapping[key_mapping_passif]) == False:     
      # Binarisation de cls avec 0 = j8 et 1 = 'undiff'
      labels_binaire=[]
      cls_binaire=[]
      labels_binaire.extend([j8, 'undiff'])
      for i9 in test_true_num_x:
        if (labels[i9] in mapping[key_mapping_passif]) == True:
          cls_binaire.append(1)                                       
        else :  
          if i9==labels.index(j8):
            cls_binaire.append(0)
          else :
            cls_binaire.append(1)       
      # Optimizer
      opt = tf.keras.optimizers.Adam()
      #*************************** switch_0_start *****************************
      model = tf.keras.models.Sequential()
      if resolution == 64 :
        CNN_model_64(model, len(labels_mapping))
      else :
        CNN_model_128(model, len(labels_mapping))
      #*************************** switch_0_end *******************************
      # Checked that the file to be loaded exists 
      if os.path.isfile("./CNN_solo/"+ eval("dossier_save_"+j8) + "/cp-" + eval("checkpoint_"+j8) + ".ckpt") == False :
        sys.exit('Error : "' + "./CNN_solo/"+ eval("dossier_save_"+j8) + "/cp-" + eval("checkpoint_"+j8) + ".ckpt" + 'is not found! Test stop.') 
      # Load save of CNN models     
      model.load_weights("./CNN_solo/"+ eval("dossier_save_"+j8) + "/cp-" + eval("checkpoint_"+j8) + ".ckpt") 
      # Save labels et nomber of assigned classes in the memories
      t3=0
      for j10 in (model.predict_classes(test_image_x)):
        if j10==0:
            attributed_classes_certitude_x[t3].append((model.predict(test_image_x[t3:t3+1]))[0,0])    #[x,y] -> value of row X of column y in the matrix (model.predict(test_image_x[t3:t3+1])
            attributed_classes_val_x[t3].append(j8)
        else:
            attributed_classes_certitude_x[t3].append(0)
            attributed_classes_val_x[t3].append('indet')    
        t3=t3+1
      print('Recognition of ' + j8 + ' completed. Phase finished at {0}/{1}'.format(t1,len(mapping[key_mapping_actif])))
      t1=t1+1    
  print('Phase completed.') 

  # For not shift in the values for the success test
  test_true_num_x_pure=[]
  for j9 in test_true_val_x:
    if (j9 in mapping[key_mapping_passif]) == True:
      test_true_num_x_pure.append(-1)
    else :
      test_true_num_x_pure.append(mapping[key_mapping_actif].index(j9))  
    
  # Filling memory lists
  success_images  = 0
  fail_images = 0
  undiff_images = 0
  uncertain_image = 0
  t4=0
  for j11 in attributed_classes_val_x:
    if j11.count('indet')==len(mapping[key_mapping_actif])-1:
      predict_val_x [t4] = mapping[key_mapping_actif][attributed_classes_certitude_x[t4].index(np.max(attributed_classes_certitude_x[t4]))]  
      predict_num_x [t4] = attributed_classes_certitude_x[t4].index(np.max(attributed_classes_certitude_x[t4]))
      if attributed_classes_certitude_x[t4].index(np.max(attributed_classes_certitude_x[t4]))==test_true_num_x_pure[t4]:  
        success_images = success_images + 1
        attributed_success_x [t4] = 'success' 
      else:
        fail_images = fail_images + 1
        attributed_success_x [t4] = 'fail'
    elif j11.count('indet')==len(mapping[key_mapping_actif]) :   
      predict_val_x [t4] = 'undiff'
      predict_num_x [t4] = len(mapping[key_mapping_actif])
      undiff_images = undiff_images + 1
      attributed_success_x [t4] = 'undiff'
    else :   
      predict_val_x [t4] = 'uncertain'
      predict_num_x [t4] = len(mapping[key_mapping_actif])+1
      uncertain_image = uncertain_image + 1
      attributed_success_x [t4] = 'uncertain'
    t4=t4+1

  # Retrieving memory lists  

  if j6==0:
    attributed_classes_certitude_amar = attributed_classes_certitude_x
    attributed_classes_val_amar = attributed_classes_val_x
    predict_val_amar = predict_val_x
    predict_num_amar = predict_num_x
    attributed_success_amar = attributed_success_x
  else :
    attributed_classes_certitude_mono = attributed_classes_certitude_x
    attributed_classes_val_mono = attributed_classes_val_x
    predict_val_mono = predict_val_x
    predict_num_mono = predict_num_x
    attributed_success_mono = attributed_success_x

  # Summary of step 3A or 3B

  p_success2= round(attributed_success_x.count('success')/(len(test_true_num_x))*100,1)
  p_indet2=round((attributed_success_x.count('undiff') + attributed_success_x.count('uncertain'))/(len(test_true_num_x))*100,1)
  p_fail2=round(attributed_success_x.count('fail')/(len(test_true_num_x))*100,1)
    
  print(" ")  
  print('Pollen successuffully classified  : {0}% ({1}/{2})'.format(p_success2, attributed_success_x.count('success'),len(test_true_num_x)))
  print('Pollen  classified in indet. : {0}% ({1}/{2})'.format(p_indet2, (attributed_success_x.count('undiff') + attributed_success_x.count('uncertain')),len(test_true_num_x)))
  print('Pollen misclassified : {0}% ({1}/{2})'.format(p_fail2, attributed_success_x.count('fail'), len(test_true_num_x)))



  # Plot of images with weight histograms
  #***************************** switch_2_1_start******************************
  if switch_2_1=='on':
    # Configuration of plot
    num_images = num_rows_b*num_cols_b
    plt.figure(figsize=(2*2*num_cols_b, 2*num_rows_b))
    test_image_x_2d=test_image_x[:,:,:,0]                 # Removal of the 3rd dimension "channels" of the images because not to be managed by the plot
    a3=0 
    a4=0
    while a3 < num_images:
      if a4< test_image_x.shape[0]:                       # here "if" function is security for the "loob for" if there are too many test images
        if eval(condition1) :
          plt.subplot(num_rows_b, 2*num_cols_b, 2*a3+1)
          plot_image(a4)
          plt.subplot(num_rows_b, 2*num_cols_b, 2*a3+2)
          plot_value_array(a4,mapping[key_mapping_actif],attributed_classes_certitude_x)
        else :
          a3=a3-1   
      else:
        break
      a4=a4+1
      a3=a3+1
    plt.show()
  #***************************** switch_2_1_end *******************************

  # Confusion matrix
  #***********************switch_2_2a + switch_2_2b_start *********************
  if switch_2_2a or switch_2_2b =='on':
    classifier = svm.SVC(kernel='linear', C=0.01)                                         
    # Compute confusion matrix and lables of matrix
    labels2 = []
    a6=0
    if (-1 in test_true_num_x_pure) :              # add row/column for 'mono' if there are any in the 'amar' or for 'amar' if there are any in the 'mono'
      labels2.append(key_mapping_passif)
      a6=1
    labels2.extend((mapping[key_mapping_actif]))
    labels2.append('undiff')
    labels2.append('uncertain')
    #confusion_matix(true labels, predict labels)
    #"lables=" List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in test_labels_mapping or y_pred are used in sorted order.
    #here we use "labels=range(-a6,num_classes+2)" and not labels=range(num_classes) because we have 2 supllementary predict class (undiff, uncertain) in addition of x mapping actif classes, -a6 because we have sometime one classe "-1". 
    cnf_matrix = confusion_matrix(test_true_num_x_pure , predict_num_x, labels=range(-a6, len(mapping[key_mapping_actif])+2))    #confusion_matix(true labels, predict labels)
    cnf_matrix2=cnf_matrix[0:-2, a6:] 

    np.set_printoptions(precision=2)  
    # Plot non-normalized confusion matrix
    #************************** switch_2_2a_start *****************************
    if switch_2_2a=='on' :
      plt.figure()
      plot_confusion_matrix_serie(cnf_matrix2, labels2, title='Confusion matrix, without normalization')
    #************************** switch_2_2a_end *******************************
    # Plot normalized confusion matrix
    #************************** switch_2_2b_start *****************************
    if switch_2_2b=='on' :
      plt.figure()
      plot_confusion_matrix_serie(cnf_matrix2, labels2, normalize=True, title='Normalized confusion matrix') 
    #************************** switch_2_2b_end *******************************
    plt.show()
  #***********************switch_2_2a + switch_2_2b_end ***********************
  
  

# Note : here are all the lists available to summarize all last results
###############################################################################
  
#test_image_amar
#test_true_num_amar
#test_true_val_amar
#test_filename_amar
#test_image_mono
#test_true_num_mono
#test_true_val_mono
#test_filename_mono 
#attributed_classes_certitude_amar 
#attributed_classes_val_amar
#predict_val_amar 
#predict_num_amar 
#attributed_success_amar 
#attributed_classes_certitude_mono 
#attributed_classes_val_mono 
#predict_val_mono 
#predict_num_mono 
#attributed_success_mono 



# Step 4
###############################################################################

print("")
print("")
print("Phase 4 : synthesis")
print("###################")
print("")

test_image_final=[]
test_filename_final=[]
test_true_num_final=[]
test_true_val_final=[]
predict_num_final=[]
predict_val_final=[]

# Bilan of step 3A
for j12 in range(len(predict_val_amar)):
  test_image_final.append(test_image_amar[j12])
  test_filename_final.append(test_filename_amar[j12])
  test_true_num_final.append(test_true_num_amar[j12])
  test_true_val_final.append(test_true_val_amar[j12]) 
  if predict_val_amar[j12] in labels :
    predict_num_final.append(labels.index(predict_val_amar[j12]))  # predict_val_amar[j12]) => to restore the original numbers lost during phase of CNN 1_1
    predict_val_final.append(predict_val_amar[j12])
#  elif predict_val_amar[j12] == "uncertain" : 
#    predict_num_final.append(len(labels))  
#    predict_val_final.append("amar.indet")
  else :    
    if predict_num_2_1_amar[j12]==predict_num_2_2_amar[j12] == 0 : # use the results of sorting CNN 2_2 : 0=amar; 1=cyp; 2=poa
      predict_num_final.append(len(labels))  
      predict_val_final.append("amar.indet")
    else :    
      predict_num_final.append(len(labels)+2)  
      predict_val_final.append("indet")

# Bilan of step 3B
for j13 in range(len(predict_val_mono)):
  test_image_final.append(test_image_mono[j13])
  test_filename_final.append(test_filename_mono[j13])
  test_true_num_final.append(test_true_num_mono[j13])
  test_true_val_final.append(test_true_val_mono[j13]) 
  if predict_val_mono[j13] in labels :
    predict_num_final.append(labels.index(predict_val_mono[j13]))  
    predict_val_final.append(predict_val_mono[j13])
#  elif predict_val_mono[j13] == "uncertain" : 
#    predict_num_final.append(len(labels)+1)  
#    predict_val_final.append("mono.indet")
  else :    
    if (predict_num_2_1_mono[j13]==predict_num_2_2_mono[j13]==1) or (predict_num_2_1_mono[j13]==predict_num_2_1_mono[j13]==2):                 #  pour 1_1 : 0=amar 1=mono ; pour 2_1 et 2_2 : 0=amar 1=cyp 2=poa
      predict_num_final.append(len(labels)+1)  
      predict_val_final.append("mono.indet")
    else :    
      predict_num_final.append(len(labels)+2)  
      predict_val_final.append("indet")      

test_image_final=np.asarray(test_image_final) # to finalize the new numpy image list   

# a copie dans v3.5.1------------------------------------------------------------------------------------

# Summary of step 4 (here we use "predict_val_final" and "test_true_val_final")
fianl_summay=[]
for j14 in predict_val_final :
  fianl_summay.append('NA')
for j15 in range(len(predict_val_final)):
  if predict_val_final[j15] == test_true_val_final[j15]:
    fianl_summay[j15] = "success_lowest_taxa"    
  elif predict_val_final[j15] ==  "amar.indet" and test_true_val_final[j15] in mapping["amar"]:
    fianl_summay[j15] = "success_amar_indet"
  elif predict_val_final[j15] ==  "mono.indet" and test_true_val_final[j15] in mapping["mono"]:
    fianl_summay[j15] = "success_mono_indet"
  elif predict_val_final[j15] ==  "indet":
    fianl_summay[j15] = "indet"
  else :
    fianl_summay[j15] = "misclassified"  

p_success_lowest_taxa = round(fianl_summay.count('success_lowest_taxa')/len(fianl_summay)*100,1) #pourcentage of success_lowest_taxa  
p_success_amar_indet = round(fianl_summay.count('success_amar_indet')/len(fianl_summay)*100,1) #pourcentage of success_lowest_taxa  
p_success_mono_indet = round(fianl_summay.count('success_mono_indet')/len(fianl_summay)*100,1) #pourcentage of success_lowest_taxa  
p_indet = round(fianl_summay.count('indet')/len(fianl_summay)*100,1) #pourcentage of success_lowest_taxa  
p_misclassified = round(fianl_summay.count('misclassified')/len(fianl_summay)*100,1) #pourcentage of success_lowest_taxa  

print(' ')
print('Pollen successuffully classified in lowest level : {0}% ({1}/{2})'.format(p_success_lowest_taxa, fianl_summay.count('success_lowest_taxa'), len(fianl_summay)))
print('Pollen successuffully classified in Amar. indet. : {0}% ({1}/{2})'.format(p_success_amar_indet, fianl_summay.count('success_amar_indet'),len(fianl_summay)))
print('Pollen successuffully classified in Mono. indet. : {0}% ({1}/{2})'.format(p_success_mono_indet, fianl_summay.count('success_mono_indet'),len(fianl_summay)))
print('Pollen  classified in indet. : {0}% ({1}/{2})'.format(p_indet, fianl_summay.count('indet'),len(fianl_summay)))
print('Pollen  misclassified : {0}% ({1}/{2})'.format(p_misclassified, fianl_summay.count('misclassified'),len(fianl_summay)))


#-----------------------------------------------------------------------------------

# Plot avec juste les photos et labels
#************************ switch_3_1_start ************************************
if switch_3_1=='on':
  # Configuration du plot
  num_images = num_rows_a*num_cols_a
  plt.figure(figsize=(2*num_cols_a, 2*num_rows_a))
  test_image_final_2d=test_image_final[:,:,:,0]       # Removal of the 3rd dimension "channels" of the images because not to be managed by the plot
  a1=0 
  a2=0
  while a1 < num_images:
    if a2< test_image_final.shape[0]:                 # here "if" function is security for the "loob for" if there are too many test images
      if eval(condition2) :
        plt.subplot(num_rows_a, num_cols_a,a1+1)
        plot_image_synthese(a2)
      else :
        a1=a1-1   
    else:
      break
    a2=a2+1
    a1=a1+1
  plt.show()
#************************ switch_3_1_end **************************************
  
# Confusion Matrix
#************************ switch_3_2a + switch_3_2b_start *********************


#-------------------------------------------------------------------------------------------------


if switch_3_2a or switch_3_2b =='on':
  classifier = svm.SVC(kernel='linear', C=0.01)                                       
  # Compute confusion matrix and lables of matrix
  labels3 = []
  labels3.extend(labels)
  labels3.append('amar.indet')
  labels3.append('mono.indet')
  labels3.append('indet')  
  #confusion_matix(true labels, predit labels )
  #"lables=" List of labels to index the matrix. This may be used to reorder or select a subset of labels. If none is given, those that appear at least once in test_labels_mapping or y_pred are used in sorted order.
  # here we use "labels=range(num_classes+3)" and not labels=range(num_classes+) because we have 3 supllementary predict class (amar.indet, mono.indet, indet) in addition of x input classes 
  cnf_matrix = confusion_matrix(test_true_num_final , predict_num_final,labels=range(num_classes+3))
  cnf_matrix2=cnf_matrix[0:-3,]                                      
  # Plot non-normalized confusion matrix
  #************************** switch_3_2a_start *******************************
  if switch_3_2a=='on' :
    plt.figure()
    plot_confusion_matrix_synthese(cnf_matrix2, labels3, title='Confusion matrix, without normalization')
  #************************** switch_3_2a_end *********************************
  # Plot normalized confusion matrix
  #************************** switch_3_2b_start *******************************
  if switch_3_2b=='on' :
    plt.figure()
    plot_confusion_matrix_synthese(cnf_matrix2, labels3, normalize=True, title='Normalized confusion matrix') 
  #************************** switch_3_2b_end *********************************
  plt.show()
#************************ switch_3_2a + switch_3_2b_end ***********************
  
  
#----------------------------------------------------------------------------------------------------  