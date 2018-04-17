import pickle
import hashlib
import os
import os.path
import re
import sys
import scipy
from scipy import ndimage
import cv2
import numpy as np
from numpy import *
import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

from IPython.core.display import Image 

TRAINING_IMAGES_DIR = os.getcwd() + '/training_images'
TEST_IMAGES_DIR = os.getcwd() + "/test_images/"
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
TrainingDataPickleFile = 'trainingdata/BikeTrainingData.pkl'
ValidationDataPickleFile = 'trainingdata/BikeValidationData.pkl'

#######################################################################################################################
def create_image_lists(image_dir, testing_percentage, validation_percentage):
    """
    Builds a list of training images from the file system.

    Analyzes the sub folders in the image directory, splits them into stable
    training, testing, and validation sets, and returns a data structure
    describing the lists of images for each label and their paths.

    Args:
        image_dir: String path to a folder containing subfolders of images.
        testing_percentage: Integer percentage of the images to reserve for tests.
        validation_percentage: Integer percentage of images reserved for validation.

    Returns:
        A dictionary containing an entry for each label subfolder, with images split
        into training, testing, and validation sets within each label.
    """
    if not gfile.Exists(image_dir):
        tf.logging.error("Image directory '" + image_dir + "' not found.")
        return None
    # end if

    result = {}
    sub_dirs = [x[0] for x in gfile.Walk(image_dir)]

    # The root directory comes first, so skip it.
    is_root_dir = True
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue
        # end if
        extensions = ['jpg', 'jpeg']#, 'JPG', 'JPEG'] # Windows is not case sensitive, so you get 2X training data!
        file_list = []
        dir_name = os.path.basename(sub_dir)
        if dir_name == image_dir:
            continue
        # end if
        tf.logging.info("Looking for images in '" + dir_name + "'")
        for extension in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
            file_list.extend(gfile.Glob(file_glob))
        # end for
        if not file_list:
            tf.logging.warning('No files found')
            continue
        # end if
        if len(file_list) < 20:
            tf.logging.warning('WARNING: Folder has less than 20 images, which may cause issues.')
        elif len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
            tf.logging.warning('WARNING: Folder {} has more than {} images. Some images will never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
        # end if
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)
            # We want to ignore anything after '_nohash_' in the file name when deciding which set to put an image in, the data set creator
            # has a way of grouping photos that are close variations of each other. For example this is used in the plant disease data set
            # to group multiple pictures of the same leaf.
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # This looks a bit magical, but we need to decide whether this file should go into the training, testing, or validation sets,
            # and we want to keep existing files in the same set even if more files are subsequently added.  To do that, we need a stable
            # way of deciding based on just the file name itself, so we do a hash of that and then use that to generate a probability value
            # that we use to assign it.
            hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) % (MAX_NUM_IMAGES_PER_CLASS + 1)) * (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            elif percentage_hash < (testing_percentage + validation_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)
            # end if
        result[label_name] = {'dir': dir_name, 'training': training_images, 'testing': testing_images, 'validation': validation_images,}
    return result
# end function

image_lists = create_image_lists(TRAINING_IMAGES_DIR, 0, 10)
class_count = len(image_lists.keys())
if class_count == 0:
    tf.logging.error('No valid folders of images found at ' + TRAINING_IMAGES_DIR)
if class_count == 1:
    tf.logging.error('Only one valid folder of images found at ' + TRAINING_IMAGES_DIR + ' - multiple classes are needed for classification.')

Batch = np.empty((1,300*400*3), float32)
Label = np.empty((1,1), int)
MBLabel = np.zeros((1,1))
RBLabel = np.ones((1,1))
for label in image_lists.keys():
    for filename in image_lists[label]["training"]:
        path = TRAINING_IMAGES_DIR + "/" + image_lists[label]["dir"] + "/" + filename
        im = cv2.imread(path)
        res = cv2.resize(im,(400, 300), interpolation = cv2.INTER_CUBIC) # Resize image to 400,300
        # !! 
        vector = res.reshape(-1, 300*400*3) # Flatten vector
        vector = vector / 255.0;             # Normalize
        Batch = np.append(Batch, vector, axis = 0)
        if(label == "mountain bikes"):
            Label = np.append(Label, MBLabel, axis = 0)
        if(label == "road bikes"):
            Label = np.append(Label, RBLabel, axis = 0)
            
Batch = np.delete(Batch, 0,0)
Label = np.delete(Label, 0,0)

print("X Training ", Batch.shape)
print("Y Training ", Label.shape)

with open(TrainingDataPickleFile, 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Batch, Label], f)

print("Training Data X, Y saved to ", TrainingDataPickleFile)