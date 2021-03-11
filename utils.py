'''
Description:
    Utils files includes all the necessary things for the training of the rot-net model
'''

from __future__ import division
import math, random, cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import Iterator
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K

def data_gen(data, batch_size = 32, shape = 224, angles = [0,90,180, 270]):
    '''
    Description:
        This function is a generator you have just to pass your images rotated at 0 degree to this function.
        this fucntion will automatically rotate the image at 4 different angles i.e [0, 90, 180 , 270] and
        assign the respective labelt it. 
        
    Input:
        data         (list): List of images paths (str)
        batch_size   (int) : Batch size according to your ram capicity
        image shape  (int) : Shape of the image in which actucal image is resized to
                             feed it to the model.
        angles       (list): List of angles on which you want to train your model
                             defualt are 4 angles on which model trains (90, 180, 270, 0)
    Output:
        Batch_x    (np_array)  : Numpy array of images 
        Batch_y    (np_array)  : Numpy array of respected labels on same indexes
    '''
    batch_x, batch_y = [], []
    while(True):
        # Ramdomly choseing Image
        rand_image = random.randint(0, len(data)-1)
        image = data[rand_image]
        # Randomly chosing rotation angle
        angle      = random.choice(angles)
        # Rotating Resizing and appending image to the array
        image = Image.open(image).convert('RGB').rotate(angle, expand = True).resize((shape,shape))
        batch_x.append(np.asarray(image))
        # converting to  categorical labels
        cat_label = [0] * len(angles)
        cat_label[angles.index(angle)] = 1
        batch_y.append(cat_label)
        # If number of images processed are equal to the batchsize than returning batches of the image
        if len(batch_x) % batch_size ==0 and len(batch_x) > 0:
            yield np.asarray(batch_x), np.asarray(batch_y)
            del batch_x, batch_y
            batch_x, batch_y = [], []
