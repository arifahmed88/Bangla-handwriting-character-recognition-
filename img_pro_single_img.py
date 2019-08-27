import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
import tensorflow as tf
from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
from PIL import Image
from six.moves import range
# standard dimensions to which all images will be rescaled
dimensions = (50, 50)
image_size = 50  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.
# function to turn grey-colored backgrounds to white. r, b and g specify the
# exact shade of grey color to eliminate. Source: stackoverflow.
def make_greyscale_white_bg(im, r, b, g):

    im = im.convert('RGBA')   # Convert to RGBA


    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace grey with white... (leaves alpha values alone...)
    grey_areas = (red == r) & (blue == b) & (green == g)
    data[..., :-1][grey_areas.T] = (255, 255, 255) # Transpose back needed

    im2 = Image.fromarray(data)
    im2 = im2.convert('L')   # convert to greyscale image


    return im2


# function to invert colors (black -> white and white-> black). Since most of the image consists
# of white areas, specified by (255, 255, 255) in RGB, inverting the colors means more zeros, making
# future operations less computationally expensive

def invert_colors(im):

    im = im.convert('RGBA')   # Convert to RGBA
    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability


    # Replace black with red temporarily... (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[..., :-1][black_areas.T] = (255, 0, 0) # Transpose back needed

    # Replace white areas with black
    white_areas = (red == 255) & (blue == 255) & (green == 255)
    data[..., :-1][white_areas.T] = (0, 0, 0) # Transpose back needed

    # Replace red areas (originally white) with black
    red_areas = (red == 255) & (blue == 0) & (green == 0)
    data[..., :-1][red_areas.T] = (255, 255, 255) # Transpose back needed

    im2 = Image.fromarray(data)
    im2 = im2.convert('L')   # convert to greyscale image


    return im2


# function for processing a specified image
def process_single_image(image_path):
    image_file = Image.open(image_path)
    image_file = image_file.resize(dimensions)
    image_file = make_greyscale_white_bg(image_file, 127, 127, 127)
    image_file = invert_colors(image_file)
    image_file.save(image_path)

    try:
        image_data = (ndimage.imread(image_path).astype(float) -  # normalize data
                      pixel_depth / 2) / pixel_depth
        if image_data.shape != (image_size, image_size):
            raise Exception('Unexpected image shape: %s' % str(image_data.shape))
    except IOError as e:
        print('Could not read:', image_path, ':', e, '- it\'s ok, skipping.')  # skip unreadable files

    try:
        with open('sample.pickle', 'wb') as f:
            pickle.dump(image_data, f, pickle.HIGHEST_PROTOCOL)
            f.close()
    except Exception as e:
        print('Unable to save data to sample.pickle:', e)


test_image_path= "sample.bmp"
process_single_image(test_image_path)
