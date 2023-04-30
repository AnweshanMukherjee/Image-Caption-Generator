# Importing libraries
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from numpy import array
import pandas as pd
# import matplotlib.pyplot as plt
import string
import os
from PIL import Image, ImageOps
import glob
import pickle
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop, Adamax
#from keras.layers.wrappers import Bidirectional
from keras.layers import add
from tensorflow.keras.applications import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

st.set_page_config(
    page_title="Image Caption Generation using Neural Networks",
    page_icon="images/SXC_LOGO.png",
    layout="wide"
)

def local_css(file_name):
	with open(file_name) as f:
		st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

local_css("stylesheets/style.css")

st.image('images/SXC_LOGO.png', width=100)
st.title("Image Caption Generation using Neural Networks")

def get_caption(img):
  text = 'STARTSEQ '
  for i in range(max_length):
    sequence = [word_to_index[word] for word in text.split() if word in word_to_index]
    sequence = pad_sequences([sequence], maxlen = max_length)
    yhat = model.predict([img, sequence], verbose=0)
    yhat_pos = np.argmax(yhat)
    word = index_to_word[yhat_pos]
    text = text + ' ' + word
    if word == 'ENDSEQ':
      break
  
  caption = text.split()[1:-1]
  if (len(caption) > 11):
    caption = caption[:11]
  
  if (len(caption) > 1):
    if caption[-1] in ['a', 'the']:
      caption = caption[:-2]
    elif caption[-1] in ['of', 'and']:
      caption = caption[:-1]

  if (len(caption) > 1):
    if caption[0] == 'of':
      caption = caption[2:]

  caption = ' '.join(caption)

  return caption

model_img = InceptionV3(weights='imagenet')
model_img = Model(model_img.input, model_img.layers[-2].output) # modified model without the final layer of InceptionV3

def process_image(image_file):
  #size = (299, 299)    
  #image_file = ImageOps.fit(image_file, size, Image.ANTIALIAS)
  #image_file = image_file.convert('RGB')
  #image_file = np.asarray(image_file)
  # Convert image to size 299x299 for Inception V3 Model
  image = tf.keras.preprocessing.image.load_img(image_file, target_size=(299, 299))
  # img = image.load_img(image_file, target_size = (299, 299))
  # image_file = tf.image.resize(image_file, (299, 299))
  # Convert PIL image to numPy 3-d array
  x = tf.keras.preprocessing.image.img_to_array(image)
  # Add another dimension
  x = np.expand_dims(x, axis=0)
  # preprocess the images using preprocess_input() from inception module
  x1 = x.copy()
  x1 = preprocess_input(x)
  return x1

def encode_image(img):
  img = process_image(img)
  feature_vector = model_img.predict(img) # outputs a (1, 2048) vector
  feature_vector = np.reshape(feature_vector, feature_vector.shape[1]) # reshape (1, 2048) --> (2048,)
  return feature_vector

model = tf.keras.models.load_model('saved/training10_8.h5', compile=False)
model.load_weights('saved/training10_8.h5')

max_length = 80

with open('saved/index_to_word.pkl', 'rb') as fp:
  index_to_word = pickle.load(fp)

with open('saved/word_to_index.pkl', 'rb') as fp:
  word_to_index = pickle.load(fp)

# Upload an image and set some options for demo purposes
img_file = st.file_uploader(label='Upload an image file', type=['png', 'jpg', 'jpeg'])

subheading = "<div><span class='highlight blue bold bigfont'>Generated Caption</span</div>"

if img_file:
	img = Image.open(img_file)
	st.image(img, width=500)
	img.save('Img.jpg')
	img1 = encode_image('Img.jpg').reshape((1, 2048))
	caption = get_caption(img1)
	st.markdown(subheading, unsafe_allow_html=True)
	st.markdown(caption)
else:
	st.text('Waiting for input!')


components.html("""
	<hr color="#013220">
	<b>By</b>
	<ul>
		<li>Anweshan Mukherjee (503)</li>
		<li>Anushka Mukherjee (513)</li>
		<li>Sunetra Dutta (515)</li>
	</ul>
	<b>Project Guide</b>: Prof. Jayati Ghosh Dastidar
	"""
	)
