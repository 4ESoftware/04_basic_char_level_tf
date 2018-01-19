# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 14:55:18 2018

@author: Andrei Ionut Damian
"""

import numpy as np
from keras.layers import LSTM, Dense, Input, RepeatVector,Flatten, TimeDistributed
from keras.models import Model

from sklearn.model_selection import train_test_split

import os

from tqdm import tqdm

def load_module(module_name, file_name):
  """
  loads modules from _pyutils Google Drive repository
  usage:
    module = load_module("logger", "logger.py")
    logger = module.Logger()
  """
  from importlib.machinery import SourceFileLoader
  home_dir = os.path.expanduser("~")
  valid_paths = [
                 os.path.join(home_dir, "Google Drive"),
                 os.path.join(home_dir, "GoogleDrive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "Google Drive"),
                 os.path.join(os.path.join(home_dir, "Desktop"), "GoogleDrive"),
                 os.path.join("C:/", "GoogleDrive"),
                 os.path.join("C:/", "Google Drive"),
                 os.path.join("D:/", "GoogleDrive"),
                 os.path.join("D:/", "Google Drive"),
                 ]

  drive_path = None
  for path in valid_paths:
    if os.path.isdir(path):
      drive_path = path
      break

  if drive_path is None:
    logger_lib = None
    print("Logger library not found in shared repo.", flush = True)
    #raise Exception("Couldn't find google drive folder!")
  else:  
    utils_path = os.path.join(drive_path, "_pyutils")
    print("Loading [{}] package...".format(os.path.join(utils_path,file_name)),flush = True)
    logger_lib = SourceFileLoader(module_name, os.path.join(utils_path, file_name)).load_module()
    print("Done loading [{}] package.".format(os.path.join(utils_path,file_name)),flush = True)

  return logger_lib


class CharOneHotConverter(object):
  """Given a set of characters:
  + Encode them to a one hot integer representation
  + Decode the one hot integer representation to their character output
  + Decode a vector of probabilities to their character output
  """
  def __init__(self, chars):
    """Initialize character table.
    # Arguments
        chars: Characters that can appear in the input.
    """
    self.chars = sorted(set(chars))
    self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
    self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

  def encode(self, C, num_rows):
    """One hot encode given string C.
    # Arguments
        num_rows: Number of rows in the returned one hot encoding. This is
            used to keep the # of rows for each data the same.
    """
    x = np.zeros((num_rows, len(self.chars)))
    for i, c in enumerate(C):
        x[i, self.char_indices[c]] = 1
    return x

  def decode(self, x, calc_argmax=True):
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(self.indices_char[x] for x in x)
  

if __name__=='__main__':
  log = load_module('logger','logger.py').Logger(lib_name='KABOT')
  # Parameters for the model and dataset.
  TRAINING_SIZE = 50000
  DIGITS = 4
  INVERT = True
  batch_size = 64
  epochs = 20
  iterations = 200
  
  # Maximum length of input is 'int + int' (e.g., '345+678'). Maximum length of
  # int is DIGITS.
  MAXLEN = DIGITS + 1 + DIGITS
  
  in_seq_len = MAXLEN
  out_seq_len = DIGITS + 1
  chars = '0123456789+ '
  nr_feats = len(chars)
  
  # All the numbers, plus sign and space for padding.
  conv = CharOneHotConverter(chars)
  
  questions = []
  expected = []
  seen = set()
  log.P('Generating data...')
  for tb in tqdm(range(TRAINING_SIZE)):
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    a = 1 if a==9999 else a
    b = 1 if b==9999 else b
    # Skip any addition questions we've already seen
    # Also skip any such that x+Y == Y+x (hence the sorting).
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    # Pad the data with spaces such that it is always MAXLEN.
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    # Answers can be of maximum size DIGITS + 1.
    ans += ' ' * (DIGITS + 1 - len(ans))
    if INVERT:
        # Reverse the query, e.g., '12+345  ' becomes '  543+21'. (Note the
        # space used for padding.)
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
    
  log.P('Total addition questions:', len(questions))
  
  log.P('Vectorization...')
  x = np.zeros((len(questions), MAXLEN, nr_feats), dtype=np.bool)
  y = np.zeros((len(questions), DIGITS + 1, nr_feats), dtype=np.bool)
  for i, sentence in enumerate(questions):
      x[i] = conv.encode(sentence, MAXLEN)
  for i, sentence in enumerate(expected):
      y[i] = conv.encode(sentence, DIGITS + 1)
  

  X_train, X_val, y_train, y_val = train_test_split(x, y, test_size=0.1)
  
  log.P('Training Data: X:{} y:{}'.format(X_train.shape, y_train.shape))  
  log.P('Validation Data: X:{} y:{}'.format(X_val.shape, y_val.shape))
  
  log.P("Preparing model...")
  
  input_layer = Input((in_seq_len, nr_feats))
  
  encoder = LSTM(128, return_sequences=True)
  encoder_output  = encoder(input_layer)
  
  encoder_embedding = Flatten()(encoder_output)
  repeat_vector = RepeatVector(out_seq_len)(encoder_embedding)
  
  
  
  decoder = LSTM(64, return_sequences=True)
  decoder_output = decoder(repeat_vector)
  
  readout_layer = TimeDistributed(Dense(nr_feats, activation='softmax'))
  readout = readout_layer(decoder_output)
  
  model = Model(input_layer, readout)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
  model.summary()
  
  
  for citer in range(iterations):
    model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
              validation_data=(X_val, y_val))
    log.P("Test:")
    a = input("a:")
    if a=='q':
      break
    b = input("b:")
    ans = int(a) + int(b)
    q = a+"+"+b
    query = q + ' ' * (in_seq_len - len(q))
    if INVERT:
      query = query[::-1]
    x_test = np.expand_dims(conv.encode(query, in_seq_len), axis=0)
    pred = model.predict(x_test)
    pans = conv.decode(pred.squeeze())
    i_pans = int(pans)
    log.P('a:{} b:{} ans:{} pred:{}'.format(a,b,ans, i_pans))
    
    
  
  
