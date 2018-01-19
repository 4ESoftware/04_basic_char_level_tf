# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 22:24:21 2018

@author: Andrei Ionut Damian
"""

from keras.layers import LSTM, TimeDistributed, Dense, Input
from keras.models import Model

import numpy as np

import os

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

def get_letter(ohv, vocab):
    i = np.argmax(ohv)
    return vocab[i]

if __name__=='__main__':
  log = load_module('logger','logger.py').Logger(lib_name='KCRNN')

  start_text = 'A fost o data ca niciodata '
  prediction_size = 200
  hidden_size = 100
  seq_len = len(start_text)
  iterations = 100
  epochs = 1
  batch_size = 1
  shuffle_train_data = False
  stateful_lstm = True


  log.P("Load data...")
  with open('text.txt','rt') as f:
    lines = f.readlines()
  full_text =" ".join(lines)
  full_text = full_text[:(len(full_text)//seq_len * seq_len + 1)]
  vocab = sorted(list(set(full_text)))
  vocab_size =len(vocab)
  vocab_oh = np.eye(vocab_size)
  char2idx = {ch:i for i, ch in enumerate(vocab)}
  idx2char = {i:ch for i, ch in enumerate(vocab)}
  full_text_idx = [char2idx[c] for c in full_text]
  train_text_size = len(full_text_idx)
  one_hot = lambda idx:vocab_oh[idx]
  str_to_oh = lambda _str: one_hot([char2idx[c] for c in _str])
  preds_to_str = lambda vect: "".join([idx2char[np.argmax(pred)] for pred in vect])
  preds_to_str_rnd = lambda vect: "".join([idx2char[np.random.choice(range(vocab_size),p=pred.ravel())] for pred in vect])
  
  X_train = str_to_oh(full_text[:-1]).reshape((-1, seq_len, vocab_size))
  y_train = str_to_oh(full_text[1:]).reshape((-1, seq_len, vocab_size))
  
  log.P("Done load data.", show_time=True)
  
  
  log.P("Creating model...")    
  input_layer = Input(batch_shape=(batch_size, seq_len, vocab_size))
  lstm_layer = LSTM(hidden_size, 
                    return_sequences=True,
                    stateful=stateful_lstm)
  lstm_output = lstm_layer(input_layer)
  softmaxes_layer = TimeDistributed(Dense(vocab_size, activation='softmax'))
  final_output = softmaxes_layer(lstm_output)
  model = Model(input_layer, final_output)
  model.compile(optimizer='adam', loss='categorical_crossentropy')
  log.P("Model created.")  
  model.summary()
  
  for i in range(iterations):
    log.P("Training iteration {}".format(i))
    model.fit(X_train, y_train, 
              batch_size=batch_size, 
              epochs=epochs, 
              shuffle=shuffle_train_data,
              verbose=2)
    model.reset_states()
    output_text = start_text
    input_text = start_text
    for p in range(prediction_size):
      x_test = str_to_oh(input_text)
      x_test = np.expand_dims(x_test, axis=0)
      preds = model.predict(x_test)
      str_preds = preds_to_str(np.squeeze(preds))
      ch = str_preds[-1]
      output_text += ch
      input_text = input_text[1:] + ch
    log.P("Testing model at iteration {} with input '{}':\n{}".format(i, start_text, output_text))