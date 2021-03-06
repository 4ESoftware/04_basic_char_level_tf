# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 12:17:21 2018

@author: Andrei Ionut Damian
"""
import numpy as np
import tensorflow as tf
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

def get_letter(ohv, vocab):
    i = np.argmax(ohv)
    return vocab[i]

if __name__=='__main__':
  log = load_module('logger','logger.py').Logger(lib_name='CRNN')
  log.P("Load data...")
  with open('text.txt','rt') as f:
    lines = f.readlines()
  full_text =" ".join(lines)
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
  log.P("Done load data.", show_time=True)
  
  start_text = 'A fost o data ca niciodata '
  prediction_size = 200

  hidden_size = 100
  seq_len = len(start_text)
  lr = 1e-3
  epochs = 100
  steps = 10
  
  log.P("Creating graph ...")  
  g = tf.Graph()
  with g.as_default():
    
    tf_x_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vocab_size], 
                              name='x_seq')
    tf_y_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vocab_size], 
                              name='y_target')
    tf_h_start = tf.placeholder(dtype=tf.float32, shape=[1,hidden_size], 
                                name='init_h')
    
    #tf_initializer = tf.random_normal_initializer(stddev=0.1)
    #tf_Wx = tf.get_variable("Wx", [vocab_size, hidden_size], initializer=tf_initializer)
    #tf_Wh = tf.get_variable("Wh", [hidden_size, hidden_size], initializer=tf_initializer)
    #tf_Wy = tf.get_variable("Wy", [hidden_size, vocab_size], initializer=tf_initializer)    
    #tf_bh = tf.get_variable("h_bias", [hidden_size], initializer=tf_initializer) 
    #tf_by = tf.get_variable("y_bias", [vocab_size], initializer=tf_initializer) 
    
    tf_Wx = tf.Variable(np.random.randn(vocab_size, hidden_size) * 0.01,
                        dtype=tf.float32, name='Wx')
    tf_Wh = tf.Variable(np.random.randn(hidden_size, hidden_size) * 0.01,
                        dtype=tf.float32, name='Wh')
    tf_Wy = tf.Variable(np.random.randn(hidden_size, vocab_size) * 0.01,
                        dtype=tf.float32, name='Wy')
    
    tf_bh = tf.Variable(np.zeros((1,hidden_size)),  
                        dtype=tf.float32, name='h_bias')
    tf_by = tf.Variable(np.zeros((1,vocab_size)),   
                        dtype=tf.float32, name='y_bias')
  
    all_seq_items = tf.split(tf_x_seq, seq_len)
    tf_h = tf_h_start
    all_seq_outputs = []
    for i,tf_step_input in enumerate(all_seq_items):
      tf_h = tf.tanh(tf.matmul(tf_step_input, tf_Wx) + tf.matmul(tf_h, tf_Wh) + tf_bh,
                     name='h_step_{}'.format(i))
      tf_y = tf.add(tf.matmul(tf_h, tf_Wy), tf_by, 'y_step_{}'.format(i))
      all_seq_outputs.append(tf_y)
    
    tf_all_unrolled_outputs = tf.concat(all_seq_outputs, axis=0)
    tf_final_softmax = tf.nn.softmax(tf_y)
    tf_final_h = tf_h
    tf_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf_y_seq, logits=tf_all_unrolled_outputs)
    tf_loss = tf.reduce_mean(tf_loss)
    opt = tf.train.AdamOptimizer(lr)
    SIMPLE_OPTIMIZE = True
    if SIMPLE_OPTIMIZE:      
      tf_train_op = opt.minimize(tf_loss)
    else:
      grads_and_vars = opt.compute_gradients(tf_loss)      
      # Gradient clipping
      tf_grad_clipping = tf.constant(5.0, name="grad_clipping")
      clipped_grads_and_vars = []
      for tf_grad, tf_var in grads_and_vars:
          tf_clipped_grad = tf.clip_by_value(tf_grad, -tf_grad_clipping, tf_grad_clipping)
          clipped_grads_and_vars.append((tf_clipped_grad, tf_var))      
      # Gradient updates
      tf_train_op = opt.apply_gradients(clipped_grads_and_vars)
      
    tf_init_op = tf.global_variables_initializer()
  log.P("Done creating graph.", show_time = True)
    
  sess = tf.Session(graph=g)
  sess.run(tf_init_op)
  zero_state = np.zeros((1,hidden_size)) 
  c_iter = 0
  show_bar_stats = False
  for epoch in range(epochs):
    # at begining of each epoch reset hiddent state
    h_start = zero_state.copy()
    #
    # we will step from seq to seq as each seq state will feed next sequence state
    # 
    epoch_steps_range = range(0,train_text_size - (seq_len+1), seq_len)
    log.P("Running epoch {} with {} steps".format(epoch, len(epoch_steps_range)))
    tpbar = tqdm(epoch_steps_range)
    for text_cursor in tpbar:
      train_seq = full_text_idx[text_cursor:text_cursor+seq_len]
      target_seq = full_text_idx[text_cursor+1 : text_cursor+seq_len+1]
      train_seq_oh = one_hot(train_seq)
      target_seq_oh = one_hot(target_seq)
      _, loss, h_start = sess.run([tf_train_op, tf_loss, tf_final_h],
                                  feed_dict={
                                      tf_x_seq: train_seq_oh,
                                      tf_y_seq: target_seq_oh,
                                      tf_h_start: h_start # prev seq state
                                      })
      c_iter += 1
      if show_bar_stats:
        tpbar.set_description("Index {}/{} loss: {:.2f}".format(text_cursor,train_text_size,loss))
    log.P("Epoch {} Loss:{:.3f}".format(epoch, loss))
    output_text1 = start_text
    input_text = start_text
    h_test = h_start.copy()
    for i in range(prediction_size):      
      input_seq_oh = str_to_oh(input_text)
      pred, h_test = sess.run([tf_final_softmax, tf_final_h],
                         feed_dict={
                          tf_x_seq: input_seq_oh,
                          tf_h_start: h_test
                          })
      pred_ch = preds_to_str(pred)
      input_text = input_text[1:] + pred_ch
      output_text1 += pred_ch
    output_text2 = start_text
    input_text = start_text
    h_test = h_start.copy()
    for i in range(prediction_size):      
      input_seq_oh = str_to_oh(input_text)
      pred, h_test = sess.run([tf_final_softmax, tf_final_h],
                         feed_dict={
                          tf_x_seq: input_seq_oh,
                          tf_h_start: h_test
                          })
      pred_ch = preds_to_str_rnd(pred)
      input_text = input_text[1:] + pred_ch
      output_text2 += pred_ch
    log.P(" Epoch {}\nPredicted text:\n{}\n\nSampled text:\n{}".format(epoch, 
          output_text1,output_text2))
      
  
  
    
  