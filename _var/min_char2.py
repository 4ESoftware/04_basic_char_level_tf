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
  
  log.P("Done load data.", show_time=True)
  
  start_text = 'A fost o data ca niciodat'
  prediction_size = 200

  hidden_size = 100
  seq_len = 25 #len(start_text)
  lr = 1e-3
  epochs = 100
  steps = 10
  
  log.P("Creating graph ...")  
  g = tf.Graph()
  with g.as_default():
    tf_initializer = tf.random_normal_initializer(stddev=0.1)
    
    tf_x_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vocab_size], 
                              name='x_seq')
    tf_y_seq = tf.placeholder(dtype=tf.float32, shape=[seq_len, vocab_size], 
                              name='y_target')
    tf_h_start = tf.placeholder(dtype=tf.float32, shape=[1,hidden_size], 
                                name='init_h')
    
    tf_Wx = tf.get_variable("Wx", [vocab_size, hidden_size], initializer=tf_initializer)
    tf_Wh = tf.get_variable("Wh", [hidden_size, hidden_size], initializer=tf_initializer)
    tf_Wy = tf.get_variable("Wy", [hidden_size, vocab_size], initializer=tf_initializer)    
    tf_bh = tf.get_variable("h_bias", [hidden_size], initializer=tf_initializer) 
    tf_by = tf.get_variable("y_bias", [vocab_size], initializer=tf_initializer) 
  
    all_seq_items = tf.split(tf_x_seq, seq_len)
    tf_h = tf_h_start
    all_seq_outputs = []
    for i,tf_step_input in enumerate(all_seq_items):
      tf_h = tf.tanh(tf.matmul(tf_step_input, tf_Wx) + tf.matmul(tf_h, tf_Wh) + tf_bh,
                     name='h_step_{}'.format(i))
      tf_y = tf.add(tf.matmul(tf_h, tf_Wy), tf_by, 'y_step_{}'.format(i))
      all_seq_outputs.append(tf_y)
    
    tf_all_unrolled_outputs = tf.concat(all_seq_outputs, axis=0)
    tf_final_softmax = tf.nn.softmax(all_seq_outputs[-1])
    tf_final_h = tf_h
    tf_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf_y_seq, logits=tf_all_unrolled_outputs)
    tf_loss = tf.reduce_mean(tf_loss)
    opt = tf.train.AdamOptimizer(lr)
    SIMPLE_OPTIMIZE = False
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
  for epoch in range(epochs):
    # at begining of each epoch (text corpus) reset hidden state
    h_start = zero_state.copy()
    max_text_pos = train_text_size - (seq_len+1)
    log.P("Running epoch {} with {} steps".format(epoch, epoch_steps))
    rng = range(0, max_text_pos, seq_len)
    USE_TQDM = False
    if USE_TQDM:
      tpbar = tqdm(rng)
    else:
      tpbar = rng
    for text_cursor in tpbar:
      train_seq = full_text_idx[text_cursor:text_cursor+seq_len]
      target_seq = full_text_idx[text_cursor+1 : text_cursor+seq_len+1]
      train_seq_oh = one_hot(train_seq)
      target_seq_oh = one_hot(target_seq)
      h_start, loss, _ = sess.run([tf_final_h, tf_loss, tf_train_op],
                                  feed_dict={
                                      tf_x_seq: train_seq_oh,
                                      tf_y_seq: target_seq_oh,
                                      tf_h_start: h_start
                                      })
      if (c_iter % 500) == 0 and (c_iter!=0):
        """
        # Progress
        print('iter: {}, p: {}, loss: {}'.format(c_iter, text_cursor, loss))

        # Do sampling
        sample_length = 200
        #start_ix      = random.randint(0, len(data) - seq_length)
        #sample_seq_ix = [char_to_ix[ch] for ch in data[start_ix:start_ix + seq_length]]
        sample_seq_ix = [char2idx[ch] for ch in start_text]
        ixes          = []
        sample_prev_state_val = np.copy(h_start)
        print("Sample start: [{}]".format("".join(idx2char[ind] for ind in sample_seq_ix)))
      
        for t in range(sample_length):
            sample_input_vals = one_hot(sample_seq_ix)
            sample_output_softmax_val, sample_prev_state_val = \
                sess.run([tf_final_softmax, tf_final_h],
                         feed_dict={tf_x_seq: sample_input_vals, tf_h_start: sample_prev_state_val})

            ix = np.random.choice(range(vocab_size), p=sample_output_softmax_val.ravel())
            ixes.append(ix)
            sample_seq_ix = sample_seq_ix[1:] + [ix]

        txt = ''.join(idx2char[ix] for ix in ixes)
        print('----\n %s \n----\n' % (txt,))
        """
        output_text = start_text
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
          output_text += pred_ch
        log.PrintPad(" Iter {} Predicted text:".format(c_iter), output_text)        
      
      c_iter += 1
      #tpbar.set_description("Index {}/{} loss: {:.2f}".format(text_cursor,train_text_size,loss))     
    log.P("Epoch {} Loss:{:.3f}".format(epoch, loss))
    output_text = start_text
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
      output_text += pred_ch
    log.P(" Epoch {} Predicted text:\n{}".format(epoch, output_text))
      
  
  
    
  