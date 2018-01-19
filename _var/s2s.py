# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:56:32 2018

@author: Andrei Ionut Damian
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense, TimeDistributed
import numpy as np

num_encoder_tokens = 100
num_decoder_tokens = 100
encoder_seq_length = None
decoder_seq_length = None
batch_size = 128
epochs = 3

# Dummy data
input_seqs = np.random.random((1000, 10, num_encoder_tokens))
target_seqs = np.random.random((1000, 10, num_decoder_tokens))

# Define training model
encoder_inputs = Input(shape=(encoder_seq_length,
                              num_encoder_tokens))
encoder = LSTM(256, return_state=True, return_sequences=True)
encoder_outputs = encoder(encoder_inputs)
_, encoder_states = encoder_outputs[0], encoder_outputs[1:]

decoder_inputs = Input(shape=(decoder_seq_length,
                              num_decoder_tokens))
decoder = LSTM(256, return_sequences=True)
decoder_outputs = decoder(decoder_inputs, initial_state=encoder_states)
decoder_outputs = TimeDistributed(
    Dense(num_decoder_tokens, activation='softmax'))(decoder_outputs)
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Training
model.compile(optimizer='rmsprop', loss='mse')
model.fit([input_seqs, target_seqs], target_seqs,
          batch_size=batch_size, epochs=epochs)

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Append the target token and repeat

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs = decoder(decoder_inputs,
                          initial_state=decoder_states)
decoder_model = Model(
    [decoder_inputs] + decoder_states,
    decoder_outputs)

# Dummy data
input_seqs = np.random.random((1000, 10, num_encoder_tokens))
target_seqs = np.random.random((1000, 1, num_decoder_tokens))

# Sampling loop for a batch of sequences
states_values = encoder_model.predict(input_seqs)
stop_condition = False
while not stop_condition:
    output_tokens = decoder_model.predict([target_seqs] + states_values)
    # sampled_token = ... # sample the next token
    # target_seqs = ...  # append token to targets
    # stop_condition = ...  # stop when "end of sequence" token is generated
    break