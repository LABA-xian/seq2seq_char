# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:51:40 2019

@author: MB207
"""

from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, Embedding, TimeDistributed, Dropout
import jieba
from keras import callbacks
import gensim
import warnings
import os
import random
import tensorflow as tf
from tensorflow import Graph, Session
file_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'chainsea_all_API_1'))

class seq2seq_all():
    
    def __init__(self):
        
        
#        self.vector = gensim.models.KeyedVectors.load_word2vec_format('wiki_word.txt.bin')
        warnings.filterwarnings("ignore")
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        self.latent_dim = 128  # Latent dimensionality of the encoding space.
        # Path to the data txt file on disk.
        data_path = file_path + '/QA_all.txt'
        self.num_samples = 60000
        # Vectorize the data.
        self.input_texts = []
        self.target_texts = []
        self.BATCH_SIZE = 64
        input_characters = set()
        target_characters = set()
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text  = line.split('\t')
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            
            
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)
        
#        self.ok_input_texts, self.ok_target_texts = self.check_vector( self.input_texts, self.target_texts, self.vector)
        
        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])
        
#        print('Number of samples:', len(self.input_texts))
#        print('Number of unique input tokens:', self.num_encoder_tokens)
#        print('Number of unique output tokens:', self.num_decoder_tokens)
#        print('Max sequence length for inputs:', self.max_encoder_seq_length)
#        print('Max sequence length for outputs:', self.max_decoder_seq_length)
        
        self.input_token_index = dict(
            [(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict(
            [(char, i) for i, char in enumerate(target_characters)])

#        self.decoder_input_data = np.zeros(
#            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
#            dtype='float32')
#        
#        self.decoder_target_data = np.zeros(
#            (len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),
#            dtype='float32')


        self.encoder_input_data = self.convert_data(self.input_texts,self.input_token_index , self.max_encoder_seq_length)
        self.teach_data = self.convert_data(self.target_texts, self.target_token_index, self.max_decoder_seq_length)
        

        self.target_data = [[self.teach_data[n][i+1] for i in range(len(self.teach_data[n])-1)] for n in range(len(self.teach_data))]
        self.target_data = pad_sequences(self.target_data, maxlen=self.max_decoder_seq_length, padding="post")
        self.target_data = self.target_data.reshape((self.target_data.shape[0], self.target_data.shape[1], 1))


        self.reverse_input_char_index = dict(
        (i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict(
        (i, char) for char, i in self.target_token_index.items())


#        for i, target_text in enumerate(self.target_texts):
#            for t, char in enumerate(target_text):
#                # decoder_target_data is ahead of decoder_input_data by one timestep
#                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
#                if t > 0:
#                    # decoder_target_data will be ahead by one timestep
#                    # and will not include the start character.
#                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.       

    def convert_data(self,data, word_dict, Max_length):
        
        convert = []
        convert_result = []
        
        for d in data:
            convert = []
            for char in d:
                if char in word_dict: 
                    convert.append(word_dict[char])
                else:
                    pass
            convert_result.append(convert)
        
        convert_result = pad_sequences(convert_result, Max_length, padding='post')
        
        return convert_result

    def generate_batch_data_random(self, x, t , y, batch_size):
        ylen = len(y)
        loopcount = ylen // batch_size
        while (True):
            i = random.randint(0,loopcount)
            yield [x[i * batch_size:(i + 1) * batch_size], t[i * batch_size:(i + 1) * batch_size]], y[i * batch_size:(i + 1) * batch_size]


    def train_test_split(self, x, t, y):
        
        train_data = x[:int(len(x) * 0.8)]
        teach_data = t[:int(len(t) * 0.8)]
        train_data_y = y[:int(len(y) * 0.8)]
        
        test_data = x[int(len(x) * 0.8):]
        test_teach_data = t[int(len(x) * 0.8):]
        test_y = y[int(len(x) * 0.8):]

        
        return train_data, teach_data, train_data_y, test_data, test_teach_data, test_y


      
        
        

#訓練
    def build_basic_model(self):
        # Encoder
        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(self.num_encoder_tokens, 200, name='encoder_embedding')(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name='encoder_lstm')
        encoder_lstm.supports_masking = True
        _, *encoder_states = encoder_lstm(encoder_embedding)
    
        # Decoder
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(self.num_decoder_tokens, 200, name='decoder_embedding')(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_state=True, return_sequences=True ,name='decoder_lstm')
        rnn_outputs, *decoder_states = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_dense.supports_masking = True
        decoder_outputs = decoder_dense(Dropout(rate=0.4, name='dropout1')(rnn_outputs))
    
        basic_model = Model([encoder_inputs, decoder_inputs], [decoder_outputs])
        basic_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        
        train_data, teach_data, train_data_y, test_data, test_teach_data, test_y = self.train_test_split(self.encoder_input_data, 
                                                                                                         self.teach_data, 
                                                                                                         self.target_data)
        
        
        basic_model.fit_generator(self.generate_batch_data_random(train_data, teach_data, train_data_y, self.BATCH_SIZE),
                               validation_data=self.generate_batch_data_random(test_data, test_teach_data, test_y, self.BATCH_SIZE),
                               steps_per_epoch=train_data.shape[0] // self.BATCH_SIZE,
                               validation_steps=test_data.shape[0] // self.BATCH_SIZE,
                               epochs=10,
                               workers=5
                               ) 


   
#        # 回调函数
#        callback_list = [callbacks.ModelCheckpoint('C:/Users/user/Desktop/chan-si/QA_Generate/web/chan-si_API_1/basic_model_best_V2.h', save_best_only=True)]
#        # 训练
#        basic_model_hist = basic_model.fit([self.encoder_input_data, self.decoder_input_data], self.decoder_target_data,
#                        batch_size=32, epochs=15,
#                        validation_split=0.2, callbacks=callback_list)
        
        basic_model.save_weights("P1_old.h5")

    def creat_model(self):

        encoder_inputs = Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = Embedding(self.num_encoder_tokens, 200, name='encoder_embedding')(encoder_inputs)
        encoder_lstm = LSTM(self.latent_dim, return_state=True, name='encoder_lstm')
        encoder_lstm.supports_masking = True
        _, *encoder_states = encoder_lstm(encoder_embedding)
    
        # Decoder
        decoder_inputs = Input(shape=(None,), name='decoder_inputs')
        decoder_embedding = Embedding(self.num_decoder_tokens, 200, name='decoder_embedding')(decoder_inputs)
        decoder_lstm = LSTM(self.latent_dim, return_state=True, return_sequences=True ,name='decoder_lstm')
        rnn_outputs, *decoder_states = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(self.num_decoder_tokens, activation='softmax', name='decoder_dense')
        decoder_dense.supports_masking = True
        decoder_outputs = decoder_dense(Dropout(rate=0.4, name='dropout1')(rnn_outputs))
    
        basic_model = Model([encoder_inputs, decoder_inputs], [decoder_outputs])
        basic_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        for op in tf.get_default_graph().get_operations():
            print(op.graph)
            break
        
        return basic_model


        # 建立推理模型
    def build_basic_inference_model(self):
        
        model = self.creat_model()
        
        model.load_weights(file_path + "/P1_old.h5")
        
        global graph
        graph = tf.get_default_graph()

        print('step3')
        # encoder
        encoder_inputs = Input(shape=(None,))
        # encoder_embedding
        encoder_embedding = model.get_layer('encoder_embedding')(encoder_inputs)
        # get encoder states
        _, *encoder_states = model.get_layer('encoder_lstm')(encoder_embedding)
        encoder_model = Model(encoder_inputs, encoder_states)
            
        # decoder
        # decoder inputs
        decoder_inputs = Input(shape=(None,))
        # decoder input states 
        decoder_state_h = Input(shape=(self.latent_dim,))
        decoder_state_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_h, decoder_state_c]
            
        # get rnn outputs and decoder states
        decoder_embedding = model.get_layer('decoder_embedding')(decoder_inputs)
        rnn_outputs, *decoder_states = model.get_layer('decoder_lstm')(decoder_embedding, initial_state=decoder_states_inputs)
        
        dropout1 = model.get_layer('dropout1')(rnn_outputs)
        
        decoder_outputs = model.get_layer('decoder_dense')(dropout1)
            
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs]+decoder_states)
            
        return encoder_model, decoder_model


#def sample(preds, temperature=0.01):
#    preds = np.asarray(preds).astype('float64')
#    preds = np.log(preds) / temperature
#    exp_preds = np.exp(preds)
#    preds = exp_preds / np.sum(exp_preds)
#    probas = np.random.multinomial(1, preds, 1)
#    return np.argmax(probas)


    def decode_sequence(self, input_seq, encoder_model, decoder_model):
        # Encode the input as state vectors.
        with graph.as_default():
            states_value = encoder_model.predict(input_seq)
        
            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1))
            # Populate the first character of target sequence with the start character.
            target_seq[0, 0] = self.target_token_index['\t']
        
            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = ''
            while not stop_condition:
                output_tokens, h, c = decoder_model.predict(
                    [target_seq] + states_value)
                # Sample a token
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = self.reverse_target_char_index[sampled_token_index]
                    
                decoded_sentence += sampled_char
        
                # Exit condition: either hit max length
                # or find stop character.
                if (sampled_char == '\n' or
                   len(decoded_sentence) > self.max_decoder_seq_length):
                    stop_condition = True
        
                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1))
                target_seq[0,0 ] = sampled_token_index
        
                # Update states
                states_value = [h, c]
    
        return decoded_sentence

    def run_model(self,input_test):
    
        # 导入模型
        inference_encoder_model, inference_decoder_model = self.build_basic_inference_model()
        
#        # 测试
#        for seq_index in range(217,218):
#            # Take one sequence (part of the training set)
#            # for trying out decoding.
#            input_seq = self.encoder_input_data[seq_index:seq_index+1]
#        #    print(input_seq)
#            decoded_sentence = self.decode_sequence(input_seq, inference_encoder_model, inference_decoder_model)
#            print('-')
#            print('Input sentence:', self.input_texts[seq_index])
#            print('Decoded sentence:', decoded_sentence)
#            print('origin sentence:', lines[seq_index].split('\t')[0])
  
        test_input_data = self.convert_data(input_test, self.input_token_index, self.max_encoder_seq_length)   
        input_seq = test_input_data[0:1]
        decoded_sentence = self.decode_sequence(input_seq, inference_encoder_model, inference_decoder_model)
#        print('Decoded sentence:', decoded_sentence)
#        print('-')
        decoded_sentence = list(decoded_sentence)
        decoded_sentence_remove = []
        [decoded_sentence_remove.append(i) for i in decoded_sentence if not i in decoded_sentence_remove]
        return_str = "".join(decoded_sentence_remove)
        return str(return_str)
    
                
 

x = seq2seq_all()
##x.build_basic_model()
##
test_text = [input('【input Answer】 \n' )]
result = x.run_model(test_text)
print('【output question】 \n', result)

    
    