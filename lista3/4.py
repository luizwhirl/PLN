# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import sys

print("TensorFlow Version:", tf.__version__)

batch_size = 64
epochs = 20 
latent_dim = 256
num_samples = 10000
data_path = "por.txt"

if not os.path.exists(data_path):
    print(f"rquivo de dados não encontrado em {data_path}")
    print("veja se por.txt está na mesma pasta que este script")
    sys.exit(1)

print("\nsetup inicial concluído")


# a)
print("\n" + "="*60)
print("INICIANDO PARTE A: MODELO A NÍVEL DE CARACTERES")
print("="*60)

# processamento dos dados a 
input_texts_char = []
target_texts_char = []
input_characters = set()
target_characters = set()

with open(data_path, "r", encoding="utf-8") as f:
    lines = f.read().split("\n")

for line in lines[: min(num_samples, len(lines) - 1)]:
    try:
        input_text, target_text, _ = line.split("\t")
        target_text = "\t" + target_text + "\n"
        input_texts_char.append(input_text)
        target_texts_char.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    except ValueError:
        pass

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens_char = len(input_characters)
num_decoder_tokens_char = len(target_characters)
max_encoder_seq_length_char = max([len(txt) for txt in input_texts_char])
max_decoder_seq_length_char = max([len(txt) for txt in target_texts_char])

print(f"Número de amostras: {len(input_texts_char)}")
print(f"Número de tokens de entrada (caracteres) únicos: {num_encoder_tokens_char}")
print(f"Número de tokens de saída (caracteres) únicos: {num_decoder_tokens_char}")
print(f"Tamanho máximo da sequência de entrada: {max_encoder_seq_length_char}")
print(f"Tamanho máximo da sequência de saída: {max_decoder_seq_length_char}")

input_token_index_char = dict([(char, i) for i, char in enumerate(input_characters)])
target_token_index_char = dict([(char, i) for i, char in enumerate(target_characters)])

encoder_input_data_char = np.zeros(
    (len(input_texts_char), max_encoder_seq_length_char, num_encoder_tokens_char), dtype="float32"
)
decoder_input_data_char = np.zeros(
    (len(input_texts_char), max_decoder_seq_length_char, num_decoder_tokens_char), dtype="float32"
)
decoder_target_data_char = np.zeros(
    (len(input_texts_char), max_decoder_seq_length_char, num_decoder_tokens_char), dtype="float32"
)

for i, (input_text, target_text) in enumerate(zip(input_texts_char, target_texts_char)):
    for t, char in enumerate(input_text):
        encoder_input_data_char[i, t, input_token_index_char[char]] = 1.0
    encoder_input_data_char[i, t + 1 :, input_token_index_char[" "]] = 1.0
    for t, char in enumerate(target_text):
        decoder_input_data_char[i, t, target_token_index_char[char]] = 1.0
        if t > 0:
            decoder_target_data_char[i, t - 1, target_token_index_char[char]] = 1.0
    decoder_input_data_char[i, t + 1 :, target_token_index_char[" "]] = 1.0
    decoder_target_data_char[i, t:, target_token_index_char[" "]] = 1.0

print("\nprocessamento de dados a nível de caracteres concluído")

encoder_inputs_char = keras.Input(shape=(None, num_encoder_tokens_char), name="encoder_inputs_char")
encoder_lstm_char = keras.layers.LSTM(latent_dim, return_state=True, name="encoder_lstm_char")
_, state_h_char, state_c_char = encoder_lstm_char(encoder_inputs_char)
encoder_states_char = [state_h_char, state_c_char]

decoder_inputs_char = keras.Input(shape=(None, num_decoder_tokens_char), name="decoder_inputs_char")
decoder_lstm_char = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm_char")
decoder_outputs_char, _, _ = decoder_lstm_char(decoder_inputs_char, initial_state=encoder_states_char)
decoder_dense_char = keras.layers.Dense(num_decoder_tokens_char, activation="softmax", name="decoder_dense_char")
decoder_outputs_char = decoder_dense_char(decoder_outputs_char)

model_char = keras.Model([encoder_inputs_char, decoder_inputs_char], decoder_outputs_char)
model_char.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
print("\nModelo a nível de caracteres construído. Iniciando treinamento")

model_char.fit(
    [encoder_input_data_char, decoder_input_data_char],
    decoder_target_data_char,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1,
)
print("\ntreinamento do modelo de caracteres concluído")

encoder_model_char = keras.Model(encoder_inputs_char, encoder_states_char)

decoder_state_input_h = keras.Input(shape=(latent_dim,))
decoder_state_input_c = keras.Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs_inf, state_h_inf, state_c_inf = decoder_lstm_char(decoder_inputs_char, initial_state=decoder_states_inputs)
decoder_states_inf = [state_h_inf, state_c_inf]
decoder_outputs_inf = decoder_dense_char(decoder_outputs_inf)
decoder_model_char = keras.Model([decoder_inputs_char] + decoder_states_inputs, [decoder_outputs_inf] + decoder_states_inf)

reverse_input_char_index = dict((i, char) for char, i in input_token_index_char.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index_char.items())

def decode_sequence_char(input_seq):
    states_value = encoder_model_char.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1, num_decoder_tokens_char))
    target_seq[0, 0, target_token_index_char["\t"]] = 1.0
    stop_condition = False
    decoded_sentence = ""
    while not stop_condition:
        output_tokens, h, c = decoder_model_char.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length_char:
            stop_condition = True
        target_seq = np.zeros((1, 1, num_decoder_tokens_char))
        target_seq[0, 0, sampled_token_index] = 1.0
        states_value = [h, c]
    return decoded_sentence

print("\n 5 exemplos de tradução (nível de caracteres):")
example_indices_char = [0, 1, 5, 20, 30]
for i in example_indices_char:
    input_seq = encoder_input_data_char[i : i + 1]
    decoded_sentence = decode_sequence_char(input_seq)
    print("-")
    print("Frase em Inglês:", input_texts_char[i])
    print("Tradução Gerada:", decoded_sentence.strip())

print("\n fim parte A")


# b
print("\n" + "="*60)
print("INICIANDO PARTE B: MODELO A NÍVEL DE PALAVRAS")
print("="*60)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

input_texts_word = input_texts_char
target_texts_word_input = ['<sos> ' + text.replace('\t', '').replace('\n', '') for text in target_texts_char]
target_texts_word_output = [text.replace('\t', '').replace('\n', '') + ' <eos>' for text in target_texts_char]

tokenizer_inputs = Tokenizer(filters='')
tokenizer_inputs.fit_on_texts(input_texts_word)
input_sequences = tokenizer_inputs.texts_to_sequences(input_texts_word)
word2idx_inputs = tokenizer_inputs.word_index
num_encoder_tokens_word = len(word2idx_inputs) + 1
max_encoder_seq_length_word = max(len(s) for s in input_sequences)

tokenizer_targets = Tokenizer(filters='')
tokenizer_targets.fit_on_texts(target_texts_word_input + target_texts_word_output)
target_sequences_input = tokenizer_targets.texts_to_sequences(target_texts_word_input)
target_sequences_output = tokenizer_targets.texts_to_sequences(target_texts_word_output)
word2idx_targets = tokenizer_targets.word_index
num_decoder_tokens_word = len(word2idx_targets) + 1
max_decoder_seq_length_word = max(len(s) for s in target_sequences_input)

print(f'Número de tokens de entrada (palavras) únicos: {num_encoder_tokens_word}')
print(f'Número de tokens de saída (palavras) únicos: {num_decoder_tokens_word}')
print(f'Tamanho máximo da sequência de entrada: {max_encoder_seq_length_word}')
print(f'Tamanho máximo da sequência de saída: {max_decoder_seq_length_word}')

encoder_input_data_word = pad_sequences(input_sequences, maxlen=max_encoder_seq_length_word, padding='post')
decoder_input_data_word = pad_sequences(target_sequences_input, maxlen=max_decoder_seq_length_word, padding='post')

decoder_target_data_word = np.zeros(
    (len(target_texts_char), max_decoder_seq_length_word, num_decoder_tokens_word),
    dtype='float32'
)
for i, seq in enumerate(target_sequences_output):
    for t, word_idx in enumerate(seq):
        if t < max_decoder_seq_length_word:
            decoder_target_data_word[i, t, word_idx] = 1.0

print("\nprocessamento de dados a nível de palavras concluído")

embedding_dim = latent_dim

encoder_inputs_word = keras.Input(shape=(None,), name="encoder_inputs_word")
encoder_embedding = keras.layers.Embedding(num_encoder_tokens_word, embedding_dim, name="encoder_embedding")(encoder_inputs_word)
encoder_lstm_word = keras.layers.LSTM(latent_dim, return_state=True, name="encoder_lstm_word")
_, state_h_word, state_c_word = encoder_lstm_word(encoder_embedding)
encoder_states_word = [state_h_word, state_c_word]

decoder_inputs_word = keras.Input(shape=(None,), name="decoder_inputs_word")
decoder_embedding_layer = keras.layers.Embedding(num_decoder_tokens_word, embedding_dim, name="decoder_embedding")
decoder_embedding = decoder_embedding_layer(decoder_inputs_word)
decoder_lstm_word = keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, name="decoder_lstm_word")
decoder_outputs_word, _, _ = decoder_lstm_word(decoder_embedding, initial_state=encoder_states_word)
decoder_dense_word = keras.layers.Dense(num_decoder_tokens_word, activation='softmax', name="decoder_dense_word")
decoder_outputs_word = decoder_dense_word(decoder_outputs_word)

model_word = keras.Model([encoder_inputs_word, decoder_inputs_word], decoder_outputs_word)
model_word.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
print("\nmodelo a nível de palavras construído - Iniciando treinamento")


model_word.fit(
    [encoder_input_data_word, decoder_input_data_word],
    decoder_target_data_word,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    verbose=1
)
print("\ntreinamento do modelo de palavras concluído")

encoder_model_word = keras.Model(encoder_inputs_word, encoder_states_word)

decoder_state_input_h_word = keras.Input(shape=(latent_dim,))
decoder_state_input_c_word = keras.Input(shape=(latent_dim,))
decoder_states_inputs_word = [decoder_state_input_h_word, decoder_state_input_c_word]

decoder_embedding_inf = decoder_embedding_layer(decoder_inputs_word)
decoder_outputs_inf_word, state_h_inf_word, state_c_inf_word = decoder_lstm_word(decoder_embedding_inf, initial_state=decoder_states_inputs_word)
decoder_states_inf_word = [state_h_inf_word, state_c_inf_word]
decoder_outputs_inf_word = decoder_dense_word(decoder_outputs_inf_word)

decoder_model_word = keras.Model([decoder_inputs_word] + decoder_states_inputs_word, [decoder_outputs_inf_word] + decoder_states_inf_word)

idx2word_input = {v: k for k, v in word2idx_inputs.items()}
idx2word_target = {v: k for k, v in word2idx_targets.items()}

def decode_sequence_word(input_seq):
    states_value = encoder_model_word.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_targets['<sos>']
    
    stop_condition = False
    decoded_sentence = []
    
    while not stop_condition:
        output_tokens, h, c = decoder_model_word.predict([target_seq] + states_value, verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = idx2word_target.get(sampled_token_index, '')
        
        if sampled_word == '<eos>' or len(decoded_sentence) > max_decoder_seq_length_word:
            stop_condition = True
        else:
            if sampled_word:
                decoded_sentence.append(sampled_word)
        
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        states_value = [h, c]
        
    return ' '.join(decoded_sentence)

print("\n 5 exemplos de tradução:")
example_indices_word = [10, 25, 40, 55, 100]
for i in example_indices_word:
    input_seq = encoder_input_data_word[i: i + 1]
    decoded_sentence = decode_sequence_word(input_seq)
    print("-")
    print("Frase em Inglês:", input_texts_word[i])
    print("Tradução Gerada:", decoded_sentence)

print("\nFIM DA PARTE B")
