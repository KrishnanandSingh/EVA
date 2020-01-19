# %%
# Word level one hot encoding

import numpy as np

# create list of samples from your dataset
# a sample can be a sentence or even a paragraph.
samples = ['The cat sat on the mat.', 'The dog ate my homeworks.']

# dict to maintain index of each word
token_index = {}

# for every word in the sample assign an index
# don't assign index 0 to word. why?
for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1

# specify maximum number of words for a sample.
# specify some value that would be enough for most of the samples in your dataset.
max_sentence_length = 10

# vectorize the samples
results = np.zeros(shape=(len(samples), max_sentence_length, max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    # ignore sample after max_sentence_length
    for j, word in enumerate(sample.split()[:max_sentence_length]):
        index = token_index.get(word)
        results[i, j, index] = 1

# %%
# character-level one hot encoding
import numpy as np
import string

samples = ['The cat sat on the mat.', 'The dog ate my homeworks.']

# get all ascii characters
characters = string.printable

# assign each character an index starting with 1
token_index = dict(zip(characters, range(1, len(characters) + 1)))

# specify max value for character length which will be apt for you dataset
max_sentence_length = 50

# vectorize the samples
results = np.zeros(shape=(len(samples), max_sentence_length, max(token_index.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        print(index, 1)
        results[i, j, index] = 1

# %%
# Using Keras for word-level one hot encoding

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homeworks.']

# create a tokenizer that can hold only 1000 most common words
# tokenizer is similar to the token_indexer we created in the "Word level one hot encoding" example
tokenizer = Tokenizer(num_words=1000)
# go through my dataset, find top 1000 most common words,
# and create indexes for them
tokenizer.fit_on_texts(samples)

# convert each sample into array of indexes
sequences = tokenizer.texts_to_sequences(samples)

# convert list of texts to numpy matrix
# shape is len(samples), 1000
# value for existing words in a sample is 1. all other positions are 0
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

# tokenizer stores the index dict in this variable
# this can be used to map indexes back to the words later
word_index = tokenizer.word_index
print('Found {num} unique tokens.'.format(num = len(word_index)))

# %%
# word level one hot encoding with hashing trick

samples = ['The cat sat on the mat.', 'The dog ate my homeworks.']

# stores words as vectors of size 1000
dimensionality = 1000
max_sentence_length = 10

results = np.zeros((len(samples), max_sentence_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_sentence_length]:
        # hash value is very big integer (~19 digits)
        hash_val = hash(word)
        positive_hash_val = abs(hash_val)
        # generate value for index
        index = positive_hash_val % dimensionality
        results[i, j, index] = 1.

# as we see the index is dependent on the dimensionality
# it should be large enough to produce different values for different words
# must not be similar to the number of words we have

# %%
# Instantiating an embedding layer
from keras.layers import Embedding

# create embedding layer for at most 1000 possible tokens,
# and each token to be represented by vectors of size 64
embedding_layer = Embedding(1000, 64)
# Instead of words Embedding takes integer arrays(sequences)
# each integer is looked up for it's vector representation of 64 size

# %%
# Loading the IMDB data for use with an Embedding layer
from keras.datasets import imdb
from keras import preprocessing

max_tokens = 1000
max_sentence_length = 20

# downloads 25000 reviews for training and 25000 for testing
# ensure that word index does not exceed 1000
# as we have limited it to have only 1000 most frequently used words
# each word is represented as integer from the index dict
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_tokens)

# this is how each review can be decoded back
word_index = imdb.get_word_index()
reverse_word_index = dict(
                        [(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in x_train[0]])
# indices are offset by 3 as 0, 1 and 2 are reserved indices for "padding", "start of sequence" and "unknown"

# padding with 0 to ensure that each sequence has same legth
# if maxlen is not provided it pads each sequence to the length of largest sequence in x_trains
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_sentence_length)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_sentence_length)


# %%
# Using an Embedding layer and classifier on the IMDB data
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Embedding
from keras.datasets import imdb
from keras import preprocessing

max_tokens, dimensionality = 1000, 8
max_sentence_length = 20

model = Sequential()
embedding_layer = Embedding(max_tokens, dimensionality, input_length=max_sentence_length)
# max_tokens will be upper cap on keys in word_index
# after this layer output shape would be num of samples x max_sentence_length x dimensionality
model.add(embedding_layer)
# convert this 2D data to 1D
model.add(Flatten())
# add classifier
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.summary()

# Input pipeline
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_tokens)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_sentence_length)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_sentence_length)

# train for 10 epochs, reserve 20% train data for validation, don't train on it
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

# test score
model.evaluate(x_test, y_test)

# %%
