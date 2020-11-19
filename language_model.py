# SOURCE https://keras.io/examples/pretrained_word_embeddings/
import os
import sys
import csv
import pickle as pkl
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.initializers import Constant

MAX_SEQUENCE_LENGTH = 200

def neural_net():
    net = Sequential()
    net.add(embedding_layer)
    net.add(Conv1D(128, 5, activation='relu'))
    net.add(MaxPooling1D(5))
    net.add(Conv1D(128, 5, activation='relu'))
    net.add(MaxPooling1D(5))
    net.add(Conv1D(128, 5, activation='relu'))
    net.add(GlobalMaxPooling1D())
    net.add(Dense(128, activation='relu'))
    return net

def x_percentage(x, y):
    '''
    Percentage of x relative to y
    '''
    return x / (x + y)

def decide(choices, model, tokenizer):
    inputs = tokenizer.texts_to_sequences(choices)
    inputs = pad_sequences(inputs, maxlen=MAX_SEQUENCE_LENGTH)
    inputs = np.expand_dims(inputs, axis=1)
    return model.predict([inputs[0], inputs[1]])[0]
    
    
def test_model(model, tokenizer):
    test_choices = [
        ["die", "live"],
        ["Having asthma every time you talk to someone",
         "Becoming the president of the United States"],
        ["Spend a day in the Sahara Desert",
         "Spend a day in the North Pole"],
        ["Be stalked by a ghost for your entire life",
         "Be stalked by a demon for three days then die"],
        ["Wear a ski suit all the time",
         "Go everywhere barefoot"]
    ] + \
    [
        ["Swim 300 meters through shit", "Swim 300 meters through dead bodies"],
        ["Have a dog with a cat’s personality", "Have a cat with a dog’s personality"],
        ["Lose the ability to lie", "Believe everything you’re told"],
        ["To be", "Not to be"]
    ]
    # Hardest questions https://lifehacks.io/would-you-rather-questions/
    # [
    #     ["Get the ability to dodge bullets",
    #      "Get the ability to forsee the future"], # 0.2710
    #     ["Get reborn an infinite amount of time",
    #      "Never die"], # 0.43906182
    #     ["World war 3",
    #      "Coronavirus"] # 0.571851
    # Who would win?
    # A president with the blessings from Jesus himself 
    # Or
    # Some cripple on a horse
    # ]
    for c1, c2 in test_choices:
        prediction = decide([c1,c2], model, tokenizer)
        print(c1)
        print(c2)
        print(prediction)
    
def save_model(model, tokenizer, path):
    model.save(path + "_model.h5")
    with open(path + "_tokenizer.pkl", "wb") as f:
        pkl.dump(tokenizer, f, protocol=pkl.HIGHEST_PROTOCOL)

def load_model(path):
    with open("eitherio_tokenizer.pkl", "rb") as f:
        tokenizer = pkl.load(f)
    return load_keras_model("eitherio_model.h5"), tokenizer

if __name__ == "__main__":
    BASE_DIR = '/code'
    GLOVE_DIR = os.path.join(BASE_DIR, 'glove.6B')
    # TEXT_DATA_DIR = os.path.join(BASE_DIR, '20_newsgroup')
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    # VALIDATION_SPLIT = 0.2

    # first, build index mapping words in the embeddings set
    # to their embedding vector

    print('Indexing word vectors.')

    embeddings_index = {}
    with open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt')) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            embeddings_index[word] = coefs

    print('Found %s word vectors.' % len(embeddings_index))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    texts = []  # list of text samples (two by two, because there are two choices)
    labels = []  # list of label ids

    with open('/code/wouldyourather.csv', 'r', encoding="utf-8") as f:
        csvreader = csv.reader(f, delimiter='|')
        next(csvreader)
        for l in csvreader:
            texts.append(l[:2])
            labels.append(x_percentage(float(l[2]), float(l[3])))

    print('Found %s wouldyourather choices.' % len(texts))

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts([ t for options in texts for t in options ])
    # tokenizer.fit_on_texts([ item for sublist in texts for item in sublist ])
    blue_sequences = tokenizer.texts_to_sequences([ t[0] for t in texts ])
    red_sequences = tokenizer.texts_to_sequences([ t[1] for t in texts ])
    # sequences = [ tokenizer.texts_to_sequences(t) for t in texts ]

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    blue_data = pad_sequences(blue_sequences + red_sequences, maxlen=MAX_SEQUENCE_LENGTH)
    red_data = pad_sequences(red_sequences + blue_sequences, maxlen=MAX_SEQUENCE_LENGTH)

    labels = np.asarray(labels + [1-l for l in labels])
    print('Shape of data tensor:', blue_data.shape)
    print('Shape of label tensor:', labels.shape)

    # # split the data into a training set and a validation set
    indices = np.arange(blue_data.shape[0])
    np.random.shuffle(indices)
    blue_data = blue_data[indices]
    red_data = red_data[indices]
    labels = labels[indices]
    # num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    # x_train = data[:-num_validation_samples]
    # y_train = labels[:-num_validation_samples]
    # x_val = data[-num_validation_samples:]
    # y_val = labels[-num_validation_samples:]

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    print('Training model.')

    # train a 1D convnet with global maxpooling
    blue_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    red_sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    # blue_embedded_sequences = embedding_layer(blue_sequence_input)
    # red_embedded_sequences = embedding_layer(red_sequence_input)

    convnet = neural_net()
    blue_features = convnet(blue_sequence_input)
    red_features = convnet(red_sequence_input)
    features = Concatenate()([blue_features, red_features])

    prediction = Dense(1, activation='sigmoid')(features)

    model = Model([blue_sequence_input, red_sequence_input], prediction)
    model.compile(loss='mse',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # model.fit([blue_data, red_data], labels,
    #           batch_size=128,
    #           epochs=20
    #           # , validation_data=(x_val, y_val)
    # )
