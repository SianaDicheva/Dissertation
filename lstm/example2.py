import csv
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Dropout, Activation, Embedding, Bidirectional
from keras.callbacks import ModelCheckpoint


import nltk


print(tf.__version__)

def lempel_ziv_complexity(binary_sequence):
    """Lempel-Ziv complexity for a binary sequence, in simple Python code."""
    u, v, w = 0, 1, 1
    v_max = 1
    length = len(binary_sequence)
    complexity = 1
    while True:
        if binary_sequence[u + v - 1] == binary_sequence[w + v - 1]:
            v += 1
            if w + v >= length:
                complexity += 1
                break
        else:
            if v > v_max:
                v_max = v
            u += 1
            if u == w:
                complexity += 1
                w += v_max
                if w > length:
                    break
                else:
                    u = 0
                    v = 1
                    v_max = 1
            else:
                v = 1
    return complexity


nltk.download('stopwords')
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

vocab_size = 5000 
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '' #OOV = Out of Vocabulary
training_portion = .8

articles = []
labels = []

with open("/bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)
print(len(labels))
print(len(articles))

labels[:2]

articles[:2]

train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

print("train_size",  train_size)
print(f"train_articles {len(train_articles)}")
print("train_labels", len(train_labels))
print("validation_articles", len(validation_articles))
print("validation_labels", len(validation_labels))

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

train_sequences = tokenizer.texts_to_sequences(train_articles)

print(train_sequences[10]), len(train_sequences[10])

print(train_sequences[0]), len(train_sequences[0])

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

train_padded[10]



print("len train_sequnces[0]: ", len(train_sequences[0]))
print("len train_padded[0]: ", len(train_padded[0]))

print("len train_sequences[1]: ", len(train_sequences[1]))
print("len train_padded[1]: ", len(train_padded[1]))

print("len train_sequences[10]: ", len(train_sequences[10]))
print("len train_padded[10]: ", len(train_padded[10]))

validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

print(set(labels))


label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

label_tokenizer.word_index


print(training_label_seq[0])
print(training_label_seq[1])
print(training_label_seq[2])
print(training_label_seq.shape)
print('-------------')
print(validation_label_seq[0])
print(validation_label_seq[1])
print(validation_label_seq[2])
print(validation_label_seq.shape)

model = Sequential()

model.add(Embedding(vocab_size, embedding_dim))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(embedding_dim)))
model.add(Dense(6, activation='softmax'))

model.summary()

epochs = 50
learning_rate = 0.001
decay_rate = learning_rate / epochs
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate, decay=1e-6)


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'],
)

output_layers = []

# def get_output_layer(epoch, logs):
#     output_layer = model.predict(train_padded)
#     output_layers.append(output_layer)

# def get_output_layer(epoch, logs):
#     output_layer = model.layers[2].output # index 2 corresponds to the LSTM layer
#     intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=output_layer)
#     intermediate_output = intermediate_layer_model.predict(train_padded)
#     binary_output = (intermediate_output > 0.5).astype(int)
#     output_layers.append(binary_output)

output_layers = []
complexities = []

def get_output_layer(epoch, logs):
    output_layer = model.layers[2].output # index 2 corresponds to the LSTM layer
    intermediate_layer_model = tf.keras.models.Model(inputs=model.input, outputs=output_layer)
    intermediate_output = intermediate_layer_model.predict(train_padded)
    binary_output = (intermediate_output > 0.5).astype(int) # Convert to binary
    complexity = lempel_ziv_complexity(binary_output[0]) # Calculate complexity
    complexities.append(complexity)


# callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=get_output_layer)

num_epochs = 10



# model.fit(train_padded, train_labels, epochs=10, batch_size=32, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=get_output_layer)])

history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=get_output_layer)])

print("outputlayer is" , output_layers)

#  = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)





import matplotlib.pyplot as plt

def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
  
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

plt.plot(range(1, len(complexities)+1), complexities)
plt.savefig("/plot.png")
plt.show()


txt = ["house prices show slight increase prices of homes in the uk rose a seasonally adjusted 0.5% in february  says the nationwide building society.  the figure means the annual rate of increase in the uk is down to 10.2%  the lowest rate since june 2001. the annual rate has halved since august last year  as interest rises have cooled the housing market. at the same time  the number of mortgage approvals fell in january to a near 10-year low  official bank of england figures have shown.  nationwide said that in january house prices went up by 0.4% on the month and by 12.6% on a year earlier.  we are not seeing the market collapsing in the way some had feared   said nationwide economist alex bannister. there have been a number of warnings that the uk housing market may be heading for a downturn after four years of strong growth to 2004. in november  barclays  which owns former building society the woolwich  forecast an 8% fall in property prices in 2005  followed by further declines in 2006 and 2007. and last summer  economists at pricewaterhousecoopers (pwc) warned house prices were overvalued and could fall by between 10% and 15% by 2009.  the price of an average uk property now stands at £152 879. homeowners now expect house prices to rise by 1% over the next six months  mr bannister said. he said if the growth continued at this level then the bank of england may increase interest rates from their current 4.75%.   i think the key is what the bank expects to happen to the housing market. we always thought we would see a small rise  they thought they would see a small decline.  house prices have risen 0.9% this year  nationwide said  and if this pace of increase persists  prices would rise by just under 6% in the year to december. this is slightly above the 0-5% range nationwide predicts.  further evidence of a slowdown in the housing market emerged from bank of england lending figures released on tuesday. new mortgage loans in january fell to 79 000 from 82 000 in december  the bank said. the past few months have seen approvals fall to levels last seen in 1995. the bank revealed that 48 000 fewer mortgages were approved in january than for the same month in 2004. overall  mortgage lending rose by £7.2bn in january  marginally up on the £7.1bn rise in december."]

seq = tokenizer.texts_to_sequences(txt)
padded = pad_sequences(seq, maxlen=max_length)
pred = model.predict(padded)
labels = ['sport', 'bussiness', 'politics', 'tech', 'entertainment'] #orig

print(pred)
print(np.argmax(pred))
print(labels[np.argmax(pred)-1])


