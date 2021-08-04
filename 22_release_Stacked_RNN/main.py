import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

with open('text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')       # Убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '', text)     # Заменяем все символы кроме кириллицы на пустые символы

max_words_count = 1000
tokenizer = Tokenizer(num_words=max_words_count, filters='!–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»',
                      lower=True, split=' ', char_level=False)
tokenizer.fit_on_texts([text])

dist = list(tokenizer.word_counts.items())
print(dist[:10])

data = tokenizer.texts_to_sequences([text])
res = np.array(data[0])

inp_words = 3
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words] for i in range(n)])
Y = to_categorical(res[inp_words:], num_classes=max_words_count)

model = Sequential()
model.add(Embedding(max_words_count, 256, input_length=inp_words))
model.add(SimpleRNN(128, activation='tanh', return_sequences=True))
model.add(SimpleRNN(64, activation='tanh'))
model.add(Dense(max_words_count, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics='accuracy', optimizer='adam')
history = model.fit(X, Y, batch_size=32, epochs=50)


def build_phrase(texts, str_len=50):
    res = texts
    data = tokenizer.texts_to_sequences([text])[0]

    for i in range(str_len):
        x = data[i:i + inp_words]
        inp = np.expand_dims(x, axis=0)

        pred = model.predict(inp)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)

        res += ' ' + tokenizer.index_word[indx]

    return res


res = build_phrase('жить здорово и')
print(res)
