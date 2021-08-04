import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import re

from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer

with open('../21_RNN_words_prediction/text.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    text = text.replace('\ufeff', '')       # Убираем первый невидимый символ
    text = re.sub(r'[^А-я ]', '', text)     # Заменяем все символы кроме кириллицы на пустые символы

# Парсим текст, как последовательность символов
num_characters = 34  # 33 + пробел
tokenizer = Tokenizer(num_words=num_characters, char_level=True)  # токенизируем на уровне символов
tokenizer.fit_on_texts([text])  # Формируем токены на основе частности в нашем тексте
print(tokenizer.word_index)

inp_chars = 3
data = tokenizer.texts_to_matrix(text)  # Преобразуем исходный текст в массив ОНЕ
n = data.shape[0] - inp_chars       # Так как мы предсказываем по трём символам -четвёртый

X = np.array([data[i:i + inp_chars, :] for i in range(n)])
Y = data[inp_chars:]  # Предсказание следующего символа

print(data.shape)

model = Sequential()
model.add(Input((inp_chars,
                 num_characters)))  # при тренировке в рекуррентные модели keras подается сразу вся последовательность, поэтому в input теперь два числа. 1-длина последовательности, 2-размер OHE
model.add(SimpleRNN(128, activation='tanh'))  # рекуррентный слой на 500 нейронов
model.add(Dense(num_characters, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

history = model.fit(X, Y, batch_size=32, epochs=100)


def build_phrase(inp_str, str_len=50):
    for i in range(str_len):
        x = []
        for j in range(i, i + inp_chars):
            x.append(tokenizer.texts_to_matrix(inp_str[j]))  # преобразуем символы в One-Hot-encoding

        x = np.array(x)
        inp = x.reshape(1, inp_chars, num_characters)

        pred = model.predict(inp)  # предсказываем OHE четвертого символа
        d = tokenizer.index_word[pred.argmax(axis=1)[0]]  # получаем ответ в символьном представлении

        inp_str += d  # дописываем строку

    return inp_str


res = build_phrase('утр')
print(res)
