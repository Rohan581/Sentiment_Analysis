import re
from nltk.tokenize.treebank import TreebankWordDetokenizer
import gensim
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.callbacks import ModelCheckpoint
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from keras import regularizers
import os
from uvicorn import run


csv_data = pd.read_csv('airline_sentiment_analysis.csv')  # Reads the csv data

train = csv_data[['airline_sentiment', 'text']]  # Filtering out the required dataframes


def purify_data(data):
    """

    :param data: any sentence that needs to be cleaned
    :return: returns the sentence after removing url's,  double quotes,removes words consisting of
     two or fewer letters e.t.c.
    """
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    data = url_pattern.sub(r'', data)
    data = re.sub('\S*@\S*\s?', '', data)
    data = re.sub('\.', '', data)
    data = re.sub('\s+', ' ', data)
    data = re.sub("\'", "", data)
    data = re.sub(r'"', '', data)

    data_list = re.split(" ", data)
    data = [word for word in data_list if len(word) > 2]
    data = " ".join(data)

    return data


def sent_to_words(sentences):
    """

    :param sentences: the sentences that need to be processed/
    :return: words in the sentence after removing punctuation marks
    """
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations


def detokenize(text):
    """

    :param text: sentence words
    :return: full sentence
    """
    return TreebankWordDetokenizer().detokenize(text)


class preprocess:
    def __init__(self, train):
        """

        :param train: the dataframe to be processed
        """
        self.train = train
        self.temp = []
        # Splitting pd.Series to list
        data_to_list = self.train['text'].values.tolist()
        for i in range(len(data_to_list)):
            self.temp.append(purify_data(data_to_list[i]))  # purifying each sentence and adding them to a list

    def modify_text(self):
        self.data_words = list(sent_to_words(self.temp))
        self.data = []
        for i in range(len(self.data_words)):
            self.data.append(detokenize(self.data_words[i]))  # separating words from the data
        self.data = np.array(self.data)

    def tockenize_func(self):
        self.labels = np.array(self.train['airline_sentiment'])
        y = []
        for i in range(len(self.labels)):
            if self.labels[i] == 'positive':
                y.append(1)
            else:
                y.append(0)
        y = np.array(y)
        self.labels = tf.keras.utils.to_categorical(y, 2, dtype="float32")  # encoding the sentiment in numerical values

        self.max_words = 5000
        self.max_len = 200

        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(self.data)
        sequences = self.tokenizer.texts_to_sequences(self.data)
        self.tweets = pad_sequences(sequences, maxlen=self.max_len)  # transforms array of texts into 2D numeric arrays.

    def train_df(self): # Training a 1D Convolutional Neural Network
        X_train, X_test, y_train, y_test = train_test_split(self.tweets, self.labels, random_state=0, test_size=0.2)  #splits the dataset into training and test set
        model = Sequential()
        model.add(layers.Embedding(self.max_words, 40, input_length=self.max_len))
        model.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-4),
                                 bias_regularizer=regularizers.l2(1e-4)))
        model.add(layers.MaxPooling1D(5))
        model.add(layers.Conv1D(20, 6, activation='relu', kernel_regularizer=regularizers.l2(l2=1e-4),
                                 bias_regularizer=regularizers.l2(1e-4)))
        model.add(layers.GlobalMaxPooling1D())
        model.add(layers.Dense(2, activation='sigmoid'))
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        checkpoint3 = ModelCheckpoint("model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto',
                                      period=1, save_weights_only=False)
        history = model.fit(X_train, y_train, epochs=70, validation_data=(X_test, y_test), callbacks=[checkpoint3])

    def choose_model(self):
        best_model = keras.models.load_model("model.hdf5")
        return best_model

    def test(self, text):
        best_model = self.choose_model()
        sentiment = ['Negative', 'Positive']
        self.text = text
        self.text = purify_data(self.text)
        print(self.text)
        sequence = self.tokenizer.texts_to_sequences([self.text])
        test = pad_sequences(sequence, maxlen=self.max_len)
        prob = best_model.predict(test).argmax(axis=1)[0]
        prediction = sentiment[prob]

        return prediction


p = preprocess(train)
p.modify_text()
p.tockenize_func()
# p.train_df()

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Welcome to the Sentiment Analysis API!"}


@app.post('/predict')
async def get_sentiment_pred(sentence: str = ""):
    if sentence == "":
        return {"message": "No text provided"}

    prediction = p.test(sentence)

    return {"Sentiment": prediction}


def my_schema():
    openapi_schema = get_openapi(
        title="Sentiment Analysis",
        version="1.0",
        description="Learn about the sentiment meant by any sentence.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = my_schema

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    run(app, host="127.0.0.1", port=port)


