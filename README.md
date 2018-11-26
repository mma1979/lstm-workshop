
## Problem Description


The problem that we will use to demonstrate sequence learning in this tutorial is the IMDB movie review sentiment classification problem. Each movie review is a variable sequence of words and the sentiment of each movie review must be classified.

The Large Movie Review Dataset (often referred to as the IMDB dataset) contains 25,000 highly-polar movie reviews (good or bad) for training and the same amount again for testing. The problem is to determine whether a given movie review has a positive or negative sentiment.

The data was collected by Stanford researchers and was used in a 2011 paper where a split of 50-50 of the data was used for training and test. An accuracy of 88.89% was achieved.

Keras provides access to the IMDB dataset built-in. The imdb.load_data() function allows you to load the dataset in a format that is ready for use in neural network and deep learning models.

The words have been replaced by integers that indicate the ordered frequency of each word in the dataset. The sentences in each review are therefore comprised of a sequence of integers.

## Word Embedding

We will map each movie review into a real vector domain, a popular technique when working with text called word embedding. This is a technique where words are encoded as real-valued vectors in a high dimensional space, where the similarity between words in terms of meaning translates to closeness in the vector space.

Keras provides a convenient way to convert positive integer representations of words into a word embedding by an Embedding layer.

We will map each word onto a 32 length real valued vector. We will also limit the total number of words that we are interested in modeling to the 5000 most frequent words, and zero out the rest. Finally, the sequence length (number of words) in each review varies, so we will constrain each review to be 500 words, truncating long reviews and pad the shorter reviews with zero values.

Now that we have defined our problem and how the data will be prepared and modeled, we are ready to develop an LSTM model to classify the sentiment of movie reviews.

## Simple LSTM for Sequence Classification

We can quickly develop a small LSTM for the IMDB problem and achieve good accuracy.

Let’s start off by importing the classes and functions required for this model and initializing the random number generator to a constant value to ensure we can easily reproduce the results.


```python
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
numpy.random.seed(7)
```

    Using TensorFlow backend.
    

We need to load the IMDB dataset. We are constraining the dataset to the top 5,000 words. We also split the dataset into train (50%) and test (50%) sets.


```python
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
```

    Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz
    17465344/17464789 [==============================] - 669s 38us/step
    

Next, we need to truncate and pad the input sequences so that they are all the same length for modeling. The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, but same length vectors is required to perform the computation in Keras.


```python
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
```

We can now define, compile and fit our LSTM model.

The first layer is the Embedded layer that uses 32 length vectors to represent each word. The next layer is the LSTM layer with 100 memory units (smart neurons). Finally, because this is a classification problem we use a Dense output layer with a single neuron and a sigmoid activation function to make 0 or 1 predictions for the two classes (good and bad) in the problem.

Because it is a binary classification problem, log loss is used as the loss function (binary_crossentropy in Keras). The efficient ADAM optimization algorithm is used. The model is fit for only 2 epochs because it quickly overfits the problem. A large batch size of 64 reviews is used to space out weight updates.


```python
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    lstm_1 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dense_1 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 213,301
    Trainable params: 213,301
    Non-trainable params: 0
    _________________________________________________________________
    None
    Train on 25000 samples, validate on 25000 samples
    Epoch 1/3
    25000/25000 [==============================] - 188s 8ms/step - loss: 0.4504 - acc: 0.7870 - val_loss: 0.3595 - val_acc: 0.8468
    Epoch 2/3
    25000/25000 [==============================] - 193s 8ms/step - loss: 0.3304 - acc: 0.8620 - val_loss: 0.3653 - val_acc: 0.8530
    Epoch 3/3
    25000/25000 [==============================] - 190s 8ms/step - loss: 0.2865 - acc: 0.8864 - val_loss: 0.3394 - val_acc: 0.8562
    




    <keras.callbacks.History at 0x2aae59a6668>




```python
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    Accuracy: 85.62%
    

## LSTM For Sequence Classification With Dropout


Dropout can be applied between layers using the Dropout Keras layer. We can do this easily by adding new Dropout layers between the Embedding and LSTM layers and the LSTM and Dense output layers. 


```python
from keras.layers import Dropout
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 500, 32)           0         
    _________________________________________________________________
    lstm_2 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 100)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 213,301
    Trainable params: 213,301
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/3
    25000/25000 [==============================] - 157s 6ms/step - loss: 0.4738 - acc: 0.7675
    Epoch 2/3
    25000/25000 [==============================] - 175s 7ms/step - loss: 0.3097 - acc: 0.8772
    Epoch 3/3
    25000/25000 [==============================] - 180s 7ms/step - loss: 0.3852 - acc: 0.8394
    Accuracy: 85.46%
    

## LSTM and Convolutional Neural Network For Sequence Classification

Convolutional neural networks excel at learning the spatial structure in input data.

The IMDB review data does have a one-dimensional spatial structure in the sequence of words in reviews and the CNN may be able to pick out invariant features for good and bad sentiment. This learned spatial features may then be learned as sequences by an LSTM layer.

We can easily add a one-dimensional CNN and max pooling layers after the Embedding layer which then feed the consolidated features to the LSTM. We can use a smallish set of 32 features with a small filter length of 3. The pooling layer can use the standard length of 2 to halve the feature map size.


```python
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 500, 32)           3104      
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 250, 32)           0         
    _________________________________________________________________
    lstm_3 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dense_3 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 216,405
    Trainable params: 216,405
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/3
    25000/25000 [==============================] - 73s 3ms/step - loss: 0.4243 - acc: 0.7908
    Epoch 2/3
    25000/25000 [==============================] - 77s 3ms/step - loss: 0.2438 - acc: 0.9041
    Epoch 3/3
    25000/25000 [==============================] - 79s 3ms/step - loss: 0.2017 - acc: 0.9224
    Accuracy: 88.15%
    

## Using keras callbacks to log training process and view using tensorboard

By using a TensorBoard callback, logs will be written to a directory that you can then examine with TensorFlow’s excellent TensorBoard visualization tool. It even works (to an extent) if you’re using a backend other than TensorFlow, like Theano, or CNTK (if you’re a glutton for punishment).


```python
from keras.callbacks import TensorBoard
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
#declar the log callback
tbCallBack = TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)
#add the callback to fit() callbacks parameter
model.fit(X_train, y_train, epochs=3, batch_size=64, callbacks=[tbCallBack])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_6 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 500, 32)           3104      
    _________________________________________________________________
    max_pooling1d_3 (MaxPooling1 (None, 250, 32)           0         
    _________________________________________________________________
    lstm_5 (LSTM)                (None, 100)               53200     
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 101       
    =================================================================
    Total params: 216,405
    Trainable params: 216,405
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/3
    25000/25000 [==============================] - 72s 3ms/step - loss: 0.4172 - acc: 0.7971
    Epoch 2/3
    25000/25000 [==============================] - 75s 3ms/step - loss: 0.2417 - acc: 0.9056
    Epoch 3/3
    25000/25000 [==============================] - 74s 3ms/step - loss: 0.1961 - acc: 0.9262
    Accuracy: 88.50%
    


```python

```
