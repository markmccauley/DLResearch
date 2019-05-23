import os
import glob
import json
import numpy
import h5py
import functions
import statistics
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Bidirectional
from keras.preprocessing import sequence
from keras.layers import Dropout
from keras.models import model_from_json
from keras.models import load_model
from nltk.tokenize import RegexpTokenizer
from collections import OrderedDict
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

path = '/home/mark/Research'
data_dir = path + '/data'
weight_matrix, word_index = functions.load_embeddings(path + '/data/glove.6B.100d.txt')

data = functions.read_data(data_dir)
train, test, val = functions.split_data(data, .8, data_dir)

train = train.reset_index()
test = test.reset_index()
val = val.reset_index()

#max_length, avg_words, seq_length = functions.maxLen(data)
train_x = functions.pipeline(train, word_index, weight_matrix)
test_x = functions.pipeline(test, word_index, weight_matrix)
val_x = functions.pipeline(val, word_index, weight_matrix)

train_y = functions.labels(train)
test_y = functions.labels(test)
val_y = functions.labels(val)

model = Sequential()
model.add(Embedding(len(weight_matrix), 100, weights=[weight_matrix], input_length=70, trainable=False))
model.add(Bidirectional(LSTM(128, dropout=0.4, recurrent_dropout=0.4)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.load_weights(path + '/data/best.hdf5')
model.summary()

total = sorted(glob.glob("/home/mark/Research/dl/processedData/*.json"))

allt = []
output = open('/home/mark/Research/dl/results/sentiment_ratings_words.txt', 'w')
for doc in total:
	try:
		with open(doc) as d:
			d = json.load(d)
			sample_list = d['Conversations']

		doc_sentiment = []
		for data_sample in sample_list:
			plist = []
			plist_np = numpy.zeros((70,1))

			tokenizer = RegexpTokenizer(r'\w+')
			data_sample_list = tokenizer.tokenize(data_sample)

			labels = numpy.array(['1','2','3','4','5','6','7','8','9','10'], dtype = "int")

			data_index = numpy.array([word_index[word.lower().strip()] if word.lower().strip() in word_index else 0 for word in data_sample_list])

			data_index_np = numpy.array(data_index)
			padded = numpy.zeros(70)

			padded[:data_index_np.shape[0]] = data_index_np
			data_index_np_pad = padded.astype(int)
			plist.append(data_index_np_pad)
			plist_np = numpy.asarray(plist)
			type(plist_np)
	
			score = model.predict(plist_np, batch_size=1, verbose=0)
			single_score = numpy.round(numpy.argmax(score)/10, decimals=2)

			best3idx = numpy.argsort(score)[0][-3:]
			best3score = score[0][best3idx]
			best3weight = best3score / numpy.sum(best3score)
			sentiment = numpy.round(numpy.dot(best3idx, best3weight)/10, decimals=2)
			doc_sentiment.append(sentiment)
			allt.append(sentiment)

		stock_sentiment = statistics.mean(doc_sentiment)
		output.write(d['Ticker'] + ' | ' + str(stock_sentiment) + '\n')

	except KeyError:
		print "failed"

print statistics.mean(allt)
print max(allt)
print min(allt)

print 'Mean: ', statistics.mean(allt)
print 'Max: ', max(allt)
print 'Min: ', min(allt)

pos = 0
neg = 0
neutral = 0
for i in allt:
    if i < 0.55 and i > 0.45:
        neutral += 1
    elif i < 0.45:
        neg += 1
    else:
        pos += 1

print 'Positive: ', pos
print 'Negetive: ', neg
print 'Neutral: ', neutral

