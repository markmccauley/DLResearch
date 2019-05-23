import re
import os
import codecs
import numpy
import pandas 
import tensorflow

home_dir = '/home/mark/Research/'

# returns weight matrix and turns dictionary from words to indices
def load_embeddings(emb_path):
	weight_vecs = []
	word_index = {}
	with codecs.open(emb_path, encoding = 'utf-8') as f:
		for i in f:
			word, vec = i.split(u' ', 1)
			word_index[word] = len(weight_vecs)
			weight_vecs.append(numpy.array(vec.split(), dtype=numpy.float32))

	word_index[u'-LRB-'] = word_index.pop(u'(')
	word_index[u'-RRB-'] = word_index.pop(u')')
	weight_vecs.append(numpy.random.uniform(-0.05, 0.05, weight_vecs[0].shape).astype(numpy.float32))

	return numpy.stack(weight_vecs), word_index

def read_data(home_dir):
	processed_data = pandas.read_csv(home_dir + '/words_w_scores.txt', sep='\t')
	processed_data = processed_data['word|sentiment'].str.split('|', expand=True)
	processed_data = processed_data.rename(columns={0: 'word', 1:'sentiment'})

	return processed_data	

# split our data into 3 parts: train - 50%, test - 25%, validate - 25% 
def split_data(data, percent, data_dir):
	m = numpy.random.rand(len(data)) < percent
	train = data[m]
	tandd = data[~m]
	
	mt = numpy.random.rand(len(tandd)) < 0.5
	test = tandd[mt]
	val = tandd[~mt]
	
	val.to_csv(os.path.join(data_dir, 'val.csv'))
	test.to_csv(os.path.join(data_dir, 'test.csv'))
	train.to_csv(os.path.join(data_dir, 'train.csv'))

	return train, test, val

def pipeline(data, word_index, weight_matrix):
	rows = len(data)
	ids = numpy.zeros((rows, 1), dtype='int32')

	word_index = {k.lower(): v for k, v in word_index.items()}
	i = 0
	
	for index, row in data.iterrows():
		word = row['word']
		word = word.strip()
		try:
			ids[i][0] = word_index[word]
		except Exception as e:
			if str(e) == word:
				ids[i][0] = 0
			continue
		i += 1

	return ids

def labels(data):
	ydata = data['sentiment']
	yfloat = ydata.astype(float)

	cats = ['0','1','2','3','4','5','6','7','8','9']
	mult = (yfloat*10).astype(int)
	dummies = pandas.get_dummies(mult, prefix='', prefix_sep='')
	dummies = dummies.T.reindex(cats).T.fillna(0)
	label_list = dummies.values

	return label_list
