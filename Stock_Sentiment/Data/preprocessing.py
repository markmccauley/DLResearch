import string
import textmining
import json
import glob
import nltk
import re
import math
import statistics
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict

lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

documents = sorted(glob.glob("/home/mark/Research/data/stocks/*.json"))

lens = []
for doc in documents:
	preprocessed = []

	try:	
		with open(doc) as data:
			data = json.load(data)
			convos = data['Conversations']
					
		for c in convos:
			processed = ''.join(i for i in c if i not in string.punctuation)
			processed = processed.split()
			processed = ' '.join(i for i in processed if i not in stop_words)
			processed = processed.lower()
			processed = processed.split()
			processed = processed[0:60]
			processed = ' '.join(i for i in processed if not i.isdigit())
			preprocessed.append(processed)
			
		data['Conversations'] = preprocessed
		with open('/home/mark/Research/dl/processedData/'+ data['Ticker'] + 'processed' + '.json', 'w') as datafile:
			json_string = json.dump(data, datafile, indent=2)

	except KeyError:
		print "preprocessing failed"	
