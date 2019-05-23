import json
import collections

with open('NTUSD_Fin_word_v1.0.json') as data:
	data = json.load(data)
	sents = {}
	for d in data:
		if d['market_sentiment'] < 0:
			sents[d['token']] = d['market_sentiment'] / -250
		else:
			sents[d['token']] = d['market_sentiment'] / 1.224

	sortedsent = collections.OrderedDict(sents)
	
	out = open('words_w_scores.txt', 'w')
	out.write('word|sentiment\n')
	for i in sortedsent:
		out.write(i.encode('utf-8') + ' | ' + str(sortedsent[i]) + '\n')

	out.close()
