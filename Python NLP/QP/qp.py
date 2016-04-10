import logging, nltk, numpy as np
from nltk import *
from nltk.corpus import wordnet as wn
from sklearn.svm import LinearSVC
np.seterr(divide='ignore', invalid='ignore')
			
class Question_Processing:
	"""
	Class for Processing Questions from the dataset for Classification.
	"""
	
	def __init__(self):
		self.q_list = []
		self.q_actual = []
		self.coarse_label = []
		self.fine_label = []
		self.q_total = 0

	def parse(self, question):
		sentence = nltk.sent_tokenize(question)
		sentence = [nltk.word_tokenize(sent) for sent in sentence]
		sentence = [nltk.pos_tag(sent) for sent in sentence]
		#sentence = [nltk.ne_chunk(sent) for sent in sentence]
		return sentence

	def wh_words(self, question):
		sentence = self.parse(question)
		for sent, tag in sentence[0]:
			if tag in ("WP","WRB","WDT","WP$"):
				return sent
		return sentence[0][0][0]
			

	def head_word(self, question):
		words = question.split()
		if words[0] in ['When','Where','Why']:
			return ''
		if words[0] == 'How':
			return words[1]
		if words[0] == 'What':
			if words[1] in ['is','are'] and words[2] in ['a','an','the'] and len(words) < 6:
				return "DESC"
			if words[1] in ['do','does'] and words[-2] == 'mean':
				return "DESC"
			if words[1] in ['is','are'] and words[-3:-2] in ['composed of','made of']:
				return "ENTY"
			if words[1] == 'does' and words[-2] == 'do':
				return "DESC"
			if words[1:3] == 'do you call':
				return "ENTY"
			if words[1] in ['causes','cause']:
				return "DESC"
			if words[1] in ['is','are'] and words[-3:-2] == 'used for':
				return "DESC"
			if words[1] in ['does','do'] and words[-3:-2] == 'stand for':
				return "ABBR"
		if words[0] == 'Who' and words[1] in ['is','was'] and words[2][0].isupper():
			return "HUM"
		sentence = self.parse(question)
		grammar = "NP: {<DT>?<JJ>*<NN|NNS>}"
		cp = nltk.RegexpParser(grammar)
		result = cp.parse(sentence[0])
		head = ''
		for i in result.subtrees(filter=lambda x: x.label() == 'NP'):
			for j in i:
				if j[1] in ['NN','NNS']:
					#print i
					head = j[0] 
					break
			break
		return head

	def unigrams(self, question):
		return list(ngrams(question.split(),1))

	def bigrams(self, question):
		return list(ngrams(question.split(),2))

	def trigrams(self, question):
		return list(ngrams(question.split(),3))

	def nouns(self, question):
		list = []
		sentence = self.parse(question)
		#print [sent for (sent,tag) in sentence[0] if tag in ("NN","NNS","NNP","NNPS")]
		list.append([sent for (sent,tag) in sentence[0] if tag in ("NN","NNS","NNP","NNPS")])
		return list[0]

	def hypernyms(self, word, question):
		hyper = []
		sentence = self.parse(question)
		pos = ''
		for sent, tag in sentence[0]: 
			if sent == word:
				pos = tag
				break
		if pos in ['JJ','JJR','JJS']:
			for synset in wn.synsets(word, pos = wn.ADJ):
				for lemma in synset.lemmas():
					if lemma.name() not in hyper and len(hyper)<7:
						hyper.append(lemma.name())
		elif pos in ['NN','NNS']:
			for synset in wn.synsets(word, pos = wn.NOUN):
				for lemma in synset.lemmas():
					if lemma.name() not in hyper and len(hyper)<7:
						hyper.append(lemma.name())
		elif pos in ['VB','VBG','VBD','VBN','VBP','VBZ']:
			for synset in wn.synsets(word, pos = wn.VERB):
				for lemma in synset.lemmas():
					if lemma.name() not in hyper and len(hyper)<7:
						hyper.append(lemma.name())
		elif pos in ['RB','RBR','RBS']:
			for synset in wn.synsets(word, pos = wn.ADV):
				for lemma in synset.lemmas():
					if lemma.name() not in hyper and len(hyper)<7:
						hyper.append(lemma.name())
		return hyper

	def ques_features(self, question):
		features = {}
		features["Wh-Words"] = self.wh_words(question)
		features["Unigrams"] = str(self.unigrams(question))
		#features["Bigrams"] = str(self.bigrams(question))
		#features["Trigrams"] = str(self.trigrams(question))
		#features["Nouns"] = str(self.nouns(question))
		features["Head_Word"] = str(self.head_word(question))
		features["Hypernyms"] = str(self.hypernyms(features["Head_Word"], question))
		return features

	def ques_preprocessing(self, fname1, fname2):
		qin = open(fname1,"r")
		qout = open(fname2,"w")
		qt = open("questions.txt","w")
		#stemmer = PorterStemmer()
		features = []
		line = qin.readline()
		while line != '' :
			dict = line.split(' ', 1)
			self.q_actual.append(dict[1])
			label = dict[0].split(':',1)
			self.coarse_label = label[0]
			self.fine_label = label[1]
			#dict[1] = ' '.join([stemmer.stem(words) for words in dict[1].split()])
			features.append((self.ques_features(dict[1]), label[0]))
			qout.write(str(features[-1])+"\n\n")
			if label[0] in ['DESC', 'ENTY'] and fname1 == "QC_Testset.txt":
				qt.write(dict[1])
			line = qin.readline()
		qin.close()
                qout.close()
		return features


	def load_dataset(self, fname1, fname2, fname3, fname4):
		"""
		Loads UIUC Dataset of Questions from the file "fname1" given as a parameter
		which contains both Training and Test Question set.
		"""
		featuresets = self.ques_preprocessing(fname1, fname3)
		testfeatures = self.ques_preprocessing(fname2, fname4)
                #print [words for words in nltk.sent_tokenize(qout)];
		#print featuresets[0]
		length1 = len(featuresets)
		length2 = len(testfeatures)
		train_set, test_set = featuresets, testfeatures
		ts = []
		ts.append([fs[0] for fs in test_set]) 
		me3_classifier = SklearnClassifier(LinearSVC()).train(train_set)
		print classify.accuracy(me3_classifier, test_set)
		print classify.accuracy(me3_classifier, train_set)
		pred = me3_classifier.classify_many(ts[0])
		qpred = open("pred.txt","w")
		qpred.write(str([str(self.q_actual[i+length1])+" : "+str(pred[i]) for i in range(length2)])+"\n")
		qpred.close()
			
	
if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	qs = Question_Processing()
	qs.load_dataset("QC_Trainset.txt", "QC_Testset.txt", "features.txt", "testfeatures.txt")