import retinasdk, operator, json
import logging, nltk, numpy as np
from nltk.tokenize import RegexpTokenizer
from threading import Thread
np.seterr(divide='ignore', invalid='ignore')
fullClient = retinasdk.FullClient("c3412e70-f345-11e5-8378-4dad29be0fab", apiServer="http://api.cortical.io/rest", retinaName="en_associative")

stopwords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am','among', 
			'an','and','any','are','as','at','be','because','been','but','by','can',
			'cannot','could','dear','did','do','does','either','else','ever','every',
			'for','from','get','got','had','has','have','he','her','hers','him','his',
			'how','however','i','if','in','into','is','it','its','just','least','let',
			'like','likely','may','me','might','most','must','my','neither','no','nor',
			'not','of','off','often','on','only','or','other','our','own','rather','said',
			'say','says','she','should','since','so','some','than','that','the','their',
			'them','then','there','these','they','this','tis','to','too','twas','us',
			'wants','was','we','were','what','when','where','which','while','who',
			'whom','why','will','with','would','yet','you','your']
			
class Question_Similarity:
	"""
	Class for finding a set of all similar Questions to a given Query Question.
	"""
	
	def __init__(self, train_size, n_no):
		self.q_list = []
		self.q_actual = []
		self.vectors = []
		self.q_total = 0
		self.n_no = n_no
		self.train_size = train_size
			
	def load_dataset(self, fname):
		"""
		Loads Quora Dataset of Questions from the file "fname" given as a parameter
		which contains both Given Question set and Query Question set.
		"""
		qin = open(fname,"r")
		line = qin.readline()
		while line != '':
			if (line[0].isdigit()):
				line = qin.readline()
				continue
			else :
				self.q_actual.append(line)
				tokenizer = RegexpTokenizer(r'\w+|\n')
				token_str = ' '.join(tokenizer.tokenize(line))
				self.q_list.append(token_str.lower())
				line = qin.readline()
		qin.close()
		
	"""	
	def ques_feature(self):
		
		Finds the feature vector for all the Questions in the dataset using Word2Vec 
		and NearPy.
		
		self.q_total = len(self.q_list)
		for q_no in range(self.q_total):
			self.vectors.append(fullClient.getFingerprintForText(self.q_list[q_no]))
	"""		
	
	def func(self, comparison, i, res):
		res[i] = fullClient.compareBulk(json.dumps(comparison))
	
	def compare(self, ques):
		"""
		Calculates distance of 'ques' from all other questions and returns a list of 
		indices of 'n_no' most similar questions
		"""
		#qin = open("output.txt", "w")
		d = {}
		d_sorted = []
		comparison = []
		for q_no in range(self.q_total):
			comparison.append([{"text": ques}, {"text": self.q_actual[q_no]}])
			
		num_threads = 100
		threads = [None] * num_threads
		res = [None] * num_threads
		for i in range(num_threads):
			#qin.write(str(comparison[i*int(self.q_total/num_threads):min(self.q_total,(i+1)*int(self.q_total/num_threads))])+"\n\n")
			threads[i] = Thread(target=self.func, args=(comparison[i*int(self.q_total/num_threads):min(self.q_total,(i+1)*int(self.q_total/num_threads))],i,res))
			threads[i].start()
		
		for j in range(num_threads):
			threads[j].join()	
		
		for i in range(num_threads-1):
			for j in range(int(self.q_total/num_threads)):
				d[i*int(self.q_total/num_threads)+j] = res[i][j].weightedScoring
				
		for j in range(int(self.q_total/num_threads)):
			if i*int(self.q_total/num_threads)+j < self.q_total:
				d[i*int(self.q_total/num_threads)+j] = res[i][j].weightedScoring
			else:
				break
		
		for i in range(self.n_no):
			d_sorted.append(max(d.items(), key=operator.itemgetter(1)))
			del d[d_sorted[-1][0]]
			
		return d_sorted

	def nearest_neighbour(self, fname) :
		"""
		Finds the "n_no" of nearest neighbours for each Query Question and writes it 
		in a file "fname" given as a parameter.
		"""
		qout = open(fname,"w")
		self.q_total = len(self.q_list)
		for i in range(self.train_size+1,self.train_size+10):
			print(i)
			qout.write(self.q_actual[i])
			distance = self.compare(self.q_actual[i])
			for j in range(self.n_no):
				qout.write("NN %d (%f) --- " % (j+1, distance[j][1])+self.q_actual[distance[j][0]])
		qout.close()		
	
if __name__ == "__main__" :
	#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	qs = Question_Similarity(19996, 10)
	qs.load_dataset("labeler_sample.in")
	#qs.ques_feature()
	qs.nearest_neighbour("Similar.txt")