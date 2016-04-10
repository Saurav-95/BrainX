import gensim, logging, nearpy, nltk, numpy as np
from nearpy.hashes import RandomBinaryProjections
from nltk.tokenize import RegexpTokenizer
from gensim.models import word2vec
from nearpy import Engine
np.seterr(divide='ignore', invalid='ignore')

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
			
class Question_Similarity :
	"""
	Class for finding a set of all similar Questions to a given Query Question.
	"""
	
	def __init__(self, dim, train_size, n_no) :
		self.q_list = [];
		self.q_actual = [];
		self.vectors = 0;
		self.q_total = 0;
		self.dim = dim;
		self.n_no = n_no;
		self.train_size = train_size;
			
	def load_dataset(self, fname) :
		"""
		Loads Quora Dataset of Questions from the file "fname" given as a parameter
		which contains both Given Question set and Query Question set.
		"""
		qin = open(fname,"r");
		line = qin.readline();
		while line != '' :
			if (line[0].isdigit()) :
				line = qin.readline();
				continue;
			else :
				self.q_actual.append(line);
				tokenizer = RegexpTokenizer(r'\w+|\n');
				token_str = ' '.join(tokenizer.tokenize(line));
				self.q_list.append(token_str.lower());
				line = qin.readline();
		qin.close();
		
	def ques_feature(self) :
		"""
		Finds the feature vector for all the Questions in the dataset using Word2Vec 
		and NearPy.
		"""
		self.q_total = len(self.q_list);
		self.vectors = np.zeros((self.q_total, self.dim));
		for q_no in range(self.q_total) :
			w_no = 0;
			for words in self.q_list[q_no].split() :
				if words not in stopwords and words in model.vocab:
					self.vectors[q_no] = self.vectors[q_no] + model[words];
					w_no = w_no + 1;
			self.vectors[q_no] /= w_no;

	def nearest_neighbour(self, fname) :
		"""
		Finds the "n_no" of nearest neighbours for each Query Question and writes it 
		in a file "fname" given as a parameter.
		"""
		rbp = RandomBinaryProjections('rbp', self.n_no);		# Create a random binary hash with 10 bits
		engine = Engine(self.dim, lshashes=[rbp]);				# Create engine with pipeline configuration
		qout = open(fname,"w");
		for i in range(self.train_size+1) :
			engine.store_vector(np.transpose(self.vectors[i]), i);
	
		for i in range(self.train_size+1,self.q_total) :
			N = engine.neighbours(np.transpose(self.vectors[i]));
			qout.write(self.q_actual[i]);
			for j in range(len(N)) :
				qout.write("NN %d --- " % (j+1)+self.q_actual[N[j][1]]);
		qout.close();		
	
if __name__ == "__main__" :
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	model = word2vec.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
	qs = Question_Similarity(300, 19996, 10);
	qs.load_dataset("labeler_sample.in");
	qs.ques_feature();
	qs.nearest_neighbour("Similar.txt");