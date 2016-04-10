import gensim, logging, nearpy, nltk, random, numpy as np
from nearpy.hashes import RandomBinaryProjections
from nltk.tokenize import RegexpTokenizer
from gensim.models import Doc2Vec, utils
from gensim.models.doc2vec import LabeledSentence, TaggedLineDocument
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

class FeatureExtract():
    def __init__(self, qlist):
        self.qlist = qlist
        self.sentences = []
    def __extract__(self):
        for uid in range(len(self.qlist)):
            self.sentences.append(LabeledSentence(self.qlist[uid].split(), [uid]))
            #print self.sentences[uid]
        model = Doc2Vec(alpha=0.025, min_alpha=0.025, dm=0, min_count=1)
        model.build_vocab(self.sentences)
        for epoch in range(10):
            try:
                print 'epoch %d' % (epoch)
                random.shuffle(self.sentences)
                model.train(self.sentences)
                model.alpha -= 0.002
                model.min_alpha = model.alpha
            except (KeyboardInterrupt, SystemExit):
                break
        model.save('my_model.doc2vec')

			
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
		qout = open("questions.txt","w");
		while line != '' :
			if (line[0].isdigit()) :
				line = qin.readline();
				continue;
			else :
				qout.write(line);
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
                feature = FeatureExtract(self.q_list)
                feature.__extract__()

	def nearest_neighbour(self, fname) :
		"""
		Finds the "n_no" of nearest neighbours for each Query Question and writes it 
		in a file "fname" given as a parameter.
		"""
                qout = open(fname, "w")
                model = Doc2Vec.load('my_model.doc2vec')
		for i in range(0,5) :
                        j=0
                        qout.write(self.q_actual[i]);
			for items in model.docvecs.most_similar(i) :
				qout.write("NN %d (%s) --- " % (j+1, items[1])+self.q_actual[items[0]])
                                j=j+1
                print "Written Successfully in file "+fname+" !!!"
		qout.close();		
	
if __name__ == "__main__" :
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	qs = Question_Similarity(300, 500, 10);
	qs.load_dataset("labeler_sample.in");
	qs.ques_feature();
	qs.nearest_neighbour("Similar.txt");