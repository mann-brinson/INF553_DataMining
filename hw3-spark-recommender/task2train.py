from pyspark import SparkContext
import json
import time
import itertools
import sys
import string
from string import digits
import math

#FUNCTIONS
def get_stopwords(stopwords_file):
	fd = open(stopwords_file, 'r') 
	lines = fd.readlines() 
	stopwords_list = [line.strip() for line in lines]
	stopwords_dict = {}
	for word in stopwords_list:
		stopwords_dict[word] = 1 
	return stopwords_dict

def punc_remove():
	sp_chars = ['(', '[', ',', '.', '!', '?', ':', ';', ']', ')', '#']
	del_d = {sp_char: '' for sp_char in string.punctuation} 
	del_d[' '] = ''
	t = str.maketrans(del_d) 
	return t

def tfidf_one(item_id, text, t, stopwords_dict):
	'''Pass #1: Construct tf_index and idf_index.
	item_id - in our case, this is business_id
	text - large string containing all concatenated text for item_id
	t - list of punctuation to remove
	stopwords_dict - a dict of stopwords
	total_words - counter for total words
	idf_index - a dictionary of words and the count of docs they appear within. '''

	remove_digits = str.maketrans('', '', digits)
	item_terms = {'terms': {}, 'max_word_freq': 0}
	text2 = text.replace('\n', ' ')                    
	word_list = text2.split(' ')
	for word in word_list:
		word1 = word.lower().strip()
		word2 = word1.translate(remove_digits)
		word3 = word2.translate(t)
		if word3 == '':
			continue
		#Remove stopwords
		if word3 in stopwords_dict:
			continue
		#Add to index
		if word3 not in item_terms.get('terms'):
			item_terms['terms'][word3] = {'freq': 1}
		else:
			item_terms['terms'][word3]['freq'] += 1

	#Store max_word_freq
	doc_word_list = list(item_terms.get('terms').keys())
	word_freq_list = [item_terms.get('terms').get(word).get('freq') for word in doc_word_list]
	max_word_freq = max(word_freq_list)
	item_terms['max_word_freq'] = max_word_freq

	sum_words = sum(word_freq_list)
	item_terms['sum_words'] = sum_words

	res = (item_id, item_terms)
	return res

def construct_idf_index(item_terms):
	doc_word_list = list(item_terms.get('terms').keys())
	return doc_word_list

def tfidf_two(item_id, item_terms, n_items, idf_index, total_words, z):
	'''Pass #2: Compute the tf and idf values and append to tf_index.
	total_words - the total word count of across all docs
	z - minimum word frequency in order to be included (ex: 0.000001) '''
	min_freq = z * total_words
	doc_words = list(item_terms.get('terms').keys())
	doc_words_scores = []
	for word in doc_words:
		#Compute TF
		freq = item_terms.get('terms').get(word).get('freq')
		if freq < min_freq:
			continue
		else:
			max_word_freq = item_terms.get('max_word_freq')
			tf = round((freq / max_word_freq), 4)
			#Compute IDF
			n_i = idf_index.get(word)
			idf = round(math.log2(n_items / n_i), 4)
			tf_idf = round((tf * idf), 4)
			doc_words_scores.append(((item_id, word), tf_idf))
	return doc_words_scores

def select_k_tfidf_words(doc_words_scores, k):
	'''Select top k tf_idf words from each doc'''
	doc_words_scores.sort(reverse=True, key = lambda x: x[1])
	top_k_tfidf = doc_words_scores[:k]
	return top_k_tfidf

def get_item_feats(item, item_profiles_dict):
	#input: (u1, b1)
	#output: (u1 {w1, w2, w3}) <--- ITEM PROFILE

	if item in item_profiles_dict:
		item_feats = list(item_profiles_dict.get(item))
	else:
		item_feats = [0]
	return item_feats

#PARAMETERS
train_file = sys.argv[1]
model_file = sys.argv[2]
stopwords_file = sys.argv[3]
z = 0.000001
k = 200

sc = SparkContext(appName="inf553")

#DRIVER

#### ITEM PROFILES
#Concatenate review text for each business. Consider all reviews for each business (even if user reviews business >1 time)
b_text_full = sc.textFile(train_file)\
.map(lambda x: json.loads(x))\
.map(lambda x: (x['business_id'], x['text']))\
.reduceByKey(lambda x, y: x + ' ' + y).persist()
n_items = len(b_text_full.map(lambda x: x[0]).collect()) #Run once. Expensive

#PASS 1: Calculate initial tf_index and idf_index
#NOTE: A collect upon this RDD takes 30s... maybe an issue. May need to flatten
start = time.time()
t = punc_remove()
stopwords_dict = get_stopwords(stopwords_file)
tfidf_one_rdd = b_text_full.map(lambda x: tfidf_one(x[0], x[1], t, stopwords_dict)).persist()
tfidf_one_rdd_l = tfidf_one_rdd.collect()
len(tfidf_one_rdd_l) #len = 10,253

#Derive total words from the rdd
total_words = 0
total_words_rdd = tfidf_one_rdd.map(lambda x: (1, x[1].get('sum_words')))\
.reduceByKey(lambda x, y: x + y)
total_words_l = total_words_rdd.collect()
total_words = total_words_l[0][1]

#Derive idf_index from the rdd
idf_index = {}
idf_index_rdd = tfidf_one_rdd.flatMap(lambda x: [(word, 1) for word in construct_idf_index(x[1])])\
.reduceByKey(lambda x, y: x + y)
idf_index = dict(idf_index_rdd.collect())
end = time.time()
t_time = end-start
print('tfidf pass one time: ', t_time)

#PASS 2: Append TF and IDF scores to each word in tf_index
#Select the top k tf_idf words from each doc
tfidf_two_rdd = tfidf_one_rdd.map(lambda x: tfidf_two(x[0], x[1], n_items, idf_index, total_words, z))\
.flatMap(lambda x: [res for res in select_k_tfidf_words(x, k)]).persist()

#Construct item_profile matrix
start = time.time()
item_profiles = tfidf_two_rdd.map(lambda x: (x[0][0], [(x[0][1], x[1])]))\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], dict(x[1]))).persist()

item_profiles_l = item_profiles.collect()
item_profiles_dict = {x[0]:x[1] for x in item_profiles_l}
end = time.time()
t_time = end-start
print('item_profile time: ', t_time)

#### USER PROFILES
#GOAL: For each user's business, get the business profile from the item_profiles_dict
# Count all features from all user's items, to construct user profile

#For each user, get list of businesses the user reviewed
user_bus = sc.textFile(train_file)\
.map(lambda x: json.loads(x))\
.map(lambda x: (x['user_id'], [x['business_id']]))\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], set(x[1])))\
.flatMap(lambda x: [(x[0], item) for item in x[1]]).persist()

#Get n_items for each user
user_n_items = user_bus.map(lambda x: (x[0], 1))\
.reduceByKey(lambda x, y: x + y)
user_n_items_d = dict(user_n_items.collect())

#For each user's item, get the item's features. Construct user_profiles
start = time.time()
user_profiles = user_bus.flatMap(lambda x: [((x[0], feat), 1) for feat in get_item_feats(x[1], item_profiles_dict)])\
.filter(lambda x: x[0][1] != 0)\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0][0], [(x[0][1], round((x[1] / user_n_items_d.get(x[0][0])), 8) )]))\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], dict(x[1])))\
.persist()

user_profiles_l = user_profiles.collect()
user_profiles_dict = {x[0]:x[1] for x in user_profiles_l}

end = time.time()
t_time = end-start
print('user_profiles time: ', t_time)

#Write the item_profile and user_profile to .json file 
model_full = {}
model_full['item_profiles'] = item_profiles_dict
model_full['user_profiles'] = user_profiles_dict

with open(model_file, 'w') as outfile:
    json.dump(model_full, outfile)


