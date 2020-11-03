from pyspark import SparkContext
import json
import time
import math
import random
import itertools
import sys

#FUNCTIONS
#CASE 1
def make_pairs(item, b_list):
	pairs = []
	for j in b_list:
		if j[0] > item[0]:
			pair = (item[1], j[1])
			pairs.append(pair)
	return pairs

#CASE 2
def hash_b_id(basket_id, h_id, a_list, b_list, m):
	item_hash = (a_list[h_id]*basket_id + b_list[h_id]) % m
	return item_hash

def get_random_hash_vars(n_hashes, max_rowid, rand_seed):
	''' Used to generate lists of unique a and b values for hash functions.
	a_list and b_list of equal length. Length of lists = n_hashes.
	Hash function template: f(x) = (ax + b) % m.'''
	var_dict = {}
	#Get a random row_id and make sure its unique, before adding to var_list
	random.seed(rand_seed) #Same random values generated each time
	for h in range(n_hashes):
		rowid_rand = random.randint(1, max_rowid)
		#If the row is not unique, try again
		while rowid_rand in var_dict:
			rowid_rand = random.randint(1, max_rowid)
		var_dict[rowid_rand] = 1
	rand_row_list = list(var_dict.keys())
	return rand_row_list

def jaccard(cand_pair, ub_set_d):
	'''Calculate jaccard of cand_pair.'''
	set1 = ub_set_d.get(cand_pair[0]).keys()
	set2 = ub_set_d.get(cand_pair[1]).keys()
	intersection = set1 & set2
	union = set1 | set2
	jaccard = len(intersection) / len(union)
	cand_pair_j = (cand_pair, jaccard)
	return cand_pair_j

def band_for_sighash(key, n_hashes, n_bands):
	'''Enables partitioning of sighashes into bands.'''
	n_rows = n_hashes / n_bands
	bands_l = list(range(n_bands))
	bands_l.reverse()
	for (idx, i) in enumerate(bands_l):
		thresh = int(n_hashes - (i * n_rows))
		if key < thresh:
			assigned_band = idx
			break
		else:
			continue
	return assigned_band

def add_band_id(band_index, band): 
	'''Add band id to each cell in sig_matrix. Enables comparison of sig_vecs.'''
	band_list = []
	for i in band:
		band_list.append((band_index, i))
	return band_list

def get_sigvec_str(sigvec_l):
	'''Take in signature vector as list, and return string.'''
	sigvec_str = ''
	for i in sigvec_l:
		char = str(i[1])
		sigvec_str = sigvec_str + char
	return sigvec_str

def order_pair(pair):
	pair_l = list(pair)
	pair_l.sort()
	pair_t = tuple(pair_l)
	return pair_t

#SHARED
def get_avg(bus_ratings):
	count_ratings = 0
	sum_ratings = 0
	for u_rating in bus_ratings:
		count_ratings += 1
		sum_ratings += u_rating[1]
	avg_ratings = sum_ratings / count_ratings
	return avg_ratings

def divide(num, denom):
	return num / denom if denom else 0

def get_wij(b_pair):
	'''
	input: b_pair - looks like (('b1', 'b2'), {'b1': ([('u2', 2), ('u3', 3), ('u4', 5)], r_b1)
	                                          'b2': ([('u3', 4), ('u4', 3), ('u5', 5)], r_b2)})
	output: w_ij - pearson correlation of b1 (i) and b2 (j)
	'''
	ratings_idx = list(range(len(b_pair[1].get(b_pair[0][0])[0])))
	b1_ratings = b_pair[1].get(b_pair[0][0])
	b2_ratings = b_pair[1].get(b_pair[0][1])
	#Calculate numerator for pearson
	num = 0
	for idx in ratings_idx:
		num_temp = (b1_ratings[0][idx][1] - b1_ratings[1]) * (b2_ratings[0][idx][1] - b2_ratings[1])
		num += num_temp
	#Calculate denominator for pearson
	denom_b1 = math.sqrt(sum([(b1_ratings[0][idx][1] - b1_ratings[1])**2 for idx in ratings_idx]))
	denom_b2 = math.sqrt(sum([(b2_ratings[0][idx][1] - b2_ratings[1])**2 for idx in ratings_idx]))
	denom = denom_b1 * denom_b2
	#Combine to form w_ij (pearson)
	w_ij = divide(num, denom)
	return w_ij

#SHARED PARAMETERS
train_file = sys.argv[1]
model_file = sys.argv[2]
cf_type = sys.argv[3]

sc = SparkContext(appName="inf553")

#DRIVER
#CASE 1
if cf_type == 'item_based':
	#CASE 1 PARAMETERS
	min_shared_users = 3

	#Find business pairs with at least 3 co-rated users
	#Get key = b1, value = {'u1': 1, 'u2': 2, ...}. The value must be of len >= min_shared_users.
	bu_set = sc.textFile(train_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['business_id'], [(x['user_id'], x['stars'])]))\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], dict(x[1])))\
	.filter(lambda x: len(x[1]) >= min_shared_users).persist()
	bu_set_d = dict(bu_set.collect())
	b_list = list(enumerate(bu_set_d.keys()))

	#Use flatmap to get all business pairs (~ 52M)
	#For each business pair, check if user intersection >= min_shared_users
	# For valid business pairs, output key = (b1, b2) and value = {u2, u3, u4, ..}

	start = time.time()
	b_pairs_valid = sc.parallelize(b_list)\
	.flatMap(lambda x: [pair for pair in make_pairs(x, b_list)])\
	.map(lambda x: (x, bu_set_d.get(x[0]).keys() & bu_set_d.get(x[1]).keys()))\
	.filter(lambda x: len(x[1]) >= min_shared_users).persist()
	end = time.time()
	t_time = end-start
	print('Business pairs search time: ', t_time)

	#Get w_ij (pearson) of each valid business pair. Filter out w_ijs that are negative or zero.
	start = time.time()
	b_pairs_model = b_pairs_valid.map(lambda b_pair: (b_pair[0], ([(u, bu_set_d.get(b_pair[0][0]).get(u)) for u in b_pair[1]],
	                                                              [(u, bu_set_d.get(b_pair[0][1]).get(u)) for u in b_pair[1]])))\
	.map(lambda x: (x[0], {x[0][0]: (x[1][0], get_avg(x[1][0])),
	                       x[0][1]: (x[1][1], get_avg(x[1][1]))}))\
	.map(lambda x: (x[0], get_wij(x)))\
	.filter(lambda x: x[1] > 0)\
	.map(lambda x: dict([('b1', x[0][0]), ('b2', x[0][1]), ('sim', x[1])])).persist()

	b_pairs_model_l = b_pairs_model.collect()
	end = time.time()
	t_time = end-start
	print('w_ij computing time: ', t_time)

	#Write each json document on a new line 
	with open(model_file, 'w') as outfile:
		for b_pair in b_pairs_model_l:
			json.dump(b_pair, outfile)
			outfile.write('\n')

#CASE 2
elif cf_type == 'user_based':
	#CASE 2 PARAMETERS
	n_hashes = 10
	n_bands = 10
	min_jaccard = 0.01
	min_shared_bus = 5

	#PREPROCESSING
	#Make b_mapping, u_mapping, and feat_vec_l

	#GET USERS (ITEMS)
	#Find user pairs with at least 3 co-rated businesses
	#Get key = u1, value = {'b1': 1, 'b2': 2, ...}. The value must be of len >= min_shared_bus.
	ub_set = sc.textFile(train_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['user_id'], [(x['business_id'], x['stars'])]))\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], dict(x[1])))\
	.filter(lambda x: len(x[1]) >= min_shared_bus).persist()
	ub_set_d = dict(ub_set.collect())
	n_items = len(ub_set.collect())

	u_bus = ub_set.flatMap(lambda x: [(x[0], bus) for bus in x[1].keys()]).persist()

	#GET BUSINESSES (BASKETS)
	business = u_bus.map(lambda x: (x[1], 1))\
	.reduceByKey(lambda x, y: 1)\
	.map(lambda x: x[0]).persist()
	m = len(business.collect())

	#Build feature vector
	feat_vec_raw = u_bus.map(lambda x: (x[1], [x[0]]))\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], set(x[1]))).persist()

	feat_vec_l = feat_vec_raw.collect()
	feat_vec_l.sort()
	feat_vec_l_idx = [(x[0], x[1][1]) for x in list(enumerate(feat_vec_l))]
	feat_vec_rdd = sc.parallelize(feat_vec_l_idx)

	#Build signature matrix
	start = time.time()
	max_rowid = feat_vec_l_idx[-1][0]
	a_list = get_random_hash_vars(n_hashes, max_rowid, 7)
	b_list = get_random_hash_vars(n_hashes, max_rowid, 42)

	sig_matrix_rdd = feat_vec_rdd.flatMap(lambda b: [(b, h) for h in range(0, n_hashes)])\
	.flatMap(lambda b_hash: [((b_hash[1], item), hash_b_id(b_hash[0][0], b_hash[1], a_list, b_list, m)) for item in b_hash[0][1]])\
	.reduceByKey(min).persist()

	sig_matrix_l = sig_matrix_rdd.sortByKey(True).collect()
	end = time.time()
	t_time = end-start
	print('Build sig_matrix time: ', t_time)

	#LSH: Given the signature matrix rdd, cut into b bands output candidate item pairs.
	#NOTE: Because h = b x r, you must pick b and r that exactly produce h. 
	# Ex: h = 10, b = 5, r = 2.
	start = time.time()
	cand_pairs_rdd = sc.parallelize(sig_matrix_l)\
	.map(lambda x: (x[0][0], x))\
	.partitionBy(n_bands, lambda x: band_for_sighash(x, n_hashes, n_bands))\
	.map(lambda x: x[1])\
	.mapPartitionsWithIndex(add_band_id)\
	.map(lambda x: ((x[0], x[1][0][1]),[(x[1][0][0], x[1][1])]))\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], get_sigvec_str(x[1])))\
	.map(lambda x: ((x[0][0], x[1]), [x[0][1]]))\
	.reduceByKey(lambda x, y: x + y)\
	.filter(lambda x: len(x[1]) >= 2)\
	.flatMap(lambda x: list(itertools.combinations(x[1], 2)))\
	.map(lambda x: order_pair(x))\
	.map(lambda x: (x, 1))\
	.reduceByKey(lambda x, y: 1)\
	.map(lambda x: x[0]).persist()
	# cand_pairs_l = cand_pairs_rdd.collect()
	end = time.time()
	t_time = end - start
	print('Find pairs time: ', t_time)

	#Calculate Jaccard
	start = time.time()
	sim_pairs_j_rdd = cand_pairs_rdd.map(lambda x: jaccard(x, ub_set_d))\
	.filter(lambda x: x[1] >= min_jaccard)
	sim_pairs_j = sim_pairs_j_rdd.collect()
	end = time.time()
	t_time = end-start
	print('get jaccard time: ', t_time)

	#Filter out pairs that don't have at least 3/5/7 co-rated businesses
	u_pairs_valid = sim_pairs_j_rdd.map(lambda x: (x[0], ub_set_d.get(x[0][0]).keys() & ub_set_d.get(x[0][1]).keys()))\
	.filter(lambda x: len(x[1]) >= min_shared_bus).persist()
	u_pairs_valid_l = u_pairs_valid.collect()

	#Get w_uv (pearson) of each valid user pair. Filter out w_uvs that are negative or zero.
	start = time.time()
	u_pairs_model = u_pairs_valid.map(lambda u_pair: (u_pair[0], ([(b, ub_set_d.get(u_pair[0][0]).get(b)) for b in u_pair[1]],
	                                                              [(b, ub_set_d.get(u_pair[0][1]).get(b)) for b in u_pair[1]])))\
	.map(lambda x: (x[0], {x[0][0]: (x[1][0], get_avg(x[1][0])),
	                       x[0][1]: (x[1][1], get_avg(x[1][1]))}))\
	.map(lambda x: (x[0], get_wij(x)))\
	.filter(lambda x: x[1] > 0)\
	.map(lambda x: dict([('u1', x[0][0]), ('u2', x[0][1]), ('sim', x[1])])).persist()

	u_pairs_model_l = u_pairs_model.collect()
	end = time.time()
	t_time = end-start
	print('w_uv computing time: ', t_time)

	#Write each json document on a new line 
	with open(model_file, 'w') as outfile:
		for u_pair in u_pairs_model_l:
			json.dump(u_pair, outfile)
			outfile.write('\n')