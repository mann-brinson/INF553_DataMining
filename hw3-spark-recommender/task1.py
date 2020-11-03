from pyspark import SparkContext
import json
import time
import itertools
import sys
from unicodedata import *
import random

#FUNCTIONS
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

def jaccard(cand_pair, bu_set_d):
  '''Calculate jaccard of cand_pair.'''
  set1 = bu_set_d.get(cand_pair[0])
  set2 = bu_set_d.get(cand_pair[1])
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
    # print(f'idx: {idx}, i: {i}')
    thresh = int(n_hashes - (i * n_rows))
    # print(f'thresh {thresh}')
    if key < thresh:
      # print(f'key is {key}. Assigned band is {idx}')
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

#PARAMETERS
train_file = sys.argv[1]
out_file = sys.argv[2]
n_hashes = 30
n_bands = 30
min_jaccard = 0.05

sc = SparkContext(appName="inf553")

#DRIVER
#PREPROCESSING
#Make b_mapping, u_mapping, and feat_vec_l_idx
b_user = sc.textFile(train_file)\
.map(lambda x: json.loads(x))\
.map(lambda x: (x['business_id'], x['user_id']))\
.map(lambda x: (x[0] + x[1], 1))\
.reduceByKey(lambda x, y: 1)\
.map(lambda x: (x[0][:22], x[0][22:])).persist()
# sum(b_user.glom().map(len).collect()) Total user-bus combinations. Length = 488,560

#GET BUSINESSES
bu_set = b_user.map(lambda x: (x[0], [x[1]]))\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], set(x[1]))).persist()
bu_set_d = dict(bu_set.collect())
n_items = len(bu_set.collect())

#GET USERS
user = b_user.map(lambda x: (x[1], 1))\
.reduceByKey(lambda x, y: 1)\
.map(lambda x: x[0]).persist()
m = len(user.collect())

#Build feature vector. Use mapped user_id and business_id, instead of actual ids.
feat_vec_raw = b_user.map(lambda x: (x[1], [x[0]]))\
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
cand_pairs_l = cand_pairs_rdd.collect()
end = time.time()
t_time = end - start
print('Find pairs time: ', t_time)

#Calculate Jaccard
start = time.time()
sim_pairs_j_rdd = cand_pairs_rdd.map(lambda x: jaccard(x, bu_set_d))\
.filter(lambda x: x[1] >= min_jaccard)\
.map(lambda x: {'b1': x[0][0], 'b2': x[0][1], 'sim': x[1]})
sim_pairs_j = sim_pairs_j_rdd.collect()
end = time.time()
t_time = end-start
print('get jaccard time: ', t_time)

#Write the business pairs to outfile
with open(out_file, 'w') as outfile:
  for sim_pair in sim_pairs_j:
    json.dump(sim_pair, outfile)
    outfile.write('\n')

