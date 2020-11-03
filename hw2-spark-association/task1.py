from pyspark import SparkContext
import sys
import json
from operator import add
import time
import itertools
import os

def construct_data(filename, case_num):
	user_bus = sc.textFile(filename)\
	.map(lambda x: tuple(x.split(',')))\
	.filter(lambda x: x[0] != 'user_id')\
	.map(lambda x: (x[0] + '-' + x[1], 1))\
	.reduceByKey(lambda x, y: 1)\
	.map(lambda x: tuple(x[0].split('-')))\
	.map(lambda x: (int(x[1]), [int(x[0]), 1]) if case_num == '1' else (int(x[0]), [int(x[1]), 1]))\
	.persist()
	# user_bus.collect() #op: (b1, (u1, 1)), (b1, (u2, 1)), ...

	#Produce (u1, [b1, b2, ...]), (u2, [b2, b4]), ...
	user_busset = user_bus.map(lambda x: (x[1][0], [str(x[0])]))\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], set(x[1])))\
	.persist()
	return user_bus, user_busset

def hash(tb_hashed, num_buckets):
	bucket = tb_hashed % num_buckets
	return(bucket)

def create_1item_bm(k, user_bus, cand_bitmap_full):
	'''Create candidate 1-itemsets bitmap.
	k - should be equal to 1
	ip/op: cand_bitmap_full - adds to this'''
	cand_bitmap = user_bus.map(lambda x: x[0])\
	.map(lambda x: hash(x, num_buckets))\
	.map(lambda x: (x, 1))\
	.reduceByKey(add)\
	.map(lambda x: ((x[0], 1) if x[1] >= support else (x[0], 0)))
	cand_bitmap_full[k] = dict(cand_bitmap.collect())

def create_1item_candidates(k, user_bus, cand_bitmap_full, cand_itemsets_full):
	'''Creates candidate 1-itemset candidates. Include those that hash to frequent buckets only.
	ip/op: cand_itemsets_full - adds to this'''

	#Count each candidate itemset, iff i) 1-itemset hashes to frequent bucket
	cand_itemset_rdd = user_bus.map(lambda x: (x[0], hash(x[0], num_buckets)))\
	.map(lambda x: (str(x[0]), 1) if cand_bitmap_full.get(k).get(x[1]) == 1 else None)\
	.filter(lambda x: x != None)\
	.reduceByKey(add)
	cand_itemset_d = dict(cand_itemset_rdd.collect())
	cand_itemsets_full[k] = cand_itemset_d
	return cand_itemset_rdd

def create_1item_frequents(cand_itemset_rdd, freq_itemsets_full):
	'''Derives frequent 1-itemsets from candidate 1-itemsets.
	ip: cand_itemset - op from create_1item_candidates
	ip/op: freq_itemsets_full - adds to this'''

	#From candidates, find frequent 1-itemsets with enough support
	freq_itemsets = cand_itemset_rdd.map(lambda x: (str(x[0]), x[1]) if x[1] >= support else None)\
	.filter(lambda x: x != None)
	freq_itemsets_full[k] = dict(freq_itemsets.collect())

def make_candidates(prior_freqs, k):
	'''Combine prior frequents into sets, to next-level candidates.
	prior_freqs - such as [('101', '99'), ('101', '102'), ('101', '103')...]
	k - size of desired candidate itemsets
	op - such as [('101', '102', '103'), ('101', '102', '99'),...] '''
	if k == 2:
		itemsets = list(itertools.combinations(prior_freqs, 2))
		cand_itemsets = []
		for i in itemsets:
			j = list(i)
			j.sort()
			cand_itemsets.append(tuple(j))
		cand_itemsets.sort()
		return cand_itemsets
	elif k >= 3:
		itemsets = list(itertools.combinations(prior_freqs, 2))
		cand_itemsets_d = {}
		for i in itemsets:
			j = list(set(i[0] + i[1]))
			j.sort()
			if len(j) == k:
				cand_itemsets_d[tuple(j)] = 1
		cand_itemsets = list(cand_itemsets_d.keys())
		cand_itemsets.sort()
		return cand_itemsets

def check_basket(basket, poss_cand_itemsets):
	'''Checks basket for poss_cands. Returns list of true_cands.
	Each returned cand can be hashed to a bucket.'''
	basket_cands = []
	for ci in poss_cand_itemsets:
		ci_s = set(ci)
		if ci_s.issubset(basket):
			ci_l = list(ci_s)
			ci_l.sort()
			ci_t = (tuple(ci_l), 1)
			basket_cands.append(ci_t)
	return basket_cands

def tb_hashed_int(itemset):
	'''Takes an itemset, with int elements and returns its tb_hashed. 
	Output should be fed to hash function'''
	res = 0
	for i in itemset:
		res += int(i) 
	return res

def itemset_sort(k, itemset_container):
	'''Sorts candidate and frequent itemset lists in lexographic order'''
	ci = list(itemset_container.get(k).keys())
	ci.sort()

	# 1-ITEMSET
	if type(ci[0]) == str:
		ci_s = str(ci).replace(', ', '),(').replace('[', '(').replace(']', ')')
		return ci_s

	#2+ ITEMSET
	elif type(ci[0]) == tuple:
		ci_s = str(ci).replace(', (', ',(').replace('[', '').replace(']', '')
		return ci_s

#Pull out arguments from command line
case_number = sys.argv[1]
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

sc = SparkContext(appName="inf553")

#DRIVER
num_buckets = 20
max_k = 100

cand_itemsets_full = {} #Master
cand_bitmap_full = {} #Master
freq_itemsets_full = {} #Master

start = time.time()
user_bus, user_busset = construct_data(input_file_path, case_number)

for k in range(1, max_k):
	#1-ITEMSET
	if k == 1:
		print('k: ', k)
		create_1item_bm(k, user_bus, cand_bitmap_full)
		cand_itemset_rdd = create_1item_candidates(k, user_bus, cand_bitmap_full, cand_itemsets_full)
		create_1item_frequents(cand_itemset_rdd, freq_itemsets_full)

	if k > 1 and k < max_k:
		#Make candidate itemsets of size k. Return candidates rdd
		print('k: ', k)

		freq_itemsets = list(freq_itemsets_full.get(k-1).keys())
		print('freq_itemsets_len: ', len(freq_itemsets))
		# print('freq_itemsets: ', freq_itemsets)
		poss_cand_itemsets = make_candidates(freq_itemsets, k)

		#If no candidates found
		if poss_cand_itemsets == []:
			#STOP and write output to file
			print('no more poss_cand_itemsets')
			# print('cand_bitmap_full: ', cand_bitmap_full)
			# print('cand_itemsets_full: ', cand_itemsets_full)
			# print('freq_itemsets_full: ', freq_itemsets_full)
			end = time.time()
			t_time = 'Duration: ' + str(end-start)
			print(t_time)

			with open(output_file_path, 'w') as fd:
				fd.write(t_time + os.linesep) #Duration
				ci_keys = list(cand_itemsets_full.keys()) #Candidates
				fd.write('Candidates:' + os.linesep)
				for key in ci_keys[:-1]:
					ci_s = itemset_sort(key, cand_itemsets_full)
					fd.write(ci_s + os.linesep + os.linesep)

				ci_keys = list(freq_itemsets_full.keys()) #Frequent Itemsets
				fd.write('Frequent Itemsets:' + os.linesep)
				for key in ci_keys:
					ci_s = itemset_sort(key, freq_itemsets_full)
					fd.write(ci_s + os.linesep + os.linesep)
			break

		#Elseif candidates found
		elif poss_cand_itemsets != []:
			print('poss_cand_itemsets_len: ', len(poss_cand_itemsets))

			#Make a pass through baskets, scanning for candidates.
			poss_cands = user_busset.flatMap(lambda x: check_basket(x[1], poss_cand_itemsets)).persist()
			poss_cands_found = poss_cands.glom().map(len).collect()
			# print('poss_cands_found: ', poss_cands_found)

			#For each poss_candidate_itemset, hash it to bucket. 
			#Create bucket bitmap after pass is complete
			poss_cands_hash = poss_cands.map(lambda x: (hash(tb_hashed_int(x[0]), num_buckets), 1))\
			.reduceByKey(lambda x, y: x + y)\
			.map(lambda x: (x[0], 1) if x[1] >= support else (x[0], 0))
			cand_hash_bm = dict(poss_cands_hash.collect())
			cand_bitmap_full[k] = cand_hash_bm
			# print('cand_bitmap_full: ', cand_bitmap_full)

			#For each poss_candidate_itemset, filter out those whose hash is not frequent
			#Filter those whose count is < support
			most_likely_cands = poss_cands.filter(lambda x: cand_bitmap_full.get(k).get(hash(tb_hashed_int(x[0]), num_buckets)) == 1)\
			.reduceByKey(lambda x, y: x + y).persist()
			cand_itemsets_full[k] = dict(most_likely_cands.collect())

			#Count the most_likely_candidates. Frequent itemsets have candidate counts >= support.
			freq_itemsets = most_likely_cands.filter(lambda x: x[1] >= support)
			freq_itemsets_full[k] = dict(freq_itemsets.collect())
