from pyspark import SparkContext
import json
import time
import math
import sys

#FUNCTIONS

#CASE 2
def union_sets(x, y):
	res = x | y
	return res

def p_ai_secondterm(ub_sim_users):
	num = sum([((x[1]-x[3])*x[2]) for x in ub_sim_users[1]])
	denom = sum([abs(x[2]) for x in ub_sim_users[1]])
	secondterm = divide(num, denom)
	return secondterm

def max_round(x):
	if x > 5:
		return 5
	else:
		return x

#SHARED
def sort_lot(lot):
	res = lot
	res.sort(key=lambda x:x[2], reverse=True)
	return res

def divide(num, denom):
	return num / denom if denom else 0

#PARAMETERS
train_file = sys.argv[1]
test_file = sys.argv[2]
model_file = sys.argv[3]
prediction_file = sys.argv[4]
cf_type = sys.argv[5]

sc = SparkContext(appName="inf553")

#DRIVER
#CASE 1
if cf_type == 'item_based':
	#CASE 1 PARAMETERS
	n_neighbor_items = 5

	#Read the train data and contruct ub_set and b_uset
	#train data -> ub_set, b_uset
	bu_distinct = sc.textFile(train_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['business_id'] + x['user_id'] , x['stars']))\
	.reduceByKey(lambda x, y: y)\
	.map(lambda x: (x[0][:22], [(x[0][22:], x[1])])).persist()

	bu_set = bu_distinct.reduceByKey(lambda x, y: x+y)\
	.map(lambda x: (x[0], dict(x[1]))).persist()
	bu_set_d = dict(bu_set.collect())

	ub_set = bu_distinct.map(lambda x: (x[1][0][0], [(x[0], x[1][0][1])]))\
	.reduceByKey(lambda x, y: x+y)\
	.map(lambda x: (x[0], dict(x[1]))).persist()
	ub_set_d = dict(ub_set.collect())

	#Read the model data and contruct model 
	# model data -> model_pairs
	model_pairs_base = sc.textFile(model_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['b1'], [(x['b2'], x['sim'])])).persist()

	model_pairs_a = model_pairs_base.reduceByKey(lambda x, y: x + y).persist()
	model_pairs_b = model_pairs_base.map(lambda x: (x[1][0][0], [(x[0], x[1][0][1])] ))\
	.reduceByKey(lambda x, y: x + y).persist()

	model_pairs_full = sc.union([model_pairs_a, model_pairs_b])\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], dict(x[1])))
	model_pairs_full_d = dict(model_pairs_full.collect())

	#For each test u-b pair
	#Check if u in ub_set AND b in bu_set
	start = time.time()
	ub_pairs = sc.textFile(test_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['user_id'], x['business_id']))\
	.filter(lambda x: x[0] in ub_set_d)\
	.filter(lambda x: x[1] in bu_set_d).persist()

	#u_items - Get all other items that user u rated from ub_set
	#i_sim_items - Get all similar model pairs for item i from model_pairs_d
	#items_intersection - Find intersection between u_items and i_sim_items
	#Sort the intersection by w_in descending
	#Select top N ui_pair neighbors (3 or 5)
	ub_sim_items = ub_pairs.map(lambda ub_pair: (ub_pair, [ub_set_d.get(ub_pair[0]), 
	                                                       model_pairs_full_d.get(ub_pair[1])]))\
	.filter(lambda x: x[1][1] != None)\
	.map(lambda x: (x[0], [x[1][0], x[1][1], x[1][0].keys() & x[1][1].keys()]))\
	.filter(lambda x: len(x[1][2]) > 0)\
	.map(lambda x: (x[0], [(item, x[1][0].get(item), x[1][1].get(item)) for item in x[1][2]]))\
	.map(lambda x: (x[0], sort_lot(x[1])))\
	.map(lambda x: (x[0], x[1][:n_neighbor_items])).persist()
	# ub_sim_items_l = ub_sim_items.collect() #op: (('u1', 'b1'), [('b3', 3, 1), ('b5', 5, 0.5)]) #WORKING

	#Compute the predcition p_ui
	ub_preds = ub_sim_items.map(lambda x: (x[0], (sum([(i[1] * i[2]) for i in x[1]]), 
	                                              sum([i[2] for i in x[1]]))))\
	.map(lambda x: (x[0], divide(x[1][0], x[1][1])))\
	.map(lambda x: {'user_id': x[0][0], 'business_id': x[0][1], 'stars': x[1]})
	ub_preds_l = ub_preds.collect()

	end = time.time()
	t_time = end-start
	print('Predict rating time: ', t_time)

	#Write each json document on a new line 
	with open(prediction_file, 'w') as outfile:
		for ub_pred in ub_preds_l:
			json.dump(ub_pred, outfile)
			outfile.write('\n')

#CASE 2
elif cf_type == 'user_based':
	#CASE 2 PARAMETERS
	n_neighbor_users = 3

	#Read the train data and contruct ub_set and b_uset
	#train data -> ub_set, b_uset
	bu_distinct = sc.textFile(train_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['business_id'] + x['user_id'] , x['stars']))\
	.reduceByKey(lambda x, y: y)\
	.map(lambda x: (x[0][:22], [(x[0][22:], x[1])])).persist()

	bu_set = bu_distinct.reduceByKey(lambda x, y: x+y)\
	.map(lambda x: (x[0], dict(x[1]))).persist()
	bu_set_d = dict(bu_set.collect())

	ub_set = bu_distinct.map(lambda x: (x[1][0][0], [(x[0], x[1][0][1])]))\
	.reduceByKey(lambda x, y: x+y)\
	.map(lambda x: (x[0], dict(x[1]))).persist()
	ub_set_d = dict(ub_set.collect())

	#Read the model data and contruct model 
	# model data -> model_pairs
	model_pairs_base = sc.textFile(model_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['u1'], [(x['u2'], x['sim'])])).persist()

	model_pairs_a = model_pairs_base.reduceByKey(lambda x, y: x + y).persist()
	model_pairs_b = model_pairs_base.map(lambda x: (x[1][0][0], [(x[0], x[1][0][1])] ))\
	.reduceByKey(lambda x, y: x + y).persist()

	model_pairs_full = sc.union([model_pairs_a, model_pairs_b])\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], dict(x[1])))
	model_pairs_full_d = dict(model_pairs_full.collect())

	#For each test u-b pair
	#Check if u in ub_set AND b in bu_set
	start = time.time()
	ub_pairs = sc.textFile(test_file)\
	.map(lambda x: json.loads(x))\
	.map(lambda x: (x['user_id'], x['business_id']))\
	.filter(lambda x: x[0] in ub_set_d)\
	.filter(lambda x: x[1] in bu_set_d).persist()
	ub_pairs_l = ub_pairs.collect() #WORKING

	#GOAL: Calculate user-based prediction score
	#u_sim_users - for active user a, get all similar users from model_pairs_full_d
	#filter out u-b pairs we can't find similar users for
	#i_users - for active item i, get all other users that rated the item from bu_set_d
	#users_intersection - Find intersection of u_sim_users and i_users
	#select top N users (3 or 5)

	#corated_items - get all items rated by user a, and for users in users_intersection
	#get average rating of corated items

	#Finds intersect of u_sim_users and i_users. Selects N neighbor users
	#Output key is user-business pair = (a, i), and value is list of similar users = [('u2', r_ui, w_uv), ...]
	ub_sim_users = ub_pairs.map(lambda ub_pair: (ub_pair, [bu_set_d.get(ub_pair[1]), model_pairs_full_d.get(ub_pair[0])]))\
	.filter(lambda x: x[1][1] != None)\
	.map(lambda x: (x[0], [x[1][0], x[1][1], x[1][0].keys() & x[1][1].keys()]))\
	.map(lambda x: (x[0], [(user, x[1][0].get(user), x[1][1].get(user)) for user in x[1][2]]))\
	.map(lambda x: (x[0], sort_lot(x[1])))\
	.map(lambda x: (x[0], x[1][:n_neighbor_users])).persist()

	#For each u-b pair's similar users, find co-rated items with active user a
	#Output key is u-b pair = (a, i), and 
	#Output value is a similar user and their co-rated items = [('u2', r_ui, w_uv, {})]
	ub_corated_items = ub_sim_users.flatMap(lambda x: [(x[0], (sim_user[0], sim_user[1], sim_user[2], 
	                                               ub_set_d.get(x[0][0]), ub_set_d.get(sim_user[0]))) for sim_user in x[1]])\
	.map(lambda x: (x[0], [(x[1][0], x[1][1], x[1][2],
	                       x[1][3].keys() & x[1][4].keys())])).persist()

	#For each u-b pair, compute its p_ai firstterm. Save result to dictionary. 
	ub_pai_firstterm = ub_corated_items.map(lambda x: (x[0], x[1][0][3]))\
	.reduceByKey(lambda x, y: union_sets(x, y))\
	.map(lambda x: (x[0], [ub_set_d.get(x[0][0]).get(i) for i in x[1]]))\
	.map(lambda x: (x[0], sum(x[1]), len(x[1])))\
	.map(lambda x: (x[0], x[1]/x[2]))
	ub_pai_firstterm_d = dict(ub_pai_firstterm.collect())

	#For each u-b pair, compute its p_ai secondterm.  
	ub_pai_secondterm = ub_corated_items.map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], [i for i in x[1][0][3]]) ) )\
	.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], [ub_set_d.get(x[1][0]).get(i) for i in x[1][3]])))\
	.map(lambda x: (x[0], (x[1][0], x[1][1], x[1][2], sum(x[1][3]), len(x[1][3]))))\
	.map(lambda x: (x[0], [(x[1][0], x[1][1], x[1][2], x[1][3]/x[1][4])]))\
	.reduceByKey(lambda x, y: x + y)\
	.map(lambda x: (x[0], p_ai_secondterm(x)))

	# Combine with first term, and output the result
	ub_pai_full = ub_pai_secondterm.map(lambda x: (x[0], x[1], ub_pai_firstterm_d.get(x[0])))\
	.map(lambda x: (x[0], sum([x[1], x[2]])))\
	.map(lambda x: (x[0], max_round(x[1])))\
	.map(lambda x: {'user_id': x[0][0], 'business_id': x[0][1], 'stars': x[1]})
	ub_preds_l = ub_pai_full.collect()

	#Write each json document on a new line 
	with open(prediction_file, 'w') as outfile:
		for ub_pred in ub_preds_l:
			json.dump(ub_pred, outfile)
			outfile.write('\n')