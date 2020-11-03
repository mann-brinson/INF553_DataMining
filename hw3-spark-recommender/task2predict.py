from pyspark import SparkContext
import json
import time
import math
import sys

#FUNCTIONS
def get_feats(item_user_pair):
	res = []
	for (idx, i) in enumerate(item_user_pair):
		if idx == 0:
			item_feats = model_dict.get('item_profiles').get(i)
			res.append(item_feats)
		else:
			user_feats = model_dict.get('user_profiles').get(i)
			res.append(user_feats)
	return res

def get_cos_sim(item_feats, user_feats):
	'''Get cosine similarity between two profiles'''
	num = 0 #Cosine Sim Numerator
	denom_item = 0 #Cosine Sim Denominator - Item
	denom_user = 0 #Cosine Sim Denominator - User
	for feat in item_feats:
		item_feat_sqr = item_feats.get(feat)**2
		denom_item += item_feat_sqr
		if feat in user_feats:
			dp = item_feats.get(feat) * user_feats.get(feat)
			num += dp

	for feat in user_feats:
		user_feat_sqr = user_feats.get(feat)**2
		denom_user += user_feat_sqr

	denom = math.sqrt(denom_item * denom_user)
	cos_sim = round((num / denom), 8)
	return cos_sim

#PARAMETERS
test_file = sys.argv[1]
model_file = sys.argv[2]
predictions_file = sys.argv[3]

sc = SparkContext(appName="inf553")

#DRIVER
#Load the model file
with open(model_file) as json_file:
	model_dict = json.load(json_file)

#Build the predictions
#Get the valid item-user pairs
start = time.time()
item_user_pairs = sc.textFile(test_file)\
.map(lambda x: json.loads(x))\
.map(lambda x: (x['business_id'], x['user_id']))\
.filter(lambda x: x[0] in model_dict['item_profiles'] and x[1] in model_dict['user_profiles'])\
.persist()

# Get the item and user's feature vector
# Calculate cos_sim for each item-user feature pair
predictions_rdd = item_user_pairs.flatMap(lambda x: [((x[0], x[1]), feat_vec) for feat_vec in get_feats(x)])\
.reduceByKey(lambda x, y: get_cos_sim(x, y))\
.filter(lambda x: x[1] >= 0.01)\
.map(lambda x: (x[0][1], x[0][0], x[1]))

#Write the result out to disk
predictions_l = predictions_rdd.collect()
predictions_final = []
for pred in predictions_l:
	pred_d = {'user_id': pred[0], 'business_id': pred[1], 'sim': pred[2]}
	predictions_final.append(pred_d)

#Write each json document on a new line 
with open(predictions_file, 'w') as outfile:
	for pred in predictions_final:
		json.dump(pred, outfile)
		outfile.write('\n')

end = time.time()
t_time = end-start
print('predict time: ', t_time)



