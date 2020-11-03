import sys
import json
from operator import add

#Pull out arguments from command line
review_file = sys.argv[1]
business_file = sys.argv[2]
output_file = sys.argv[3]
if_spark = sys.argv[4]
n = int(sys.argv[5])

def spark(review_file, business_file, output_file, n):
	from pyspark import SparkContext
	sc = SparkContext(appName="inf553")

	#REVIEW
	r_lines = sc.textFile(review_file)\
	  .map(lambda x: json.loads(x))\
	  .map(lambda x: (x['business_id'], x['stars'])) 
	#op: (b_id, stars)

	agg = r_lines.aggregateByKey((0,0), 
	                             lambda acc ,v: (acc[0] + v, acc[1] + 1), 
	                             lambda acc1,acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1])) 
	#op: (b_id, (stars_sum, stars_count))

	r_res = agg.map(lambda x: (x[0], round((float(x[1][0])/x[1][1]), 3) ))
	#op: (b_id, avg_stars)

	#BUSINESS
	b_lines = sc.textFile(business_file)\
	  .map(lambda x: json.loads(x))\
	  .map(lambda x: (x['business_id'], x['categories']))\
	  .filter(lambda x: x[1] != None)\
	  .map(lambda x: (x[0], x[1].split(', ')))
	#op: (b_id, [cat1, cat2, ...])

	b_r = b_lines.join(r_res)
	bus_fm = b_r.flatMap(lambda x: [(val, x[1][1]) for val in x[1][0]])
	#op: [(active, stars_b1), (golf, stars_b1), (golf, stars_b2), ...]

	b_agg = bus_fm.aggregateByKey((0,0), 
	                                lambda acc ,v: (acc[0] + v, acc[1] + 1), 
	                                lambda acc1,acc2: (acc1[0] + acc2[0], acc1[1] + acc2[1]))
	#op: (category, (stars_sum, stars_count))

	b_res = b_agg.map(lambda x: (x[0], round((float(x[1][0])/x[1][1]), 3) ))
	#op: (category, avg_stars)

	b_sort = b_res.sortBy(lambda x: (-x[1], x[0]))\
	  .take(n)
	b_list = [list(i) for i in b_sort] #op: top n tuples ranked by avg_stars, then category
	result = {"result": b_list}
	with open(output_file, 'w') as outfile:
		json.dump(result, outfile)

def inv_index(review_file):
	'''Creates inverted index on review.json using business_id as key
	review_file: an input .json file with one json object per review
	inv_index key: business_id
	inv_index vals: list of {review_id : stars}
	'''
	review_list = []
	inv_index = {}
	with open(review_file) as json_file:
		for json_obj in json_file:
			review_dict = json.loads(json_obj)
			b_id = review_dict['business_id']
			if b_id not in inv_index:
				inv_index[b_id] = [(review_dict['review_id'], review_dict['stars'])]
			else:
				inv_index[b_id].append((review_dict['review_id'], review_dict['stars']))
		return inv_index

def category_avg(business_file, review_inv_index, n, output_file):
	'''For each business in business.json, get the average stars. Then for each
	business"s category, add and update the average. Keep re-calculating each category star average
	for each busienss. 
	business_file: business.json containing one json object per business
	review_inv_index: inv_index created in first function
	n: the top n categories with the highest star'''
	category = {}
	with open(business_file) as json_file:
		for json_obj in json_file:
			business_obj = json.loads(json_obj)
			b_id = business_obj['business_id']

			#Get categories
			categories = business_obj['categories']
			if categories == None:
				continue
			else:
				cat_list = business_obj['categories'].split(', ')

			#Get business review count and sum
			if b_id in review_inv_index: #Reviews found
				review_stars = review_inv_index[b_id]
				review_count = len(review_stars) #Get review_count

				stars = [review[1] for review in review_stars] #Get review_sum
				review_sum = sum(stars)

				review_avg = review_sum / review_count #Get review_avg

			else: #No reviews found
				review_avg = None

			#Accumulate each business count and sum by category
			for cat in cat_list:
				if cat not in category: #net-new category
					new_avg = review_avg
					category[cat] = review_avg

				else:
					prev_avg = category[cat]
					if prev_avg == None and review_avg == None:
						category[cat] = review_avg
					elif prev_avg == None and review_avg != None:
						category[cat] = review_avg
					elif prev_avg != None and review_avg == None:
						pass
					elif prev_avg != None and review_avg != None:
						new_avg = (prev_avg + review_avg) / 2
						category[cat] = new_avg
			            
	#Select top n categories
	cat_tups = list(category.items())
	cat_valid = [tup for tup in cat_tups if tup[1] != None]
	cat_valid.sort(key=lambda x: (-x[1], x[0]))
	cat_valid = cat_valid[:n]

	#Formatting requirement
	res = [list(item) for item in cat_valid]
	res_dict = {"result": res}

	with open(output_file, 'w') as outfile:
		json.dump(res_dict, outfile)

def no_spark(review_file, business_file, output_file, n):
	review_inv_index = inv_index(review_file)
	category_avg(business_file, review_inv_index, n, output_file)

if __name__ == "__main__":
    if if_spark == 'spark':
    	spark(review_file, business_file, output_file, n)
    elif if_spark == 'no_spark':
    	no_spark(review_file, business_file, output_file, n)

