from pyspark import SparkContext
import sys
import json
from operator import add

sc = SparkContext(appName="inf553")

#Pull out arguments from command line
input_file = sys.argv[1]
output_file = sys.argv[2]
partition_type = sys.argv[3]
n_partitions = int(sys.argv[4])
n = int(sys.argv[5])

if partition_type == 'default':
  r = sc.textFile(input_file)\
  .map(lambda x: json.loads(x))\
  .map(lambda x: (x['business_id'], 1))

elif partition_type == 'customized':
  r = sc.textFile(input_file)\
  .map(lambda x: json.loads(x))\
  .map(lambda x: (x['business_id'], 1))\
  .partitionBy(n_partitions)

#Get num_partitions and num_items
num_partitions = r.getNumPartitions()
num_items = r.glom().map(len).collect()

#Count and filter
r_final = r.reduceByKey(add)\
  .filter(lambda x: x[1] > n) #filter out businesses with less than n reviews
res = r_final.collect() #op: (b_id, num_reviews)
res_list = [list(i) for i in res]

result = {"n_partitions": num_partitions, "n_items": num_items, 
          "result": res_list}
with open(output_file, 'w') as outfile:
	json.dump(result, outfile)