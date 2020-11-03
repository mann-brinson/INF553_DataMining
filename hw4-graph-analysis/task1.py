from pyspark import SparkConf,SparkContext,SQLContext
import time
import os
import sys
from graphframes import *

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11  pyspark-shell")

#FUNCTIONS
def make_pairs(item, u_list):
	pairs = []
	for u in u_list:
		if u[0] > item[0]:
			pair = (item[1], u[1])
			pairs.append(pair)
	return pairs

def order_pair(pair):
	pair_l = list(pair)
	pair_l.sort()
	pair_t = tuple(pair_l)
	return pair_t

#PARAMETERS
filter_thresh = int(sys.argv[1])
input_file_path = sys.argv[2]
community_output_file_path = sys.argv[3]

sc = SparkContext(appName="inf553")
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

conf = SparkConf().setMaster("local")\
.setAppName("task0")\
.set("spark.executor.memory", "4g")\
.set("spark.driver.memory", "4g")

#DRIVER
#Load the u-b data and create reference objects ub_set_d and u_list
ub_set = sc.textFile(input_file_path)\
.map(lambda x: tuple(x.split(',')))\
.filter(lambda x: x[0] != 'user_id')\
.map(lambda x: (x[0], [x[1]]) )\
.reduceByKey(lambda x, y: x + y)\
.map(lambda x: (x[0], set(x[1])))\
.filter(lambda x: len(x[1]) >= filter_thresh)\
.persist()

ub_set_d = dict(ub_set.collect())
u_list = list(enumerate(ub_set_d.keys()))

#Use flatmap to get all user pairs (~ 5.6M)
#For each user pair, check if business intersection >= thresh
# For valid user pairs, output key = u1 and value = u2
start = time.time()
u_pairs_edges_a = sc.parallelize(u_list)\
.flatMap(lambda x: [pair for pair in make_pairs(x, u_list)])\
.map(lambda x: (x, ub_set_d.get(x[0]) & ub_set_d.get(x[1])))\
.filter(lambda x: len(x[1]) >= filter_thresh)\
.map(lambda x: (x[0])).persist()
u_pairs_edges_b = u_pairs_edges_a.map(lambda x: (x[1], x[0])).persist()
u_pairs_edges_full = sc.union([u_pairs_edges_a, u_pairs_edges_b])

#EDGES
u_pairs_edges_l = u_pairs_edges_full.collect()
end = time.time()
t_time = end-start
print('User pairs edges search time: ', t_time)
print('u_pairs_valid_l length: ', len(u_pairs_edges_l))

#NODES 
u_nodes_a = u_pairs_edges_a.reduceByKey(lambda x, y: 1)\
.map(lambda x: (x[0], 1))\
.persist()

u_nodes_b = u_pairs_edges_b.reduceByKey(lambda x, y: 1)\
.map(lambda x: (x[0], 1))\
.persist()

u_nodes_full = sc.union([u_nodes_a, u_nodes_b])\
.reduceByKey(lambda x, y: 1)\
.map(lambda x: (x[0],))\
.persist()

u_nodes_full_l = u_nodes_full.collect()

#Create a graphframe with the nodes and edges data
user_edges = sqlContext.createDataFrame(u_pairs_edges_l, ["src", "dst"])
user_nodes = sqlContext.createDataFrame(u_nodes_full_l, ["id"])
g = GraphFrame(user_nodes, user_edges)

#Run the Label Propagation algorithm to detect communities
start = time.time()
result_l = g.labelPropagation(maxIter=5).coalesce(1).collect()
end = time.time()
t_time = end-start
print('community detection time: ', t_time)

#Group the nodes by their labels to form communities
node_labels = []
for node in result_l:
	node_label = tuple(node)
	node_labels.append(node_label)

node_comms = sc.parallelize(node_labels)\
.map(lambda x: (x[1], [x[0]]))\
.reduceByKey(lambda x,y : x + y)\
.map(lambda x: x[1]).persist()
node_comms_l = node_comms.collect()

#Sort the communities in the required order
[comm.sort() for comm in node_comms_l]
node_comms_l_s = sorted(node_comms_l, key=lambda x: (len(x), x[0]))

#Write the community nodes to a file
with open(community_output_file_path, 'w') as outfile:
	for line in node_comms_l_s:
		line.sort()
		line_str = str(line).replace('[', '').replace(']', '')
		outfile.write(line_str)
		outfile.write('\n')
print('finished')

