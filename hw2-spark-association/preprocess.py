from pyspark import SparkContext
import sys
import csv

business_file = sys.argv[1]
review_file = sys.argv[2]
output_file_path = sys.argv[3]

def write_csv(outfile, header, content):
	'''Writes a list of tuples to csv.
	outfile - filepath to google drive
	content - list of tuples such as [('u1', 'b1'), ('u1', 'b2')] '''
	with open(outfile, 'w') as fd:
		write = csv.writer(fd)
		write.writerow(header)
		for row in content:
			write.writerow(row)

#Businesses in Nevada
b_file = "/content/drive/My Drive/Spring2020INF553_hw2/business.json"
b_nevada = sc.textFile(b_file)\
.map(lambda x: json.loads(x))\
.filter(lambda x: x['state'] == 'NV')\
.map(lambda x: x['business_id']).persist()

#Only run once. Time-intensive.
b_nevada_s = set(b_nevada.collect()) 

#Reviews of businesses in Nevada - TESTING
r_file = "/content/drive/My Drive/Spring2020INF553_hw2/review.json"
r_nevada = sc.textFile(r_file)\
.map(lambda x: json.loads(x))\
.map(lambda x: (x['user_id'], x['business_id']))\
.filter(lambda x: x[1] in b_nevada_s)\
.map(lambda x: (x[0] + x[1], 1))\
.reduceByKey(lambda x, y: 1)\
.map(lambda x: (x[0][:22], x[0][22:])).persist()

#Only run once. Time-intensive.
r_nevada_l = r_nevada.collect()

header = ('user_id', 'business_id')
outfile =  '/content/drive/My Drive/Colab Notebooks/user_business.csv'
write_csv(outfile, header, r_nevada_l)
