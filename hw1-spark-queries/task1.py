from pyspark import SparkContext
import sys
import json
from operator import add
import string

#Pull out arguments from command line
input_file = sys.argv[1]
output_file = sys.argv[2]
stopwords = sys.argv[3]
year = int(sys.argv[4])
m = int(sys.argv[5])
n = int(sys.argv[6])

#Set up spark context and root RDD
sc = SparkContext(appName="inf553")
r_lines = sc.textFile(input_file)
r_json = r_lines.map(lambda x: json.loads(x))

#PART A: Get total number of reviews
r_count = r_json.map(lambda x: ('count', 1))
r_final = r_count.reduceByKey(lambda x, y: x + y)
result = r_final.collect() #op: one tuple of ('count', 100)
resultA = result[0][1]
print('Result A: ', resultA)

#PART B: Get total number of reviews for given year #NOT WORKING
r_select = r_json.map(lambda x: (int(x['date'][:4]), 1)) #output: ('YYYY', 1)
r_filter = r_select.filter(lambda x: x[0] == year) #filter for desired year
r_final = r_filter.reduceByKey(lambda x, y: x + y) #output: ('YYYY', count)
result = r_final.collect()
resultB = result[0][1]
print('Result B: ', resultB)

#PART C: Get number distinct reviews for given year
r = r_json.map(lambda x: (x['user_id'], 1))\
.reduceByKey(lambda x, y: 1)\
.map(lambda x: (1, 1))\
.reduceByKey(lambda x, y: x + y)\
.collect()

resultC = r[0][1]
print('Result C: ', resultC)

#PART D: Top m users who have the largest number of reviews and its count
r_select = r_json.map(lambda x: (x['user_id'], 1)) #output: (user_id, 1)
r_reduce = r_select.reduceByKey(add) #output: (distinct user_id, count)
r_map = r_reduce.map(lambda x: (x[1], x[0])) #output: (count, distinct user_id)
r_sort = r_map.sortByKey(False) #sort descending by user's count of reviews
r_final = r_sort.map(lambda x: (x[1], x[0]))
r_take = r_final.take(m) #select top m users
resultD = [list(i) for i in r_take]
print('Result D: ', resultD)


#PART E: Top n frequent words. Remove punctuation and stopwords.
#stopword removal
with open(stopwords) as f:
	stop_list = [line.rstrip('\n') for line in f]
stop_dict = {}
for word in stop_list:
	stop_dict[word] = 1

#punctuation removal
sp_chars = ['(', '[', ',', '.', '!', '?', ':', ';', ']', ')']
del_d = {sp_char: '' for sp_char in string.punctuation} 
del_d[' '] = ''
t = str.maketrans(del_d)

r = r_json.flatMap(lambda x: [(val.lower().strip().translate(t), 1) for val in x['text'].split(' ')])\
.filter(lambda x: x[0] not in stop_dict)\
.filter(lambda x: x[0] != '')\
.reduceByKey(add)\
.sortBy(lambda x: -x[1])\
.take(n)
resultE = [i[0] for i in r]

result = {'A': resultA, 'B': resultB, 'C': resultC, 'D': resultD, 'E': resultE}

with open(output_file, 'w') as outfile:
	json.dump(result, outfile)
