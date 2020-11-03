import csv

def write_csv(outfile, header, content):
	'''Writes a list of tuples to csv.
	outfile - filepath to google drive
	content - list of tuples such as [('u1', 'b1'), ('u1', 'b2')] '''
	with open(outfile, 'w') as fd:
		write = csv.writer(fd)
		write.writerow(header)
		for row in content:
			write.writerow(row)

content = [('=221', 'b1'), ('-7u1', 'b2')]
header = ('user_id', 'business_id')
outfile =  'test.csv'

write_csv(outfile, header, content)