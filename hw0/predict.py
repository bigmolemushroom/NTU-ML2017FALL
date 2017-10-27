import sys
import csv

w = 

with open(sys.argv[1], encoding='mac_roman', newline = '') as csv_file:
	csv_obj = csv.reader(csv_file)
	all_data = list(csv_obj)

	for index in range(240):
		for hour in range(9):
			data[index].append(float(all_data[18*index+arg][hour+2]))
	
with open(sys.argv[2], 'w') as output:
	output.write('id,value\n')
	for n in range(240):
		output.write('id_')
		output.write(str(n))
		output.write(',')
		pre = 0
		for x in range(163):
			pre += w[x] * data[n][x]
		print(n, ' : ', pre)
		output.write(str(pre))
		output.write('\n')


