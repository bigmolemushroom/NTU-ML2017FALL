import csv
from cmath import sqrt

w = [1.7331572776124444, -0.033181621329013797, -0.025917448928015483, 0.21810639485228228, -0.23961750510023933, -0.054494165678226972, 0.53082020381567918, -0.57206786159740419, 0.00080128061453303711, 1.0945962965911327]

data = []

with open('test.csv', encoding='mac_roman', newline = '') as csv_file:
	csv_obj = csv.reader(csv_file)
	all_data = list(csv_obj)

	for index in range(240):
		data.append([all_data[18*index+9][hour+2] for hour in range(9)])
		
with open('pm2.5_9hours_lamda0.0001.csv', 'w') as output:
	output.write('id,value\n')
	for index in range(240):
		output.write('id_')
		output.write(str(index))
		output.write(',')
		pre = w[0]
		for x in range(9):
			pre += w[x+1]*float(data[index][x])
		print(index, ' : ', pre)
		output.write(str(pre))
		output.write('\n')
