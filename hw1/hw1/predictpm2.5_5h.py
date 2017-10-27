import csv
from cmath import sqrt

w = [1.7579913413647081, -0.10162218311095216, 0.42987702622783253, -0.47154524301859146, -0.023459741251845797, 1.0844860036747845]

data = []

with open('test.csv', encoding='mac_roman', newline = '') as csv_file:
	csv_obj = csv.reader(csv_file)
	all_data = list(csv_obj)

	for index in range(240):
		data.append([all_data[18*index+9][hour+6] for hour in range(5)])
		
with open('pm2.5_5hours_lamda0.0.csv', 'w') as output:
	output.write('id,value\n')
	for index in range(240):
		output.write('id_')
		output.write(str(index))
		output.write(',')
		pre = w[0]
		for x in range(5):
			pre += w[x+1]*float(data[index][x])
		print(index, ' : ', pre)
		output.write(str(pre))
		output.write('\n')
