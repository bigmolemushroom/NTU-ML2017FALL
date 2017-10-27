import sys
import csv

w = [4.711194709090968, -0.017197019590548732, -0.065565469068645812, 0.28432684805901598, -0.29353689542879996, -0.10953749348433817, 0.60313531502714213, -0.51615385669989, -0.072242438746427051, 0.96503775019328686, -0.00036678797040240593, 0.00044836783234705399, -0.00019560426024042642, 7.8315877997188759e-05, 0.00058499165497514181, -7.8912668349373667e-05, -0.0015393297628089062, 0.00082757726074066855, 0.0016936383932704938]

data = []

with open(sys.argv[1], encoding='mac_roman', newline = '') as csv_file:
	csv_obj = csv.reader(csv_file)
	all_data = list(csv_obj)

	for index in range(240):
		data.append([])
		for hour in range(9):
			data[index].append(float(all_data[18*index+9][hour+2]))
	
with open(sys.argv[2], 'w') as output:
	output.write('id,value\n')
	for index in range(240):
		output.write('id_')
		output.write(str(index))
		output.write(',')
		pre = w[0]
		for x in range(9):
			pre += w[x+1] * data[index][x] + w[x+10] * (data[index][x])**2
		output.write(str(pre))
		output.write('\n')