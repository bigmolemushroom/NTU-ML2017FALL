import sys
import csv

w = [4.711194709090968, -0.017197019590548632, -0.065565469068645715, 0.28432684805901609, -0.29353689542880007, -0.10953749348433826, 0.60313531502714202, -0.51615385669989011, -0.072242438746427148, 0.96503775019328675, -0.00036678797040230591, 0.00044836783234715401, -0.00019560426024032643, 7.8315877997088754e-05, 0.00058499165497524178, -7.8912668349273663e-05, -0.0015393297628088062, 0.00082757726074056859, 0.0016936383932703938]

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