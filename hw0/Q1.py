import sys

result = []
with open(sys.argv[1], 'r') as file:
	dataString = file.read()
	dataList = dataString.split(" ")
	dataList[-1] = dataList[-1].replace('\n', '')
	for x in dataList:
		for y in result:
			if(x == y[0]):
				y[1] += 1
				break
		else:
			result.append([x, 1])
with open('Q1.txt', 'w') as file:
	for i in range(0, len(result)):
		if i != 0:
			file.write('\n')
		file.write(result[i][0]+' ')
		file.write(str(i)+' ')
		file.write(str(result[i][1]))