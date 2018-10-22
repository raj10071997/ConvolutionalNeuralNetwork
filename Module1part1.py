import pickle
import numpy as np
from NearestNeighbourClass import NearestNeighbour, KNearestNeighbour

train_dic={}
train_data=[]
train_labels=[]
test_dic={}
test_data=[]
test_labels=[]

for i in range(1,6):
	file = "data_batch_%d" % i 
	with open(file,'rb') as fo:
		train_dic[file] = pickle.load(fo, encoding='latin1')

with open("test_batch",'rb') as fo:
	test_dic = pickle.load(fo,encoding='latin1')

def load_cifar():
	global train_data, train_labels, test_data, test_labels
	for key in train_dic.keys():
		if train_data==[]:
			train_data = train_dic[key]["data"]
			train_labels = train_dic[key]["labels"]
		else:
			# print("key = ",key)
			# print(train_dic[key]["data"][0])
			# print(train_dic[key]["labels"][0])
			train_data = np.concatenate((train_data,train_dic[key]["data"]),axis=0)
			train_labels.extend(train_dic[key]["labels"])

	test_data = test_dic["data"]
	test_labels = test_dic["labels"]

load_cifar()

nrc = NearestNeighbour()

nrc.train(train_data,train_labels)

results = nrc.predict(test_data[:20])
acurracyList = [int(a==b) for a,b in zip(results,test_labels[:20])]
print("acurracy of nearest neighbour: %f" % (np.mean(acurracyList)))

knn = KNearestNeighbour()
accuracies=[]
for k in [1, 3, 5, 10, 20, 50, 100]:
	knn.train(train_data,train_labels)
	results = knn.predict(test_data[:200],k)

	accuracy = np.mean(results==test_labels[:200])

	accuracies.append((k,accuracy))

print(accuracies)


# print(type(train_data))
# print(train_data[10000])
# print(train_labels[10000])




		
		







