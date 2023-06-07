from data_utils import *
from train_utils import *
from Model import *
from predict_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Loading data
train_data, test_data = load__data()

total_classes = len(train_data)
num_classes = 1
# Train preparation
perm_id = [i for i in range(total_classes)]
all_classes = np.arange(total_classes)
for i in range(len(all_classes)):
	all_classes[i] = perm_id[all_classes[i]]

n_cl_temp = 0
num_iters = total_classes//num_classes
class_map = {}
map_reverse = {}
for i, cl in enumerate(all_classes):
	if cl not in class_map:
		class_map[cl] = int(n_cl_temp)
		n_cl_temp += 1

for cl, map_cl in class_map.items():
	map_reverse[map_cl] = int(cl)

# Train
incre_model = incremental_train(outfile_train("incremental"), class_map, map_reverse, num_iters, train_data, test_data)
joint_model = joint_train(outfile_train("joint"), class_map, map_reverse, num_iters, train_data, test_data)

# Test BASR
for testNum in testDataNum:
	del_test_data = del_test(test_data, delClass, testNum)
	#predict(incre_model, del_test_data, outfile_train("incremental"), map_reverse, 100)
	predict(joint_model, del_test_data, outfile_train("joint"), map_reverse, 100)

# Delete and Retrain
minNum = delClass[0] - 1
copy_delClass = delClass.copy()
for delnum in copy_delClass:
	for _ in range(clientNum):
		train_data.pop(clientNum + clientNum*delnum)
		test_data.pop(clientNum + clientNum*delnum)
	for ele in copy_delClass:
		ele -= 1
start_train = clientNum + clientNum*(minNum + 1)
model_num = minNum
num_iters -= clientNum*delClassNum
#new_incre_model = re_incremental_train(start_train, model_num, outfile_train("incremental"), class_map, map_reverse, num_iters, train_data, test_data)
new_joint_model = re_joint_train(outfile_train("joint"), class_map, map_reverse, num_iters, train_data, test_data)
# Test BASR, after retrain
for testNum in testDataNum:
	del_test_data = del_test(test_data, delClass, testNum)
	#predict(new_incre_model, del_test_data, outfile_train("incremental"), map_reverse, testNum)
	predict(new_joint_model, del_test_data, outfile_train("joint"), map_reverse, testNum)