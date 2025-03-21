# -*- coding: utf-8 -*-
import numpy as np
import csv

data_name = "waterquality"
data_file_path = "data/"+ data_name + ".npz"

history_seq_len = 14
future_seq_len = 14
train_ratio = 0.6
valid_ratio = 0.2
mask_ratio = 0.2
target_channel = 1

data = np.load(data_file_path)["data"]
print("data", data.shape)

data = data[..., target_channel]
print("raw time series shape: {0}".format(data.shape))

l, n = data.shape
num_samples = data.shape[0] - history_seq_len - future_seq_len + 1
train_num_short = round(num_samples * train_ratio)
valid_num_short = round(num_samples * valid_ratio)
test_num_short = num_samples - train_num_short - valid_num_short

masx_samples_1 = round(n * mask_ratio)

print("number of training samples:{0}".format(train_num_short))
print("number of validation samples:{0}".format(valid_num_short))
print("number of test samples:{0}".format(test_num_short))
print("number of mask samples:{0}".format(masx_samples_1))

# with open('log/run.txt', 'a+', encoding='utf-8') as fw:
#     fw.write("number of training samples:{0}\n".format(train_num_short))
#     fw.write("number of validation samples:{0}\n".format(valid_num_short))
#     fw.write("number of test samples:{0}\n".format(test_num_short))
#     fw.write("number of mask samples:{0}\n".format(masx_samples_1))

def normalize(x,max_data,min_data):
    return (x - min_data) / (max_data - min_data)

max_data,min_data = data.max(), data.min()
max_min = [max_data,min_data]
max_min = np.array(max_min)

data_new = normalize(data,max_data,min_data)

def feature_target(data,input_len,output_len):
    fin_feature = []
    fin_target = []
    data_len = data.shape[0]
    for i in range(data_len-input_len - output_len + 1):
        lin_fea_seq = data[i:i+input_len,:]
        lin_tar_seq = data[i+input_len:i+input_len + output_len,:]
        fin_feature.append(lin_fea_seq)
        fin_target.append(lin_tar_seq)
    fin_feature = np.array(fin_feature).transpose((0,2,1))
    fin_target = np.array(fin_target).transpose((0,2,1))
    return fin_feature, fin_target

raw_feature, fin_target = feature_target(data_new ,history_seq_len, future_seq_len)
# print("raw_feature", raw_feature.shape)
# print("fin_feature", fin_target.shape)

train_x_raw = raw_feature[0:train_num_short,:,:]
train_y = fin_target[0:train_num_short,:,:]

vail_x_raw = raw_feature[train_num_short:train_num_short+valid_num_short,:,:]
vail_y = fin_target[train_num_short:train_num_short+valid_num_short,:,:]

test_x_raw = raw_feature[train_num_short+valid_num_short:,:,:]
test_y = fin_target[train_num_short+valid_num_short:,:,:]

def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="distance") -> np.array:

    A = np.zeros([int(num_nodes), int(num_nodes)])

    if id_file:
        with open(id_file, "r") as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3:
                        continue
                    i, j, distance = int(item[0])-1, int(item[1])-1, float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0])-1, int(item[1])-1, float(item[2])

            if graph_type == "connect":
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    return A

graph_file_path = "data/"+ data_name + "/raw/"+ data_name + ".csv"
graph_data = np.array(get_adjacent_matrix(distance_file=graph_file_path, num_nodes=n)) + np.identity(n)

np.savez("data/" + str(history_seq_len)+ ".npz",
         train_x_raw=train_x_raw,
         train_y = train_y,
         vail_x_raw=vail_x_raw,
         vail_y=vail_y,
         test_x_raw=test_x_raw,
         test_y=test_y,
         max_min = max_min,
         graph = graph_data
         )