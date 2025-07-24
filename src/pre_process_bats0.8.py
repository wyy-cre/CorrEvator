import json
import pickle


data = []
label = []
for i in range(1, 11):
    with open(f"../data/RQ1/BATS0.8/preprocessed_BATS0.8_{i}_test_0.txt", "r", encoding="utf-8") as f:
        data.append(json.load(f))
    with open(f"../data/RQ1/BATS0.8/preprocessed_BATS0.8_{i}_test_1.txt", "r", encoding="utf-8") as f:
        data[-1] += json.load(f)
    with open(f"../data/RQ1/BATS0.8/preprocessed_BATS0.8_{i}_test_0y.txt", "r", encoding="utf-8") as f:
        label.append(json.load(f))
    with open(f"../data/RQ1/BATS0.8/preprocessed_BATS0.8_{i}_test_1y.txt", "r", encoding="utf-8") as f:
        label[-1] += json.load(f)
for i in range(10):
    train_data = []
    train_label = []
    for j in range(10):
        if j != i:
            train_data += data[j]
            train_label += label[j]
    test_data = data[i]
    test_label = label[i]
    with open(f"../data/group{i + 1}/train_X.pkl", "wb") as f:
        pickle.dump(train_data, f)
    with open(f"../data/group{i + 1}/train_y.pkl", "wb") as f:
        pickle.dump(train_label, f)
    with open(f"../data/group{i + 1}/test_X.pkl", "wb") as f:
        pickle.dump(test_data, f)
    with open(f"../data/group{i + 1}/test_y.pkl", "wb") as f:
        pickle.dump(test_label, f)
