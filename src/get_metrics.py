import torch
from torch_geometric.loader import DataLoader
import pickle
from decimal import Decimal
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score

from train import BATCH_SIZE, seed_everything, test_similarity

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def test():
    y_true = []
    similarity = []
    for i in range(1, 11, 1):
        with open(f"../data/group{i}/group{i}.test_graphs", 'rb') as f:
            test_data = pickle.load(f)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
        model = torch.load(f"../model/model_{i}.pth")
        model.to(device)
        model.eval()
        with torch.no_grad():
            for index, batch in enumerate(test_dataloader):
                batch.to(device)
                y_true.extend(batch.y.tolist())
                s, logits = model(batch)
                if test_similarity:
                    similarity.extend(s.tolist())
                else:
                    similarity.extend(torch.softmax(logits, dim=1).tolist())
    return y_true, similarity


def get_metrics_1(y_true, similarity):
    table = [
        ['auc'], ['Threshold', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        # ['auc'], ['Threshold', 0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74, 0.77],
        ['TP'], ['TN'], ['FP'], ['FN'],
        ['+Recall(%)'], ['-Recall(%)'], ['MCC'], ['F1-Score'], ['ACC']
    ]
    # 计算指标
    auc = Decimal(roc_auc_score(y_true, similarity)).quantize(Decimal("0.001"))
    table[0].append(str(auc))
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    # for threshold in [0.53, 0.56, 0.59, 0.62, 0.65, 0.68, 0.71, 0.74, 0.77]:
        y_pred = list(map(lambda x: 1.0 if x > threshold else 0.0, similarity))  # 将相似度值换成0或1
        (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
        f1 = float(Decimal(f1_score(y_true, y_pred)).quantize(Decimal("0.001")))
        p_recall = float(Decimal(tp / (tp + fn)).quantize(Decimal("0.001")) * 100)
        n_recall = float(Decimal(tn / (tn + fp)).quantize(Decimal("0.001")) * 100)
        mcc = float(Decimal(matthews_corrcoef(y_true, y_pred)).quantize(Decimal("0.001")))
        acc = float(Decimal(accuracy_score(y_true, y_pred)).quantize(Decimal("0.001")))
        for idx, value in enumerate([tp, tn, fp, fn, p_recall, n_recall, mcc, f1, acc], start=2):
            table[idx].append(value)
    # 存储与显示
    print(tabulate(table, numalign='center', tablefmt='simple_outline'))
    with open("../result/metrics.txt", 'w', encoding='utf-8') as f:
        print(tabulate(table, numalign='center', tablefmt='simple_outline'), file=f)


def get_metrics_2(y_true, similarity):
    tn, tp, fn, fp = 0, 0, 0, 0
    auc = roc_auc_score(y_true, [prob[1] for prob in similarity])
    similarity = [prob.index(max(prob)) for prob in similarity]
    for i in range(len(y_true)):
        if (y_true[i] == 1) and (similarity[i] == 1):
            tp += 1
        elif (y_true[i] == 0) and (similarity[i] == 1):
            fp += 1
        elif (y_true[i] == 1) and (similarity[i] == 0):
            fn += 1
        elif (y_true[i] == 0) and (similarity[i] == 0):
            tn += 1

    pre = tp / (tp + fp)
    p_recall = tp / (tp + fn)
    n_recall = tn / (tn + fp)
    f1 = 2 * pre * p_recall / (pre + p_recall)
    acc = (tp + tn) / (tp + tn + fp + fn)
    print(f"auc: {auc}")
    print(f"acc: {acc}")
    print(f"f1: {f1}")
    print(f"+recall: {p_recall}")
    print(f"-recall: {n_recall}")
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    print(f"pre: {pre}\n")


if __name__ == "__main__":
    seed_everything(1)
    y_true, similarity = test()
    if test_similarity:
        get_metrics_1(y_true, similarity)
    else:
        get_metrics_2(y_true, similarity)
        get_metrics_1(y_true, [prob[1] for prob in similarity])

# if __name__ == "__main__":
#     table = [
#         ['auc'], ['Threshold', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
#         ['TP'], ['TN'], ['FP'], ['FN'],
#         ['+Recall(%)'], ['-Recall(%)'], ['MCC'], ['F1-Score'], ['ACC']
#     ]
#     with open("../y_true.pickle", 'rb') as f:
#         y_true = pickle.load(f)
#     with open("../similarity.pickle", 'rb') as f:
#         similarity = pickle.load(f)
#     # 计算指标
#     auc = Decimal(roc_auc_score(y_true, similarity)).quantize(Decimal("0.001"))
#     table[0].append(str(auc))
#     for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#         y_pred = list(map(lambda x: 1.0 if x > threshold else 0.0, similarity))  # 将相似度值换成0或1
#         (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
#         f1 = float(Decimal(f1_score(y_true, y_pred)).quantize(Decimal("0.001")))
#         p_recall = float(Decimal(tp / (tp + fn)).quantize(Decimal("0.001")) * 100)
#         n_recall = float(Decimal(tn / (tn + fp)).quantize(Decimal("0.001")) * 100)
#         mcc = float(Decimal(matthews_corrcoef(y_true, y_pred)).quantize(Decimal("0.001")))
#         acc = float(Decimal(accuracy_score(y_true, y_pred)).quantize(Decimal("0.001")))
#         for idx, value in enumerate([tp, tn, fp, fn, p_recall, n_recall, mcc, f1, acc], start=2):
#             table[idx].append(value)
#     # 存储与显示
#     print(tabulate(table, numalign='center', tablefmt='simple_outline'))
#     with open("../metrics.txt", 'w', encoding='utf-8') as f:
#         print(tabulate(table, numalign='center', tablefmt='simple_outline'), file=f)

# tn, tp, fn, fp = 0, 0, 0, 0
# with open("../y_true.pickle", 'rb') as f:
#     y_true = pickle.load(f)
# with open("../similarity.pickle", 'rb') as f:
#     similarity = pickle.load(f)
# auc = roc_auc_score(y_true, [prob[1] for prob in similarity])
# for i in range(len(y_true)):
#     if (y_true[i] == 1) and (similarity[i] == 1):
#         tp += 1
#     elif (y_true[i] == 0) and (similarity[i] == 1):
#         fp += 1
#     elif (y_true[i] == 1) and (similarity[i] == 0):
#         fn += 1
#     elif (y_true[i] == 0) and (similarity[i] == 0):
#         tn += 1
#
# pre = tp / (tp + fp)
#     p_recall = tp / (tp + fn)
#     n_recall = tn / (tn + fp)
#     f1 = 2 * pre * p_recall / (pre + p_recall)
#     acc = (tp + tn) / (tp + tn + fp + fn)
#     print(f"auc: {auc}")
#     print(f"acc: {acc}")
#     print(f"f1: {f1}")
#     print(f"+recall: {p_recall}")
#     print(f"-recall: {n_recall}")
#     print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
#     print(f"pre: {pre}\n")
