import time
import torch
from torch_geometric.loader import DataLoader
import pickle
from decimal import Decimal
from tabulate import tabulate
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, matthews_corrcoef, accuracy_score
from train2 import BATCH_SIZE, seed_everything, test_similarity, GraphMatchingNet, EPOCH


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def format_number(n, size: str):
    return Decimal(n).quantize(Decimal(size))


def get_metrics_1(y_true, similarity):
    table = [
        ['auc'], ['Threshold', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ['TP'], ['TN'], ['FP'], ['FN'],
        ['+Recall(%)'], ['-Recall(%)'], ['MCC'], ['F1-Score'], ['ACC']
    ]
    # 计算指标
    auc = format_number(roc_auc_score(y_true, similarity), '0.001')
    table[0].append(str(auc))
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = list(map(lambda x: 1.0 if x > threshold else 0.0, similarity))  # 将相似度值换成0或1
        (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
        f1 = float(format_number(f1_score(y_true, y_pred), "0.001"))
        p_recall = float(format_number(tp / (tp + fn), "0.001") * 100)
        n_recall = float(format_number(tn / (tn + fp), "0.001") * 100)
        mcc = float(format_number(matthews_corrcoef(y_true, y_pred), "0.001"))
        acc = float(format_number(accuracy_score(y_true, y_pred), "0.001"))
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
    print(f"auc: {format_number(auc, '0.001')}")
    print(f"acc: {format_number(acc, '0.001')}")
    print(f"f1: {format_number(f1, '0.001')}")
    print(f"+recall: {format_number(p_recall, '0.001')}")
    print(f"-recall: {format_number(n_recall, '0.001')}")
    print(f"tp: {tp}, tn: {tn}, fp: {fp}, fn: {fn}")
    print(f"pre: {format_number(pre, '0.001')}\n")


def get_best_model():
    best_metric = [() for _ in range(10)]
    for group_id in range(1, 11, 1):
        group_start_time = time.time()
        print(f"group {group_id}")
        with open(f"../data/group{group_id}/group{group_id}.test_graphs", "rb") as f:
            test_data = pickle.load(f)
        test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
        best_auc = 0
        for epoch_id in range(1, EPOCH + 1, 1):
            y_true, similarity, prob = [], [], []
            model = GraphMatchingNet()
            model.to(device)
            model.load_state_dict(torch.load(f"../model/group{group_id}/state_dict_{epoch_id}.pth", weights_only=True))
            model.eval()
            with torch.no_grad():
                for index, batch in enumerate(test_dataloader):  # type: ignore
                    batch.to(device)
                    y_true.extend(batch.y.tolist())
                    s, logits = model(batch)
                    if test_similarity:
                        similarity.extend(s.tolist())
                    else:
                        prob.extend(torch.softmax(logits, dim=1).tolist())
            if test_similarity:
                test_auc = roc_auc_score(y_true, similarity)
            else:
                test_auc = roc_auc_score(y_true, [p[1] for p in prob])
            if test_auc >= best_auc:
                best_auc = test_auc
                best_metric[group_id-1] = (epoch_id, best_auc, [y_true, similarity if test_similarity else prob])  # type: ignore
            print(f"epoch {epoch_id}, valid auc: {format_number(test_auc, '0.000001')}")
        print(f"group_time {format_number((time.time() - group_start_time) / 60, '0.001')}min, valid best auc: epoch {best_metric[group_id-1][0]}, {format_number(float(best_metric[group_id-1][1]), '0.000001')}\n")
    return best_metric


def main():
    seed_everything(1)
    best_metric = get_best_model()
    y_true_all, y_pred_all = [], []
    for i in best_metric:
        y_true_all.extend(i[2][0])
        y_pred_all.extend(i[2][1])
    if test_similarity:
        get_metrics_1(y_true_all, y_pred_all)
    else:
        get_metrics_2(y_true_all, y_pred_all)
        get_metrics_1(y_true_all, [prob[1] for prob in y_pred_all])


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"耗时：{(time.time()-start)/60}min")
