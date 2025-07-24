import os
import time
import pickle
import torch
import random
import numpy as np
from decimal import Decimal
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from gmn import GraphMatchingNet
from loss import PairwiseMarginLoss, FocalLoss, MultiLoss, MSELoss, NLLLoss, KLDivLoss, BCELoss, SoftMarginLoss, CosineEmbeddingLoss, DiceLoss, LovaszLoss, OhemCELoss, SELoss, GHMLoss, DynamicBoundaryLoss
from torch import nn


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
BATCH_SIZE = 128
SEED = 1
EPOCH = 10
LR = 0.00005
test_similarity = True  # True表示采用相似度进行评估，False表示采用概率值进行评估


def seed_everything(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)  # LovaszLoss需要注释该行代码


def format_number(n, size: str):
    return Decimal(n).quantize(Decimal(size))


def main():
    for group_id in range(1, 11, 1):
        torch.cuda.empty_cache()
        begin = time.time()
        print(f"group {group_id}")
        # 读取数据集
        with open(f"../data/group{group_id}/group{group_id}.train_graphs", 'rb') as f:
            train_data = pickle.load(f)
        # with open(f"../data/group{group_id}/group{group_id}.test_graphs", 'rb') as f:
        #     test_data = pickle.load(f)
        train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        # test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)
        model = GraphMatchingNet()
        model.to(device)
        # criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
        # criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.4, 0.6]).to(device))  # 带权重的交叉熵损失函数
        # criterion = PairwiseMarginLoss(gamma=0.7)  # 自定义边界Loss
        criterion = DynamicBoundaryLoss(margin=0)
        # criterion = FocalLoss(alpha=0.95, gamma=2)
        # criterion = DiceLoss()
        # criterion = MultiLoss()
        # criterion = MSELoss()
        # criterion = NLLLoss()
        # criterion = KLDivLoss()
        # criterion = BCELoss(pos_weight=torch.tensor([8.0]).to(device))
        # criterion = SoftMarginLoss()
        # criterion = CosineEmbeddingLoss()
        # criterion = LovaszLoss()
        # criterion = OhemCELoss(0.7)
        # criterion = SELoss().to(device)
        # criterion = GHMLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # Adam优化器
        # 训练
        model.train()
        # best_metric = 0  # 记录某个epoch的最佳验证集指标
        for epoch_id in range(EPOCH):
            start = time.time()
            loss_epoch = 0  # 记录每个epoch的loss的和，用于输出
            y_pred_epoch = []
            y_true_epoch = []
            for index, batch in enumerate(train_dataloader):  # type: ignore
                batch.to(device)
                # label = batch.y.to(torch.float32)
                label = batch.y  # 交叉熵损失需要使用整型
                optimizer.zero_grad()
                s, logits = model(batch)
                # loss = criterion(s, logits, label)  # MultiLoss
                if test_similarity:
                    loss = criterion(s, label)
                else:
                    loss = criterion(logits, label)
                # loss = criterion(s, label, model)  # 加入L2正则化的Focal Loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                loss_epoch += loss * len(label)
                y_true_epoch.extend(label.tolist())
                if test_similarity:
                    y_pred_epoch.extend(s.tolist())
                else:
                    y_pred_epoch.extend(torch.softmax(logits, dim=1).tolist())
            if test_similarity:
                print(f"epoch: {epoch_id + 1}, train loss: {format_number(float(loss_epoch) / len(train_data), '0.000001')}, auc: {format_number(roc_auc_score(y_true_epoch, y_pred_epoch), '0.000001')}, time: {format_number((time.time() - start)/60, '0.001')}min")
            else:
                print(f"epoch: {epoch_id + 1}, train loss: {format_number(float(loss_epoch) / len(train_data), '0.000001')}, auc: {format_number(roc_auc_score(y_true_epoch, [prob[1] for prob in y_pred_epoch]), '0.000001')}, time: {format_number((time.time() - start) / 60, '0.001')}min")
            torch.save(model.state_dict(), f"../model/group{group_id}/state_dict_{epoch_id + 1}.pth")
        print(f"group_time: {format_number((time.time()-begin)/60, '0.001')}\n")
        # time.sleep(1)


if __name__ == "__main__":
    seed_everything(SEED)
    main()
