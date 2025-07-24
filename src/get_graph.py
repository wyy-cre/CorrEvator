import time
import torch
import random
import pickle
from tqdm import tqdm
from torch_geometric.data import Data


WINDOW_SIZE = 9
WORD_EMBEDDINGS_DIM = 300
WORD_EMBEDDINGS = {}
random.seed(1)
torch.manual_seed(1)


def get_data(group_id):
    with open(f"../data/group{group_id}/train_X.pkl", "rb") as f:
        train_data = pickle.load(f)
    with open(f"../data/group{group_id}/train_y.pkl", "rb") as f:
        train_label = pickle.load(f)
    with open(f"../data/group{group_id}/test_X.pkl", "rb") as f:
        test_data = pickle.load(f)
    with open(f"../data/group{group_id}/test_y.pkl", "rb") as f:
        test_label = pickle.load(f)
    return train_data, train_label, test_data, test_label


def build_graph_sw(doc_words, graph_id):
    x = []
    edge_index = []
    edge_attr = []

    doc_vocab = sorted(list(set(doc_words)), key=doc_words.index)  # 对单词去重，且保证顺序不变
    doc_word_id_map = {}  # 构建一个由单词和id组成的字典
    for i in doc_vocab:
        doc_word_id_map[i] = doc_vocab.index(i)

    # 通过滑动窗口构建图
    windows = []
    if len(doc_words) <= WINDOW_SIZE:
        windows.append(doc_words)
    else:
        for j in range(len(doc_words) - WINDOW_SIZE + 1):
            window = doc_words[j: j + WINDOW_SIZE]
            windows.append(window)

    word_pair_count = {}
    # 统计两个单词的组合的数量
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_q = window[q]
                if word_p == word_q:
                    continue
                # 单词共现次数作为权重
                word_pair_key = (word_p, word_q)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
                # 反向
                word_pair_key = (word_q, word_p)
                if word_pair_key in word_pair_count:
                    word_pair_count[word_pair_key] += 1.
                else:
                    word_pair_count[word_pair_key] = 1.
    # 统计边的权重
    for key in word_pair_count:
        p = key[0]
        q = key[1]
        edge_index.append([doc_word_id_map[p], doc_word_id_map[q]])
        edge_attr.append([word_pair_count[key]])
    # 获得节点的词嵌入
    for k, v in sorted(doc_word_id_map.items(), key=lambda item: item[1]):
        x.append(torch.tensor(WORD_EMBEDDINGS[k]) if k in WORD_EMBEDDINGS else torch.empty(WORD_EMBEDDINGS_DIM).uniform_(-0.01, 0.01))
        # x.append(torch.empty(WORD_EMBEDDINGS_DIM).uniform_(-0.01, 0.01))

    return Data(
        x=torch.stack(x) if len(x) != 0 else torch.tensor([]),
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.tensor(edge_attr),
        node_graph_id=torch.tensor([graph_id]*len(doc_vocab))
    )


def build_graph_pmi(doc_words, graph_id):
    x = []
    edge_index = []
    edge_attr = []
    doc_vocab = sorted(list(set(doc_words)), key=doc_words.index)  # 对单词去重，且保证顺序不变
    doc_word_id_map = {}  # 构建一个由单词和id组成的字典
    for i in doc_vocab:
        doc_word_id_map[i] = doc_vocab.index(i)
    word_counts = {word: 0 for word in doc_word_id_map}  # word_counts[i]表示包含该单词i的滑动窗口数量
    word_cooccur = {word_i: {word_j: 0 for word_j in doc_word_id_map} for word_i in doc_word_id_map}  # 共现矩阵
    # 通过滑动窗口构建图
    windows = []
    if len(doc_words) <= WINDOW_SIZE:
        windows.append(doc_words)
    else:
        for j in range(len(doc_words) - WINDOW_SIZE + 1):
            window = doc_words[j: j + WINDOW_SIZE]
            windows.append(window)
    # 统计包含每个单词的滑动窗口数量
    for window in windows:
        for word in window:
            word_counts[word] += 1
    # 统计单词共现的滑动窗口数量
    for window in windows:
        for p in range(1, len(window)):
            for q in range(0, p):
                word_p = window[p]
                word_q = window[q]
                if word_p == word_q:
                    continue
                word_cooccur[word_p][word_q] += 1
                word_cooccur[word_q][word_p] += 1
    # 统计边的权重
    word_pair_pmi = {}
    for word_i in doc_word_id_map:
        for word_j in doc_word_id_map:
            if word_i == word_j:
                continue
            p_i = word_counts[word_i] / len(windows)
            p_j = word_counts[word_j] / len(windows)
            p_i_j = word_cooccur[word_i][word_j] / len(windows) + 1e-6
            pmi = torch.log(torch.tensor(p_i_j / (p_i * p_j)))
            if pmi <= 0:
                pmi = 0.01
            word_pair_pmi[(word_i, word_j)] = pmi
    for key in word_pair_pmi:
        p = key[0]
        q = key[1]
        edge_index.append([doc_word_id_map[p], doc_word_id_map[q]])
        edge_attr.append([word_pair_pmi[key]])
    # 获得节点的词嵌入
    for k, v in sorted(doc_word_id_map.items(), key=lambda item: item[1]):
        x.append(torch.tensor(WORD_EMBEDDINGS[k]) if k in WORD_EMBEDDINGS else torch.empty(WORD_EMBEDDINGS_DIM).uniform_(-0.01, 0.01))

    return Data(
        x=torch.stack(x) if len(x) != 0 else torch.tensor([]),
        edge_index=torch.tensor(edge_index).t().contiguous(),
        edge_attr=torch.tensor(edge_attr),
        node_graph_id=torch.tensor([graph_id]*len(doc_vocab))
    )


def link_graph(g1, g2, label):
    # 合并边
    edge_index = torch.cat((g1.edge_index, g2.edge_index+len(g1.x)), dim=1)
    # 合并节点特征矩阵
    x1 = g1.x
    x2 = g2.x
    if x1.shape[0] == 0 and x2.shape[0] == 0:
        x = [[]]
    elif x1.shape[0] == 0:
        x = x2
    elif x2.shape[0] == 0:
        x = x1
    else:
        x = torch.cat((x1, x2))  # 将两个图的节点特征进行拼接
    # 合并边的特征
    edge_attr1 = g1.edge_attr
    edge_attr2 = g2.edge_attr
    if edge_attr1.shape[0] == 0 and edge_attr2.shape[0] == 0:
        edge_attr = [[]]
    elif edge_attr1.shape[0] == 0 or edge_attr1.shape[1] == 0:
        edge_attr = edge_attr2
    elif edge_attr2.shape[0] == 0 or edge_attr2.shape[1] == 0:
        edge_attr = edge_attr1
    else:
        edge_attr = torch.cat((edge_attr1, edge_attr2))

    node_graph_id = torch.cat((g1.node_graph_id, g2.node_graph_id))
    if label == -1:
        label = 0

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=torch.tensor(label),
        node_graph_id=node_graph_id
    )


def build_graph(data, label, method):
    graphs = []
    for item in tqdm(data):
        a_clean_title = item["A_clean_title"]
        a_title_graph = method(a_clean_title, 0)
        b_clean_title = item["B_clean_title"]
        b_title_graph = method(b_clean_title, 1)
        # 将两个图合并
        graph = link_graph(a_title_graph, b_title_graph, label[data.index(item)])
        graphs.append(graph)
    # 打乱顺序
    random.shuffle(graphs)
    return graphs


def main():
    # 加载词嵌入
    with open('../glove.6B.' + str(WORD_EMBEDDINGS_DIM) + 'd.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            _ = line.split()
            WORD_EMBEDDINGS[str(_[0])] = list(map(float, _[1:]))
    for i in range(1, 11):  # 需要处理10组数据
        print(f"group {i}")
        train_data, train_label, test_data, test_label = get_data(i)

        # 通过滑动窗口构建图
        train_graph = build_graph(train_data, train_label, build_graph_sw)
        test_graph = build_graph(test_data, test_label, build_graph_sw)

        # 通过pmi构建图
        # train_graph = build_graph(train_data, train_label, build_graph_pmi)
        # test_graph = build_graph(test_data, test_label, build_graph_pmi)

        with open(f"../data/group{i}/group{i}.train_graphs", 'wb') as f:
            pickle.dump(train_graph, f)
        with open(f"../data/group{i}/group{i}.test_graphs", 'wb') as f:
            pickle.dump(test_graph, f)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"耗时: {(time.time()-start)/60}min")
