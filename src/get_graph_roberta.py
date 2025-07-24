import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # huggingface换源
import time
import torch
import random
import pickle
from tqdm import tqdm
from torch_geometric.data import Data
from transformers import RobertaTokenizer, RobertaModel


WINDOW_SIZE = 9
WORD_EMBEDDINGS_DIM = 768
random.seed(1)
torch.manual_seed(1)


tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
roberta_model.eval()
roberta_model.cuda()  # 如果使用CPU请注释这行

# ✨ 关键修改3：重命名缓存变量
ROBERTA_EMBEDDING_CACHE = {}


def get_roberta_embedding(word):
    # ✨ 关键修改4：使用RoBERTa缓存
    if word in ROBERTA_EMBEDDING_CACHE:
        return ROBERTA_EMBEDDING_CACHE[word]

    # ✨ 关键修改5：禁用特殊标记（避免添加<s>等标记）
    inputs = tokenizer(word, return_tensors="pt", truncation=True,
                       max_length=5, add_special_tokens=False)
    inputs = {k: v.cuda() for k, v in inputs.items()}  # 如果使用CPU请删除这行

    with torch.no_grad():
        # ✨ 关键修改6：使用RoBERTa模型获取嵌入
        outputs = roberta_model(**inputs)

    # ✨ 关键修改7：取所有子词的平均作为单词表示
    embedding = outputs.last_hidden_state[0].mean(dim=0)
    ROBERTA_EMBEDDING_CACHE[word] = embedding  # 更新缓存
    return embedding


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

    doc_vocab = sorted(list(set(doc_words)), key=doc_words.index)
    doc_word_id_map = {word: idx for idx, word in enumerate(doc_vocab)}

    windows = [doc_words[j: j + WINDOW_SIZE] for j in range(max(1, len(doc_words) - WINDOW_SIZE + 1))]

    word_pair_count = {}
    for window in windows:
        for p in range(1, len(window)):
            for q in range(p):
                word_p = window[p]
                word_q = window[q]
                if word_p == word_q:
                    continue
                for a, b in [(word_p, word_q), (word_q, word_p)]:
                    word_pair_key = (a, b)
                    word_pair_count[word_pair_key] = word_pair_count.get(word_pair_key, 0) + 1.

    for (p, q), weight in word_pair_count.items():
        edge_index.append([doc_word_id_map[p], doc_word_id_map[q]])
        edge_attr.append([weight])

    # ✨ 关键修改8：调用RoBERTa嵌入函数
    for k in sorted(doc_word_id_map, key=lambda x: doc_word_id_map[x]):
        x.append(get_roberta_embedding(k))

    return Data(
        x=torch.stack(x) if x else torch.tensor([]),
        edge_index=torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr) if edge_attr else torch.empty((0, 1)),
        node_graph_id=torch.tensor([graph_id] * len(doc_vocab))
    )


def build_graph_pmi(doc_words, graph_id):
    x = []
    edge_index = []
    edge_attr = []

    doc_vocab = sorted(list(set(doc_words)), key=doc_words.index)
    doc_word_id_map = {word: idx for idx, word in enumerate(doc_vocab)}

    windows = [doc_words[j: j + WINDOW_SIZE] for j in range(max(1, len(doc_words) - WINDOW_SIZE + 1))]

    word_counts = {word: 0 for word in doc_word_id_map}
    word_cooccur = {w1: {w2: 0 for w2 in doc_word_id_map} for w1 in doc_word_id_map}

    for window in windows:
        for word in window:
            word_counts[word] += 1
        for i in range(1, len(window)):
            for j in range(i):
                if window[i] != window[j]:
                    word_cooccur[window[i]][window[j]] += 1
                    word_cooccur[window[j]][window[i]] += 1

    word_pair_pmi = {}
    for w1 in doc_word_id_map:
        for w2 in doc_word_id_map:
            if w1 == w2:
                continue
            p_i = word_counts[w1] / len(windows)
            p_j = word_counts[w2] / len(windows)
            p_ij = word_cooccur[w1][w2] / len(windows) + 1e-6
            pmi = torch.log(torch.tensor(p_ij / (p_i * p_j)))
            if pmi <= 0:
                pmi = torch.tensor(0.01)
            word_pair_pmi[(w1, w2)] = pmi

    for (p, q), pmi in word_pair_pmi.items():
        edge_index.append([doc_word_id_map[p], doc_word_id_map[q]])
        edge_attr.append([pmi.item()])

    # ✨ 关键修改9：调用RoBERTa嵌入函数
    for k in sorted(doc_word_id_map, key=lambda x: doc_word_id_map[x]):
        x.append(get_roberta_embedding(k))

    return Data(
        x=torch.stack(x) if x else torch.tensor([]),
        edge_index=torch.tensor(edge_index).t().contiguous() if edge_index else torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.tensor(edge_attr) if edge_attr else torch.empty((0, 1)),
        node_graph_id=torch.tensor([graph_id] * len(doc_vocab))
    )


def link_graph(g1, g2, label):
    edge_index = torch.cat((g1.edge_index, g2.edge_index + len(g1.x)), dim=1)
    x = torch.cat((g1.x, g2.x)) if g1.x.shape[0] and g2.x.shape[0] else (g1.x if g1.x.shape[0] else g2.x)
    edge_attr = torch.cat((g1.edge_attr, g2.edge_attr)) if g1.edge_attr.shape[0] and g2.edge_attr.shape[0] else (
        g1.edge_attr if g1.edge_attr.shape[0] else g2.edge_attr)
    node_graph_id = torch.cat((g1.node_graph_id, g2.node_graph_id))
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(label if label != -1 else 0),
                node_graph_id=node_graph_id)


def build_graph(data, label, method):
    graphs = []
    for item in tqdm(data):
        a_title_graph = method(item["A_clean_title"], 0)
        b_title_graph = method(item["B_clean_title"], 1)
        graph = link_graph(a_title_graph, b_title_graph, label[data.index(item)])
        graphs.append(graph)
    random.shuffle(graphs)
    return graphs


def main():
    for i in range(1, 11):
        print(f"group {i}")
        train_data, train_label, test_data, test_label = get_data(i)

        # 使用 RoBERTa + 滑动窗口构图
        train_graph = build_graph(train_data, train_label, build_graph_sw)
        test_graph = build_graph(test_data, test_label, build_graph_sw)

        # 如果改用 PMI 图构建，请注释上面两行，取消下方注释：
        # train_graph = build_graph(train_data, train_label, build_graph_pmi)
        # test_graph = build_graph(test_data, test_label, build_graph_pmi)

        with open(f"../data/group{i}/group{i}.train_graphs", 'wb') as f:
            pickle.dump(train_graph, f)
        with open(f"../data/group{i}/group{i}.test_graphs", 'wb') as f:
            pickle.dump(test_graph, f)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"耗时: {(time.time() - start) / 60:.2f}min")