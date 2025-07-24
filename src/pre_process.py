import spacy
import pickle
import random
import time
from tqdm import tqdm


random.seed(1)
nlp = spacy.load("en_core_web_sm")


def process_data(data, flag):
    data_processed = []
    labels = []
    for group in data:
        for bug in group.values():
            for patch in bug:
                a_title = patch['bug']
                b_title = patch['patch']
                label = patch['label']
                # token
                a_clean_title = []
                for token in nlp(a_title):
                    if not token.is_punct and not token.is_space:
                        a_clean_title.append(token.lemma_)
                b_clean_title = []
                for token in nlp(b_title):
                    if not token.is_punct and not token.is_space:
                        b_clean_title.append(token.lemma_)
                assert len(b_clean_title) > 0
                data_processed.append({
                    "A_title": a_title, "A_clean_title": a_clean_title,
                    "B_title": b_title, "B_clean_title": b_clean_title
                })
                labels.append(label)
                # 随机匹配
                patch_id = patch['patch_id']
                if '_Developer_' in patch_id and flag:
                    b_title = random.choice(random.choice(list(random.choice(data).values())))["patch"]
                    b_clean_title = []
                    for token in nlp(b_title):
                        if not token.is_punct and not token.is_space:
                            b_clean_title.append(token.lemma_)
                    data_processed.append({
                        "A_title": a_title, "A_clean_title": a_clean_title,
                        "B_title": b_title, "B_clean_title": b_clean_title
                    })
                    labels.append(0)
    return data_processed, labels


def main():
    data: dict = {}
    with open("../data/bugreport_patch.txt", 'r', encoding='utf-8') as f:
        # count = 0
        for line in f:
            line_split = line.split("$$")
            if line_split[0] not in data:
                data[line_split[0]] = []
            data[line_split[0]].append({"bug": line_split[1] + " " + line_split[2], "patch_id": line_split[3], "patch": line_split[4], "label": int(line_split[5])})
            # count += 1
            # if count >= 9135 * 0.8:  # 用来控制数据集的规模
            #     break

    # 将data划分成10个组，按bug_id平均划分
    # n = len(data) // 10 + 1
    # groups = [dict(list(data.items())[i:i+n]) for i in range(0, len(data), n)]

    # 将data划分成10个组，按patch条数尽可能平均划分
    data_sorted = sorted(data.items(), key=lambda x: len(x[1]), reverse=True)
    groups = [{}, {}, {}, {}, {}, {}, {}, {}, {}, {}]
    group_len = [0] * 10
    for key, value in data_sorted:
        target_group_id = group_len.index(min(group_len))
        groups[target_group_id][key] = value
        group_len[target_group_id] += len(value)
        if len(groups[target_group_id]) == 131:
            group_len[target_group_id] += 10000

    for i in tqdm(range(10)):
        train_group = groups[:i] + groups[i+1:]
        test_group = groups[i:i+1]

        train_data, train_labels = process_data(train_group, True)
        test_data, test_labels = process_data(test_group, False)

        with open(f"../data/group{i + 1}/train_X.pkl", "wb") as f:
            pickle.dump(train_data, f)
        with open(f"../data/group{i + 1}/train_y.pkl", "wb") as f:
            pickle.dump(train_labels, f)
        with open(f"../data/group{i + 1}/test_X.pkl", "wb") as f:
            pickle.dump(test_data, f)
        with open(f"../data/group{i + 1}/test_y.pkl", "wb") as f:
            pickle.dump(test_labels, f)


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"耗时: {(time.time() - start) / 60} min")
