from BERTScoreModel import BERTScoreModel
import ast
import pandas as pd
import numpy as np
from scipy.stats import entropy
from tqdm.notebook import tqdm
from datasets import Dataset
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt

label_mapping = {
    'accurate': 0.0,
    'minor_inaccurate': 0.5,
    'major_inaccurate': 1.0,
}

def read_csv(path):
    df = pd.read_csv(path)

    for col in df.columns[2:]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))
    return Dataset.from_pandas(df)
def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled

dataset = read_csv('../Data/File/Vietnamese_hallucination_annotated.csv')


human_label_detect_False   = {}
human_label_detect_False_h = {}
human_label_detect_True    = {}
for idx, i_ in enumerate(range(len(dataset))):
    dataset_i = dataset[i_]
    raw_label = np.array([label_mapping[x] for x in dataset_i['annotation']])
    human_label_detect_False[idx] = (raw_label > 0.499).astype(np.int32).tolist()
    human_label_detect_True[idx]  = (raw_label < 0.499).astype(np.int32).tolist()
    average_score = np.mean(raw_label)
    if average_score < 0.99:
        human_label_detect_False_h[idx] = (raw_label > 0.99).astype(np.int32).tolist()

selfcheck_scores_avg_list = []
selfcheck_scores_max_list = []
n_gram = 5
selfcheck_scores_avg = {} # average sentence-level scores
selfcheck_scores_max = {} # max sentence-level scores
for i in tqdm(range(len(dataset))):
    x = dataset[i]
    selfcheck_scores_ = BERTScoreModel().predict(
        sentences=x['gemini_sentences'],
        sampled_passages=x['gemini_text_samples']
    )

    selfcheck_scores_avg[i] = selfcheck_scores_

selfcheck_scores_avg_list.append(selfcheck_scores_avg)
selfcheck_scores_avg = selfcheck_scores_avg_list[0]

def tmp_fix(selfcheck_scores):
    for i in range(len(selfcheck_scores)):
        for k, v in selfcheck_scores[i].items():
            for j in range(len(v)):
                if v[j] > 10e5:
                    selfcheck_scores[i][k][j] = 10e5

tmp_fix(selfcheck_scores_avg_list)



def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
    indices = [k for k in human_labels.keys()]
    unroll_preds = unroll_pred(preds, indices)
    if oneminus_pred:
        unroll_preds = [1.0-x for x in unroll_preds]
    unroll_labels = unroll_pred(human_labels, indices)
    assert len(unroll_preds) == len(unroll_labels)
    print("len:", len(unroll_preds))
    P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    return P, R

def generate_table_results(selfcheck_scores):
    df = pd.DataFrame(columns=['n-gram', 'NoFac', 'NoFac*', 'Fac'])
    for idx, n in enumerate(range(1, n_gram + 1)):
        selfcheck_scores_ = selfcheck_scores[idx]
        try:
            df.loc[idx, 'n-gram'] = n

            Prec, Rec = get_PR_with_human_labels(selfcheck_scores_, human_label_detect_False, pos_label=1)
            df.loc[idx, 'NoFac'] = auc(Rec, Prec)*100

            print(df.loc[idx, 'NoFac'])

            Prec, Rec = get_PR_with_human_labels(selfcheck_scores_, human_label_detect_False_h, pos_label=1)
            df.loc[idx, 'NoFac*'] = auc(Rec, Prec)*100

            print(df.loc[idx, 'NoFac*'])

            Prec, Rec = get_PR_with_human_labels(selfcheck_scores_, human_label_detect_True, pos_label=1, oneminus_pred=True)
            df.loc[idx, 'Fac'] = auc(Rec, Prec)*100

        except Exception as e:
            print(e)
            continue

    return df

generate_table_results(selfcheck_scores_max_list)