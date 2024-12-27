from BERTScoreModel import BERTScoreModel, SimpleRescaleBaseline
from MQAGModel import MQAGModel  # Import MQAGModel
import ast
import pandas as pd
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from datasets import Dataset
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import torch
import argparse  # Import argparse for argument parsing

label_mapping = {
    'accurate': 0.0,
    'minor_inaccurate': 0.5,
    'major_inaccurate': 1.0,
}

def read_csv(path):
    df = pd.read_csv(path)

    for col in df.columns[2:]:
        df[col] = df[col].apply(lambda x: ast.literal_eval(x))
    return Dataset.from_pandas(df.head(2))

def unroll_pred(scores, indices):
    unrolled = []
    for idx in indices:
        unrolled.extend(scores[idx])
    return unrolled

def tmp_fix(selfcheck_scores):
    for k, v in selfcheck_scores.items():
        for j in range(len(v)):
            if v[j] > 10e5:
                selfcheck_scores[k][j] = 10e5

def get_PR_with_human_labels(preds, human_labels, pos_label=1, oneminus_pred=False):
    indices = [k for k in human_labels.keys()]
    unroll_preds = unroll_pred(preds, indices)
    if oneminus_pred:
        unroll_preds = [1.0-x for x in unroll_preds]
    unroll_labels = unroll_pred(human_labels, indices)
    assert len(unroll_preds) == len(unroll_labels)
    # print("len:", len(unroll_preds))
    P, R, thre = precision_recall_curve(unroll_labels, unroll_preds, pos_label=pos_label)
    return P, R

def plot_PR_curve(ax, human_labels, R, P, title, model_name):
    arr = []
    for k, v in human_labels.items():
        arr.extend(v)
    random_baseline = np.mean(arr)
    ax.hlines(y=random_baseline, xmin=0, xmax=1.0, color='grey', linestyles='dotted', label='Random')
    ax.plot(R, P, '-', label=model_name)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend()
    return ax

def main():
    parser = argparse.ArgumentParser(description="Run hallucination detection metrics")
    parser.add_argument('--baseline', nargs='+', choices=['MQAG', 'BERTScore-vi', 'BERTScore-en'], required=True, 
                       help="Choose one or more models to use: MQAG, BERTScore-vi, BERTScore-en")
    args = parser.parse_args()

    dataset = read_csv('../Data/File/Vietnamese_hallucination_annotated.csv')
    # dataset = read_csv('/content/Vietnamese_hallucination_annotated.csv')

    human_label_detect_False = {}
    human_label_detect_False_h = {}
    human_label_detect_True = {}
    for idx, i_ in enumerate(range(len(dataset))):
        dataset_i = dataset[i_]
        raw_label = np.array([label_mapping[x] for x in dataset_i['annotation']])
        human_label_detect_False[idx] = (raw_label > 0.499).astype(np.int32).tolist()
        human_label_detect_True[idx] = (raw_label < 0.499).astype(np.int32).tolist()
        average_score = np.mean(raw_label)
        if average_score < 0.99:
            human_label_detect_False_h[idx] = (raw_label > 0.99).astype(np.int32).tolist()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    baselines = args.baseline  # Now baselines will be a list of selected models
    
    selfcheck_scores_list = {}
    for baseline in baselines:
        selfcheck_scores = {}

        if baseline == 'BERTScore-vi':
            model = BERTScoreModel(default_model='vi')
        elif baseline == 'BERTScore-en':
            model = BERTScoreModel(default_model='en', rescale_with_baseline=True)

        # elif baseline == 'MQAG-vi':
        #     model = MQAGModel(device=device, lang='vi')
        # elif baseline == 'MQAG-en':
        #     model = MQAGModel(device=device, lang='en')

        for i in tqdm(range(len(dataset))):
            x = dataset[i]
            if baseline == 'BERTScore-vi' or baseline == 'BERTScore-en':
                bert_score_array = model.predict(
                    sentences=x['gemini_sentences'],
                    sampled_passages=x['gemini_text_samples']
                )
                selfcheck_scores[i] = bert_score_array
            elif baseline == 'MQAG':
                selfcheck_scores_ = model.predict(
                    sentences=x['gemini_sentences'],
                    passage=x['gemini_text'],
                    sampled_passages=x['gemini_text_samples']
                )
                selfcheck_scores[i] = selfcheck_scores_

        selfcheck_scores_list[baseline] = selfcheck_scores


    final_scores = {}
    for baseline, selfcheck_scores in selfcheck_scores_list.items():
        if baseline == 'BERTScore-vi':
            rescale = SimpleRescaleBaseline()

            # no rescale
            selfcheck_scores_no_rescale = {}
            for k, bert_score_array in selfcheck_scores.items():
                bert_score_per_sent = bert_score_array.mean(axis=-1)
                one_minus_bert_score_per_sent = 1.0 - bert_score_per_sent
                selfcheck_scores_no_rescale[k] = one_minus_bert_score_per_sent
            final_scores[baseline + "_no_rescale"] = selfcheck_scores_no_rescale

            # min max rescale
            min = None
            max = None
            selfcheck_scores_min_max_rescale = {}
            for bert_score_array in selfcheck_scores.values():
                min = min if (min is not None) and (min < bert_score_array.min()) else bert_score_array.min()
                max = max if (max is not None) and (max > bert_score_array.max() )else bert_score_array.max()

            for k, v in selfcheck_scores.items():
                bert_score_array = rescale.MinMaxScaler(v, min, max)
                bert_score_per_sent = bert_score_array.mean(axis=-1)
                one_minus_bert_score_per_sent = 1.0 - bert_score_per_sent
                selfcheck_scores_min_max_rescale[k] = one_minus_bert_score_per_sent
            final_scores[baseline + "_MinMaxScaler"] = selfcheck_scores_min_max_rescale
 
        elif baseline == 'BERTScore-en':
            for k, v in selfcheck_scores.items():
                bert_score_array = v
                bert_score_per_sent = bert_score_array.mean(axis=-1)
                one_minus_bert_score_per_sent = 1.0 - bert_score_per_sent
                selfcheck_scores[k] = one_minus_bert_score_per_sent
            final_scores[baseline] = selfcheck_scores
        else:
            final_scores[baseline] = selfcheck_scores

    for baseline, selfcheck_scores in final_scores.items():
        try:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            Prec, Rec = get_PR_with_human_labels(selfcheck_scores, human_label_detect_False, pos_label=1)
            noFactAuc =  auc(Rec, Prec)*100
            noFactCurve = plot_PR_curve(axs[0], human_label_detect_False, Rec, Prec, 'Non-Factual Sentences', baseline)

            Prec, Rec = get_PR_with_human_labels(selfcheck_scores, human_label_detect_False_h, pos_label=1)
            noFactHAuc = auc(Rec, Prec)*100
            noFactHAucCurve = plot_PR_curve(axs[1],human_label_detect_False_h, Rec, Prec, 'Non-Factual* Sentences', baseline)


            Prec, Rec = get_PR_with_human_labels(selfcheck_scores, human_label_detect_True, pos_label=1, oneminus_pred=True)
            factAuc = auc(Rec, Prec)*100
            factAucCurve = plot_PR_curve(axs[2],human_label_detect_True, Rec, Prec, 'Factual Sentences', baseline)

            fig.tight_layout()
            plt.show()
            print("NonFact AUC-PR: ", noFactAuc, "    NonFact* AUC-PR: ", noFactHAuc, "      Factual AUC-PR: ", factAuc)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()