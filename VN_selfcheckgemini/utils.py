import numpy as np
# SelfCheck - BERTScore utils
def expand_list1(mylist, num):
    expanded = []
    for x in mylist:
        for _ in range(num):
            expanded.append(x)
    return expanded

def expand_list2(mylist, num):
    expanded = []
    for _ in range(num):
        for x in mylist:
            expanded.append(x)
    return expanded


def prepare_qa_input(t5_tokenizer, context, device):
    """
    input: context
    output: question <sep> answer
    """
    encoding = t5_tokenizer(
        [context],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids


def prepare_distractor_input(t5_tokenizer, context, question, answer, device, separator='<sep>'):
    """
    input: question <sep> answer <sep> article
    output: distractor1 <sep> distractor2 <sep> distractor3
    """
    input_text = question + ' ' + separator + ' ' + answer + ' ' + separator + ' ' + context
    encoding = t5_tokenizer(
        [input_text],
        return_tensors="pt",
    )
    input_ids = encoding.input_ids.to(device)
    return input_ids


def prepare_answering_input(
    tokenizer, # longformer_tokenizer
    question, options, context,
    device, max_seq_length=4096,
):
    c_plus_q = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_4 = [c_plus_q] * len(options)

    tokenized_examples = tokenizer(
        c_plus_q_4, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    tokenized_examples = tokenized_examples.to(device)
    input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)

    example_encoded = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    return example_encoded

def method_simple_counting(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    AT,
):
    """
    simple counting method score => count_mismatch / (count_match + count_mismatch)
    :return score: 'inconsistency' score
    """
    # bad questions, i.e. not answerable given the passage
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_good_sample, count_match = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            count_good_sample += 1
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
    if count_good_sample == 0:
        score = 0.5
    else:
        score = (count_good_sample-count_match) / count_good_sample
    return score

def method_vanilla_bayes(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2, AT,
):
    """
    (vanilla) bayes method score: compute P(sentence is non-factual | count_match, count_mismatch)
    :return score: 'inconsistency' score
    """
    if u_score < AT:
        return 0.5
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        if u_score_s[s] >= AT:
            a_S = np.argmax(prob_s[s])
            if a_DT == a_S:
                count_match += 1
            else:
                count_mismatch += 1
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score

def method_bayes_with_alpha(
    prob,
    u_score,
    prob_s,
    u_score_s,
    num_samples,
    beta1, beta2,
):
    """
    bayes method (with answerability score, i.e. soft-counting) score
    :return score: 'inconsistency' score
    """
    a_DT = np.argmax(prob)
    count_match, count_mismatch = 0, 0
    for s in range(num_samples):
        ans_score = u_score_s[s]
        a_S = np.argmax(prob_s[s])
        if a_DT == a_S:
            count_match += ans_score
        else:
            count_mismatch += ans_score
    gamma1 = beta2 / (1.0-beta1)
    gamma2 = beta1 / (1.0-beta2)
    score = (gamma2**count_mismatch) / ((gamma1**count_match) + (gamma2**count_mismatch))
    return score