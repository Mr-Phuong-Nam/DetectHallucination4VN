from typing import List
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, LongformerTokenizer, LongformerForMultipleChoice, LongformerForSequenceClassification
from utils import prepare_qa_input, prepare_distractor_input, prepare_answering_input, method_simple_counting, method_vanilla_bayes, method_bayes_with_alpha
from utils import MQAGConfig



class MQAGModel:
    """
    SelfCheckGPT (MQAG varaint): Checking LLM's text against its own sampled texts via MultipleChoice Question Answering
    """
    def __init__(
        self,
        g1_model: str = None,
        g2_model: str = None,
        answering_model: str = None,
        answerability_model: str = None,
        device = None
    ):

        g1_model = g1_model if g1_model is not None else MQAGConfig.generation1_squad
        g2_model = g2_model if g2_model is not None else MQAGConfig.generation2
        answering_model = answering_model if answering_model is not None else MQAGConfig.answering
        answerability_model = answerability_model if answerability_model is not None else MQAGConfig.answerability

        # Question Generation Systems (G1 & G2)
        self.g1_tokenizer = AutoTokenizer.from_pretrained(g1_model)
        self.g1_model = AutoModelForSeq2SeqLM.from_pretrained(g1_model)
        self.g2_tokenizer = AutoTokenizer.from_pretrained(g2_model)
        self.g2_model = AutoModelForSeq2SeqLM.from_pretrained(g2_model)

        # Question Answering System (A)
        self.a_tokenizer = LongformerTokenizer.from_pretrained(answering_model)
        self.a_model = LongformerForMultipleChoice.from_pretrained(answering_model)

        # (Un)Answerability System (U)
        self.u_tokenizer = LongformerTokenizer.from_pretrained(answerability_model)
        self.u_model = LongformerForSequenceClassification.from_pretrained(answerability_model)

        self.g1_model.eval()
        self.g2_model.eval()
        self.a_model.eval()
        self.u_model.eval()

        if device is None:
            device = torch.device("cpu")
        self.g1_model.to(device)
        self.g2_model.to(device)
        self.a_model.to(device)
        self.u_model.to(device)
        self.device = device
        print("SelfCheck-MQAG initialized to device", device)

    @torch.no_grad()
    def predict(
        self,
        sentences: List[str],
        passage: str,
        sampled_passages: List[str],
        num_questions_per_sent: int = 5,
        scoring_method: str = "bayes_with_alpha",
        **kwargs,
    ):
        """
        This function takes sentences (to be evaluated) with sampled passages (evidence), and return sent-level scores
        :param sentences: list[str] -- sentences to be evaluated, e.g. GPT text response spilt by spacy
        :param passage: str -- the passage to be evaluated, note that splitting(passage) ---> sentences
        :param sampled_passages: list[str] -- stochastically generated responses (without sentence splitting)
        :param num_questions_per_sent: int -- number of quetions to be generated per sentence
        :return sent_scores: sentence-level score of the same length as len(sentences) # inconsistency_score, i.e. higher means likely hallucination
        """
        assert scoring_method in ['counting', 'bayes', 'bayes_with_alpha']
        num_samples = len(sampled_passages)
        sent_scores = []
        for sentence in sentences:

            # Question + Choices Generation
            questions = self.question_generation_sentence_level(
                self.g1_model, self.g1_tokenizer,
                self.g2_model, self.g2_tokenizer,
                sentence, passage, num_questions_per_sent, self.device)

            # Answering
            scores = []
            max_seq_length = 4096 # answering & answerability max length
            for question_item in questions:
                question, options = question_item['question'], question_item['options']
                # response
                prob = self.answering(
                    self.a_model, self.a_tokenizer,
                    question, options, passage,
                    max_seq_length, self.device)

                u_score = self.answerability_scoring(
                    self.u_model, self.u_tokenizer,
                    question, passage,
                    max_seq_length, self.device)

                prob_s = np.zeros((num_samples, 4))
                u_score_s = np.zeros((num_samples,))
                for si, sampled_passage in enumerate(sampled_passages):

                    # sample
                    prob_s[si] = self.answering(
                        self.a_model, self.a_tokenizer,
                        question, options, sampled_passage,
                        max_seq_length, self.device)
                    u_score_s[si] = self.answerability_scoring(
                        self.u_model, self.u_tokenizer,
                        question, sampled_passage,
                        max_seq_length, self.device)

                # doing comparision
                if scoring_method == 'counting':
                    score = method_simple_counting(prob, u_score, prob_s, u_score_s, num_samples, AT=kwargs['AT'])
                elif scoring_method == 'bayes':
                    score = method_vanilla_bayes(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'], AT=kwargs['AT'])
                elif scoring_method == 'bayes_with_alpha':
                    score = method_bayes_with_alpha(prob, u_score, prob_s, u_score_s, num_samples, beta1=kwargs['beta1'], beta2=kwargs['beta2'])
                scores.append(score)
            sent_score = np.mean(scores)
            sent_scores.append(sent_score)

        return np.array(sent_scores)

    def question_generation_sentence_level(
        self,  # Add self here
        g1_model,
        g1_tokenizer,
        g2_model,
        g2_tokenizer,
        sentence,
        passage,
        num_questions_per_sent,
        device,
    ):
        qa_input_ids = prepare_qa_input(
                g1_tokenizer,
                context=sentence,
                device=device,
        )
        num_valid_questions = 0
        questions = []
        for q_ in range(num_questions_per_sent):
            # Stage G.1: question+answer generation
            outputs = g1_model.generate(
                qa_input_ids,
                max_new_tokens=128,
                do_sample=True,
            )
            question_answer = g1_tokenizer.decode(outputs[0], skip_special_tokens=False)
            question_answer = question_answer.replace(g1_tokenizer.pad_token, "").replace(g1_tokenizer.eos_token, "")
            question_answer_split = question_answer.split(g1_tokenizer.sep_token)
            if len(question_answer_split) == 2:
                # valid Question + Annswer output
                num_valid_questions += 1
            else:
                continue
            question = question_answer_split[0].strip()
            answer = question_answer_split[1].strip()

            # Stage G.2: Distractor Generation
            distractor_input_ids = prepare_distractor_input(
                g2_tokenizer,
                context = passage,
                question = question,
                answer = answer,
                device = device,
                separator = g2_tokenizer.sep_token,
            )
            outputs = g2_model.generate(
                distractor_input_ids,
                max_new_tokens=128,
                do_sample=True,
            )
            distractors = g2_tokenizer.decode(outputs[0], skip_special_tokens=False)
            distractors = distractors.replace(g2_tokenizer.pad_token, "").replace(g2_tokenizer.eos_token, "")
            distractors = re.sub("<extra\S+>", g2_tokenizer.sep_token, distractors)
            distractors = [y.strip() for y in distractors.split(g2_tokenizer.sep_token)]
            options = [answer] + distractors

            while len(options) < 4:
                # print("Warning: options =", options)
                options.append(options[-1])

            question_item = {
                'question': question,
                'options': options,
            }
            questions.append(question_item)
        return questions
    def answering(
        self,
        a_model,
        a_tokenizer,
        question,
        options,
        context,
        max_seq_length,
        device,
    ):
        answering_given_passage = prepare_answering_input(
            tokenizer=a_tokenizer,
            question=question,
            options=options,
            context=context,
            device=device,
            max_seq_length=max_seq_length,
        )
        answering_outputs = a_model(**answering_given_passage)
        prob = torch.softmax(answering_outputs['logits'], dim=-1)[0].cpu().numpy()
        return prob
    def answerability_scoring(
        self,
        u_model,
        u_tokenizer,
        question,
        context,
        max_length,
        device,
    ):
        """
        :return prob: prob -> 0.0 means unanswerable, prob -> 1.0 means answerable
        """
        input_text = question + ' ' + u_tokenizer.sep_token + ' ' + context
        inputs = u_tokenizer(input_text, max_length=max_length, truncation=True, return_tensors="pt")
        inputs = inputs.to(device)
        logits = u_model(**inputs).logits
        logits = logits.squeeze(-1)
        prob = torch.sigmoid(logits).item()
        return prob