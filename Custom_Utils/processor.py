import nltk
import collections
import pytrec_eval
from tqdm import tqdm
import pandas as pd
import numpy as np

def tokenizer(sentences_list: list, remove_empty: bool = True):
    if remove_empty:
        sentences = [s for s in sentences_list if len(s) > 0]
    else:
        sentences = sentences_list
    nl = [nltk.word_tokenize(' '.join(sentence).lower()) for sentence in sentences]
    closed_vocabulary = []
    words_count = collections.Counter([token for sentence in nl for token in sentence])
    for word, count in words_count.items():
        if count >= 6:
            closed_vocabulary.append(word)
    return nl, closed_vocabulary


def adding_unknown(tokenized_sentences: list, vocabulary: list, unknown_token: str = "<unknown>"):
    vocabulary = set(vocabulary)
    new_tokenized_sentences = []
    for sentence in tokenized_sentences:
        new_sentence = []
        for token in sentence:
            if token in vocabulary:
                new_sentence.append(token)
            else:
                new_sentence.append(unknown_token)
        new_tokenized_sentences.append(new_sentence)
    return new_tokenized_sentences


def load_birkbeck_file(file_loc):
    with open(file_loc) as f:
        lines = f.readlines()
    test_lines = []
    incorrect_words = [line.split()[0] for line in lines if line[0] != '$']
    correct_words = [line.split()[1] for line in lines if line[0] != '$']
    test_lines = [' '.join(line.split()[2:]).replace('\n', '') for line in lines if line[0] != '$']
    fill_in_char = '*'
    previous_tokens = [line.split()[:line.split().index(fill_in_char)] for line in test_lines]
    test_df = pd.DataFrame()
    test_df['fill-in-word'] = correct_words
    test_df['previous-tokens'] = previous_tokens
    test_df['test-seq'] = test_lines
    return test_df

def evaluate_models(query,result_eval,suggestions,testing_data,model_testing_data,n):
    for fill_in_word, test_previous_tokens, suggestion in tqdm(
            zip(testing_data['fill-in-word'], model_testing_data, suggestions),
            total=len(model_testing_data)):
        query[f"{' '.join(test_previous_tokens)} *"] = {fill_in_word: 1}
        result_eval[f"{' '.join(test_previous_tokens)} *"] = {}
        if len(suggestion) >= n:
            words_1 = np.array([w[0] for w in suggestion[1]])
            words_5 = np.array([w[0] for w in suggestion[5]])
            words_10 = np.array([w[0] for w in suggestion[10]])

            result_eval[f"{' '.join(test_previous_tokens)} *"] = {word: 1 for word in words_1}

            unique_words_5 = np.setdiff1d(words_5, words_1)
            result_eval[f"{' '.join(test_previous_tokens)} *"].update({word: 1/5 for word in unique_words_5})

            unique_words_10 = np.setdiff1d(words_10, np.union1d(words_1, words_5))
            result_eval[f"{' '.join(test_previous_tokens)} *"].update({word: 1/10 for word in unique_words_10})
        else:
            print(f"Error: the length of the suggestion list is less than {n}")
            return
    print(f'S@k average of {n}-gram model')
    print('*' * 30)
    evaluator = pytrec_eval.RelevanceEvaluator(query, {'success'})
    eval = evaluator.evaluate(result_eval)
    for measure in sorted(list(eval[list(eval.keys())[0]].keys())):
        avg = pytrec_eval.compute_aggregated_measure(measure,[query_measures[measure] for query_measures in eval.values()])
        print(f"{measure} average: {avg}")
    print('=' * 50)
