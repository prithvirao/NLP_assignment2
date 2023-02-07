import concurrent

from joblib import Memory
memory = Memory("./cache", verbose=0)
import operator
import os
import pickle



def count_n_grams(data, n, first_token="<s>", last_token="<e>"):
    n_grams = {}
    for sentence in data:
        sentence = [first_token] * n + sentence + [last_token]
        sentence = tuple(sentence)
        m = len(sentence) if n == 1 else len(sentence) - 1
        for i in range(m):
            n_gram = sentence[i:i + n]

            if n_gram in n_grams.keys():
                n_grams[n_gram] += 1
            else:
                n_grams[n_gram] = 1
    return n_grams


def prob_for_single_word(word, prev_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary_size, k=1.0):

    if len(prev_n_gram) != len(list(nminus1_gram_counts.keys())[0]):
        prev_n_gram = ['<s>'] * (len(list(nminus1_gram_counts.keys())[0]) - len(prev_n_gram)) + \
                      list(prev_n_gram)

    prev_n_gram = tuple(prev_n_gram)
    previous_n_gram_count = nminus1_gram_counts[prev_n_gram] if prev_n_gram in nminus1_gram_counts else 0
    denom = previous_n_gram_count + k * vocabulary_size
    n_gram = prev_n_gram + (word,)
    nplus1_gram_count = n_gram_counts[n_gram] if n_gram in n_gram_counts else 0
    num = nplus1_gram_count + k
    prob = num / denom
    return prob

def helper_prob_for_single_word(args_):
    return prob_for_single_word(args_[0], args_[1], args_[2], args_[3], args_[4], k=args_[5])


def probs(previous_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary, k=1.0, parallelize=False):
    previous_n_gram = tuple(previous_n_gram)
    vocabulary = vocabulary + ["<e>", "<unk>"]
    vocabulary_size = len(vocabulary)
    probabilities = {}
    if not parallelize:
        for word in vocabulary:
            probability = prob_for_single_word(word, previous_n_gram,
                                               nminus1_gram_counts, n_gram_counts,
                                               vocabulary_size, k=k)
            probabilities[word] = probability
    else:
        results = []
        args_ = [(word, previous_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary_size, k) for word in vocabulary]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results.append(executor.map(helper_prob_for_single_word, args_))

        for res, word in zip(results, vocabulary):
            probabilities[word] = res

        print(probabilities)
    return probabilities


def top_n_selection(word_probability_tuple: list, top_n: int = 1):
    d_nth = word_probability_tuple[top_n-1][1]
    ix = top_n
    ix_top_n = top_n
    for i in range(ix, len(word_probability_tuple)):
        if word_probability_tuple[i][1] != d_nth:
            break
        else:
            ix_top_n = i+1

    top_n_words = word_probability_tuple[:ix_top_n]
    return top_n_words


def auto_complete(previous_tokens, nminus1_gram_counts, n_gram_counts, vocabulary, k=1.0):
    n = len(list(n_gram_counts.keys())[0])
    previous_n_gram = previous_tokens[-(n-1):]
    probabilities = probs(previous_n_gram, nminus1_gram_counts, n_gram_counts, vocabulary, k=k)
    sorted_pf = [(k, v) for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)]
    top_1 = [sorted_pf[0]]
    top_5 = sorted_pf[:5]
    top_10 = sorted_pf[:10]

    return {1: top_1, 5: top_5, 10: top_10}


@memory.cache
def unigram_auto_complete(n_gram_counts, k=0.0):
    probabilities = {}
    total_counts = sum(n_gram_counts.values())
    for w, f in n_gram_counts.items():
        probabilities[w] = (f + k)/(total_counts + k)
    sorted_pf = [(k[0], v) for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)]
    top_1_results = [sorted_pf[0]]
    top_5_results = sorted_pf[:5]
    top_10_results = sorted_pf[:10]
    return {1: top_1_results, 5: top_5_results, 10: top_10_results}


class SimpleNGram:
    def __init__(self, n: int, model_loc: str, vocabulary):
        self.n_gram_counts = {}
        self.n = n
        self.n_gram_counts['n'] = self.n
        self.model_loc = model_loc
        self.vocabulary = vocabulary

    def fit(self, data):
        if self.n > 1:
            n_counts = count_n_grams(data, self.n)
            n_minus_one_counts = count_n_grams(data, self.n-1)
            self.n_gram_counts['n-1_counts'] = n_minus_one_counts
            self.n_gram_counts['n_counts'] = n_counts
        else:
            n_counts = count_n_grams(data, self.n)
            self.n_gram_counts['n_counts'] = n_counts

    def get_suggestions(self, previous_tokens: list, k: float = 1.0):
        if self.n > 1:
            suggestion = auto_complete(previous_tokens, self.n_gram_counts['n-1_counts'],
                                       self.n_gram_counts['n_counts'], self.vocabulary, k=k)
        else:
            suggestion = unigram_auto_complete(self.n_gram_counts['n_counts'])
        return suggestion