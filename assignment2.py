import concurrent
import os
import pickle
import nltk
from nltk.corpus import brown


from processor import adding_unknown, tokenizer, load_birkbeck_file
from ngram_model import NGramModel


def save_model(model):
    os.makedirs(model.model_loc, exist_ok=True)
    with open(os.path.join(model.model_loc, f'{model.n}-gram-counts.pickle'), 'wb') as handle:
        pickle.dump(model.n_gram_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(model):
    with open(os.path.join(model.model_loc, f'{model.n}-gram-counts.pickle'), 'rb') as handle:
        model.n_gram_counts = pickle.load(handle)


if __name__ == "__main__":
    nltk.download('punkt')
    nltk.download('brown')
    new_list = brown.sents()
    min_freq = 6
    tokenized_sent, vocabulary = tokenizer(new_list)
    final_train = adding_unknown(tokenized_sent, vocabulary)

    ns = [1, 2, 3, 5, 10]
    #model_loc = 'models'

    for n in ns:
        model = NGramModel(n=n, model_loc='models', vocabulary=vocabulary)
        model.fit(final_train)
        save_model(model)

    previous_tokens = ["the", "jury"]

    for n in ns:
        model = NGramModel(n=n, model_loc='models', vocabulary=vocabulary)
        load_model(model)
        print(f"{n}-gram model prediction: {model.get_suggestions(previous_tokens)}")
        

    test_df = load_birkbeck_file(file_loc='data/APPLING1DAT.643')
    tokenized_sent, a = tokenizer(test_df['previous-tokens'].values.tolist(), remove_empty=False)
    final_test = adding_unknown(tokenized_sent, vocabulary)
    test_df['final-test'] = final_test

    queries = [{} for _ in ns]
    results_eval = [{} for _ in ns]

    for idx, n in enumerate(ns):
        model = NGramModel(n=n, model_loc='models', vocabulary=vocabulary)
        load_model(model)

        argument_list = final_test
        suggestions = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for result in executor.map(model.get_suggestions, argument_list):
                suggestions.append(result)

        query = queries[idx]
        result_eval = results_eval[idx]
        evaluate_models(queries[idx],results_eval[idx],suggestions,test_df,final_test,n)
        
        