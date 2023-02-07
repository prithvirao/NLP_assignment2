import concurrent
import os
import pickle
import nltk
from nltk.corpus import brown


from custom_utils.processor import adding_unknown, tokenizer, load_birkbeck_file
from custom_utils.ngram_model import NGramModel


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
    tokenized_sent, vocabulary = tokenizer(new_list)
    model_train_data = adding_unknown(tokenized_sent, vocabulary)

    ns = [1, 2, 3, 5, 10]
    new_list = brown.sents()
	tokenized_sent, vocabulary = tokenizer(new_list)
	model_train_data = adding_unknown(tokenized_sent, vocabulary)
	test_sentence_tokens=["The", "morning"]
	ns = [1, 2, 3, 5, 10]
	for n in ns:
    	model = SimpleNGram(n=n, model_loc='models', vocabulary=vocabulary)
    	model.fit(model_train_data)
    	save_model(model)
    	print(f"{n}-gram model prediction: {model.get_suggestions(test_sentence_tokens)}")
    	print()
    testing_data = load_birkbeck_file(file_loc='Data/APPLING1DAT.643')
	tokenized_sent, a = tokenizer(testing_data['previous-tokens'].values.tolist(), remove_empty=False)
	model_testing_data = adding_unknown(tokenized_sent, vocabulary)
	testing_data['final-test'] = model_testing_data
	
	print(model_testing_data)
	queries = [{} for _ in ns]
	results_eval = [{} for _ in ns]

	for i, n in enumerate(ns):
    	model = SimpleNGram(n=n, model_loc='models', vocabulary=vocabulary)
    	load_model(model)
    	argument_list = model_testing_data
    	suggestions = []
    	with concurrent.futures.ProcessPoolExecutor() as executor:
        	for result in executor.map(model.get_suggestions, argument_list):
            	suggestions.append(result)

    	query = queries[i]
    	result_eval = results_eval[i]
    	evaluate_models(queries[i],results_eval[i],suggestions,testing_data,model_testing_data,n)