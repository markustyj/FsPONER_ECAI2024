## create the few-shot list for each software requirement based on random selection, sentence embedding, and TF-IDF selection
## @ yongjian.tang@tum.de
import random
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def load_csv(file_path):  
    """ load csv file line by line, in list format  -> [ [first-row requirement, first-row category, .. ], [second-row requirement, second-row category, .. ], [], ... ] """
    data = []  
    with open(file_path, 'r') as file:  
        csv_reader = csv.reader(file)  
        for row in csv_reader:  
            data.append(row)  
    return data[1:]  # do not need the first row of title ['RequirementText', 'class']



def load_specific_dataset(dataset):
    """load the specific dataset using load_csv() method
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    output
    loaded_data: the loaded csv file in format [ [first-row requirement, first-row multi-class category, first-row binary-class category,.. ], [second-row requirement, second-row multi-class category, .. ], [], ... ]
    """
    if dataset == "nfr": 
        loaded_data_train = load_csv('./processed_nfr_so_train.csv')
        loaded_data_val = load_csv('./processed_nfr_so_val.csv')  # '/Users/yongjiantang/Desktop/tang/code/re_genai/data/processed_nfr_so_val.csv'
        loaded_data_test = load_csv('./processed_nfr_so_test.csv')

    elif dataset == "promise":
        loaded_data_train = load_csv('./processed_promise_train.csv')
        loaded_data_val = load_csv('./processed_promise_val.csv')
        loaded_data_test = load_csv('./processed_promise_test.csv')
    else:
        raise ValueError("dataset name must be 'nfr' or 'promise'.")

    return loaded_data_train, loaded_data_val, loaded_data_test



def get_random_few_shot_list(dataset):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    loaded_data: the loaded csv file in format [ [first-row requirement, first-row category, .. ], [second-row requirement, second-row category, .. ], [], ... ]
    output
    few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    """
    loaded_data_train, loaded_data_val, loaded_data_test = load_specific_dataset(dataset)
    loaded_data_train_val = loaded_data_train + loaded_data_val
    num_k_closest_examples = 160

    few_shot_list = []
    k_closest_examples = []

    # create the list of k closest few-shot examples
    values = list(range(0, len(loaded_data_train_val)))
    random.shuffle(values)
    for i in range(num_k_closest_examples):
        index = values.pop()
        k_closest_examples.append(loaded_data_train_val[index])

    # repeat and reuse this list of k closest few-shot examples for each software requirement, i.e. same list for each requirement
    for i in range(len(loaded_data_test)):
        few_shot_list.append(k_closest_examples)
    
    file_path = './few_shot_list/' + dataset + '_random.npy'
    np.save(file_path, np.array(few_shot_list))

    #return few_shot_list



def get_embedding_few_shot_list(dataset):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    loaded_data: the loaded csv file in format [ [first-row requirement, first-row category, .. ], [second-row requirement, second-row category, .. ], [], ... ]
    output
    few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    """
    few_shot_list = []

    loaded_data_train, loaded_data_val, loaded_data_test = load_specific_dataset(dataset)
    loaded_data_train_val = loaded_data_train + loaded_data_val
    num_k_closest_examples = 160

    # collect all textual requirements
    loaded_data_train_val_requirement = [sublist[0] for sublist in loaded_data_train_val] 
    loaded_data_test_requirement = [sublist[0] for sublist in loaded_data_test] 
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    embedding_train_val = model.encode(loaded_data_train_val_requirement)
    embedding_test = model.encode(loaded_data_test_requirement)

    # for each single test_requirement sample, rank the top relevant requirements in training dataset and obtain their index 
    index = find_most_relevant_vectors(embedding_test, embedding_train_val, num_k_closest_examples)

    for i in range(len(loaded_data_test)):
        k_closest_examples = []
        for j in range(num_k_closest_examples):
            k_closest_examples.append(loaded_data_train_val[index[i][j]])
        few_shot_list.append(k_closest_examples)

    file_path = './few_shot_list/' + dataset + '_embedding.npy'
    np.save(file_path, np.array(few_shot_list))


def find_most_relevant_vectors(embedding_test, embedding_train_val, num_k_closest_examples):  
    """for each single test_requirement sample, rank the top relevant requirements in training dataset and obtain their index 
    embedding_test: embedding matrix, the number of test samples x embedding dimensions
    embedding_train_val: embedding matrix, the number of test samples x embedding dimensions
    output
    index: 2-dimensional list,  [ [the indexes of the most relevant training samples for the first requirement in test dataset], [the indexes of the most relevant training samples for requirement 2 in test dataset], ...]
    """
    index = []
    for i, vec1 in enumerate(embedding_test):  
        distances = []  
        for j, vec2 in enumerate(embedding_train_val):  
            distance = np.linalg.norm(vec1 - vec2)  # Euclidean distance  
            distances.append((distance, i, j))  
        distances.sort()  # Sort distances in ascending order  

        index_i_requirement = [ distances[k][2] for k in range(num_k_closest_examples)]   # iterate over the top K indices of the most relevant vectors 
        index.append(index_i_requirement)

    return index  


def get_tfidf_few_shot_list(dataset):
    """
    input
    dataset: the name of the processed datasets -> "nfr" or "promise"
    loaded_data: the loaded csv file in format [ [first-row requirement, first-row category, .. ], [second-row requirement, second-row category, .. ], [], ... ]
    output
    few_shot_list: three-dimensional list --> each requirement sentence, *times* its k closest few-shot examples, *times* [the few-shot examples, the multi-class, the binary-class] 
    """
    few_shot_list = []

    loaded_data_train, loaded_data_val, loaded_data_test = load_specific_dataset(dataset)
    loaded_data_train_val = loaded_data_train + loaded_data_val
    num_k_closest_examples = 160

    # collect all textual requirements
    loaded_data_train_val_requirement = [sublist[0] for sublist in loaded_data_train_val] 
    loaded_data_test_requirement = [sublist[0] for sublist in loaded_data_test] 

    #transform to tfidf vectors
    v = TfidfVectorizer(min_df=0.001, max_df=0.8, analyzer='word')
    fitted_corpus = v.fit(loaded_data_train_val_requirement)
    tfidf_train_val = v.transform(loaded_data_train_val_requirement)
    tfidf_test = v.transform(loaded_data_test_requirement)
    # transform sparse matrix to numpy array
    tfidf_train_val = np.float32(  tfidf_train_val.toarray() )   
    tfidf_test = np.float32(  tfidf_test.toarray() )

    # for each single test_requirement sample, rank the top relevant requirements in training dataset and obtain their index 
    index = find_most_relevant_vectors(tfidf_test, tfidf_train_val, num_k_closest_examples)
    
    for i in range(len(loaded_data_test)):
        k_closest_examples = []
        for j in range(num_k_closest_examples):
            k_closest_examples.append(loaded_data_train_val[index[i][j]])
        few_shot_list.append(k_closest_examples)

    file_path = './few_shot_list/' + dataset + '_tfidf.npy'
    np.save(file_path, np.array(few_shot_list))

    #return few_shot_list

