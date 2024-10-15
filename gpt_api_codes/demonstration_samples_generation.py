"""
Generate the original task description 
and demonstration examples for the prompt in three different ways "random", "sentence" and "entity"
following the GPT-NER paper.

"""
__author__ = 'yongjian.tang'

import random
import numpy as np
import faiss
from simcse import SimCSE
from sklearn.feature_extraction.text import TfidfVectorizer
#from gpt_ner_api_codes.api_gpt3 import *
from gpt_ner_api_codes.api_gpt4 import *


def get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset):
    """The old instruction and prompt for running GPT API 3.5"""
    if dataset == "film":
        instruction = "You are an excellent linguist in the domain of thin film head technology. The task is to label the entities in the given sentence. " \
            "The optional entities include Component, Location, Function, EnergyFlow (Energy Flow), Attribution, Material, Effect, System, ScientificConcept (Scientific Concept), "\
            "Shape,  Value, InfoFlow (Information Flow), Consequence, PhysicsFlow (Physics Flow), Measure, State, AnnotatorNotes. \n"\
        #    "Below are some examples.\n\n"
    elif dataset == "fabner":
        instruction = "You are an excellent linguist in manufacturing domain. The task is to label the entities in the given sentence. " \
            "The optional entities include CONPRI (Concept/principles), MATE (Material), MANP (Manufacturing process), PRO (Mechanical properties), PARA (Process parameters), CHAR (Process characterization), "\
            "MACEQ (Machine/equipment), APPL (Application), FEAT (Engineering features), ENAT (Enabling technology), MANS (Manufacturing standards), BIOP (Biomedical). \n"\
        #    "Below are some examples.\n\n"  
    elif dataset == "assembly":
        instruction = "You are an excellent linguist in assembly instructions. The task is to label the entities in the given sentence. " \
            "The optional entities include PART (parts), RPOS (relative positions), OPER (operations), TOOL (tools), ID (identification numbers), QTY (quantity), "\
            "DIM (dimensions), WGT (weights), PROP (general properties). \n"\
        #    "Below are some examples.\n\n"    
    else:
        raise  ValueError ("You can only select dataset between 'film', 'fabner', 'assembly'." )   

    if num_examples == 0:
        instruction = instruction + "The extracted entities should have the following format. \nExtracted words: entity types \nExtracted words: entity types \n etc.\n\n"
    elif num_examples == 1:
        instruction = instruction + "Below is an example.\n"
    else:
        instruction = instruction + "Below are some examples.\n\n"        

    # for each input sentence, iterate over the index of all few shot examples
    for i in range( num_examples ):      
        sen_element = fewshot_sen_one_sentence[i]
        dem_element = fewshot_dem_one_sentence[i]
        # instruction = instruction + "Input{i}:{sen_element}".format(i=i+1, sen_element=sen_element) \
        #       + "Output{i}:\n{dem_element}".format(i = i+1, dem_element = dem_element) 
        instruction = instruction + "Example {i}:\n".format(i=i+1) + "Sentence: {sen_element}".format(sen_element=sen_element) \
              + "Extracted entities of the above sentence:\n{dem_element}".format(i = i+1, dem_element = dem_element) 
        
    return instruction 


#####################################
# The function below is not used in the code, completely same as get_prompt_with_demonstration_examples() above
def get_prompt_with_demonstration_examples_gpt4( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset):
    """The old instruction and prompt for running GPT API 3.5"""
    if dataset == "film":
        instruction = "Extract the entity types in the input sentence." \
            "The optional entity types include Component, Location, Function, EnergyFlow (Energy Flow), Attribution, Material, Effect, System, ScientificConcept (Scientific Concept), "\
            "Shape, Value, InfoFlow (Information Flow), Consequence, PhysicsFlow (Physics Flow), Measure, State, AnnotatorNotes. "\
            "Below are some examples.\n\n"
        
    elif dataset == "fabner":
        instruction = "Extract the entity types in the input sentence." \
            "The optional entity types include CONPRI (Concept/principles), MATE (Material), MANP (Manufacturing process), PRO (Mechanical properties), PARA (Process parameters), CHAR (Process characterization), "\
            "MACEQ (Machine/equipment), APPL (Application), FEAT (Engineering features), ENAT (Enabling technology), MANS (Manufacturing standards), BIOP (Biomedical). "\
            "Below are some examples.\n\n"  
        
    elif dataset == "assembly":
        instruction = "Extract the entity types in the input sentence." \
            "The optional entity types include PART (parts), RPOS (relative positions), OPER (operations), TOOL (tools), ID (identification numbers), QTY (quantity), "\
            "DIM (dimensions), WGT (weights), PROP (general properties). "\
            "Below are some examples.\n\n"    
        
    else:
        raise  ValueError ("You can only select dataset between 'film', 'fabner', 'assembly'." )   

    # for each input sentence, iterate over the index of all few shot examples
    for i in range( num_examples ):      
        sen_element = fewshot_sen_one_sentence[i]
        dem_element = fewshot_dem_one_sentence[i]
        # instruction = instruction + "Input{i}:{sen_element}".format(i=i+1, sen_element=sen_element) \
        #       + "Output{i}:\n{dem_element}".format(i = i+1, dem_element = dem_element) 
        instruction = instruction + "Example {i}:\n".format(i=i+1) + "Sentence: {sen_element}".format(sen_element=sen_element) \
              + "Extracted entities of the above sentence:\n{dem_element}".format(i = i+1, dem_element = dem_element) 
        
    return instruction 


def get_combined_dataset(sentences, demonstrations, entity_list ):
    """ Based on the input, combine them to a large numpy array for stratification
    """
    combi_dataset = []
    for i, sentence in enumerate(sentences):
        for entity in entity_list[i]:
            if entity == "":
                continue
            else:
                combi_dataset.append( [sentence, demonstrations[i], entity])
    combi_dataset = np.asarray(combi_dataset)
    x = combi_dataset[:,:2]
    y = combi_dataset[:,2]
    #print("combi_dataset", combi_dataset[0][2], "\nx below\n", x, "y below\n", y)
    return x, y


def get_entity_from_demonstrations(demonstrations):
    """get the entity from the dataset of few-shot examples, and list all entities for each original sentence
    return: entity_list
    the first dimension is len(sentences), the second dimension is len(entities in this sentence)"""
    entity_list = []
    for i, sin_response in enumerate(demonstrations):
        sin_sentence_entity_list = []
        sin_item_entity_list = sin_response.split("\n\n")[0] # just to prevent \n split the \n\n
        sin_item_entity_list = sin_item_entity_list.split("\n") 
        for sin_item_entity in sin_item_entity_list:
            #print(sin_item_entity)
            if len(sin_item_entity.split(": ")) == 1:
                sin_sentence_entity_list.append("")
            elif len(sin_item_entity.split(": ")) > 2:
                sin_entity = sin_item_entity.split(": ")[-1] 
                sin_sentence_entity_list.append(sin_entity)
            else:
                sin_item, sin_entity = sin_item_entity.split(": ")
                sin_sentence_entity_list.append(sin_entity)
        entity_list.append(sin_sentence_entity_list)    
    
    return entity_list 


def get_prompt( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset):
    
    if dataset == "film":
        instruction = "Instruction:\n"\
            "You are a named entity extractor in the domain of thin film head technology. The task is to label the entity types in the given sentence.\n"\
            "The optional entities include Component, Location, Function, EnergyFlow (Energy Flow), Attribution, Material, Effect, System, ScientificConcept (Scientific Concept), "\
            "Shape,  Value, InfoFlow (Information Flow), Consequence, PhysicsFlow (Physics Flow), Measure, State, AnnotatorNotes. "\
            "\nBelow are some examples:\n\n"
        # instruction = "Instruction:\n"\
        #     "You are a named entity extractor in the domain of thin film head technology. The task is to label the entity types in the given sentence.\n"\
        #     "The optional entities include: "\
        #     "Component, Location, Function, Attribution, Material, Effect, System, Shape,  Value, Consequence, Measure, State, "\
        #     "EnergyFlow: Energy Flow, "\
        #     "ScientificConcept: Scientific Concept, "\
        #     "InfoFlow: Information Flow, "\
        #     "PhysicsFlow: Physics Flow, "\
        #     "AnnotatorNotes: Annotation Notes. "\
        #     "\nBelow are some examples:\n\n"
        
    elif dataset == "fabner":
        instruction = "Instruction:\n"\
            "You are a named entity extractor in manufacturing domain. The task is to label the entity types in the given sentence.\n" \
            "The optional entities include CONPRI (Concept/principles), MATE (Material), MANP (Manufacturing process), PRO (Mechanical properties), PARA (Process parameters), CHAR (Process characterization), "\
            "MACEQ (Machine/equipment), APPL (Application), FEAT (Engineering features), ENAT (Enabling technology), MANS (Manufacturing standards), BIOP (Biomedical). "\
            "Below are some examples.\n\n"       
    elif dataset == "assembly":
        instruction = "Instruction:\n"\
            "You are a named entity extractor in assemly instructions. The task is to label the entity types in the given sentence.\n" \
            "The optional entities include PART (parts), RPOS (relative positions), OPER (operations), TOOL (tools), ID (identification numbers), QTY (quantity), "\
            "DIM (dimensions), WGT (weights), PROP (general properties). "\
            "Below are some examples.\n\n"    
    else:
        raise  ValueError ("You can only select dataset between 'film', 'fabner', 'assembly'." )   

    # for each input sentence, iterate over the index of all few shot examples
    for i in range( num_examples-15 ):      
        sen_element = fewshot_sen_one_sentence[i]
        dem_element = fewshot_dem_one_sentence[i]
        # instruction = instruction + "Input{i}:{sen_element}".format(i=i+1, sen_element=sen_element) \
        #       + "Output{i}:\n{dem_element}".format(i = i+1, dem_element = dem_element) 
        instruction = instruction + "Example {i}:\n".format(i=i+1) + "Sentence: {sen_element}".format(sen_element=sen_element) 
        #instruction = instruction + "Sentence {i}:\n{sen_element}".format(i=i+1, sen_element=sen_element)         
        instruction = instruction + "Extracted entities of the above sentence:\n{dem_element}".format(i = i+1, dem_element = dem_element)
        #instruction = instruction +  "Response {i}:\n{dem_element}".format( i = i+1,  dem_element = dem_element)
        
    return instruction 



# Above are the currently used codes 
#################################################################################################
# Below is some draft of codes
# the 4 functions, i.e., random_demonstration(), sentence_demonstration(), tfidf_demonstration(), 
# and entity_demonstration() are replaced by get_fewshot_sen_dem_random(), get_fewshot_sen_dem_sentembedding() ...

def random_demonstration(sentences_dev, demonstrations_dev, num_examples, stratify):
    sen = []
    dem = [] 
    
    if stratify :
        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        
        unique, counts = np.unique(development_entity, return_counts=True)
        index = []
        specific_entity_sentence_and_demon = []
        few_shot_examples = []

        ########### Handle few shot learning dataset
        ## get numpy array (type of entity, its corresponding sentence_and_demon_train)
        for i, entity in enumerate(unique):
            # get the index of a certain entity in a numpy array
            index.append( np.where(development_entity == entity) ) 
            # dimension 1 is number of unique entities, dimension 2 is all corresponding sentence_and_demon_train of an entity
            specific_entity_sentence_and_demon.append(sentence_and_demon[index[i]])
        
        iteration_of_all_entities = int( num_examples/len(unique) )  # how many total round of entites is needed for demon_size
        for entity_index in range( len(unique) ):
            # a list of random int numbers to replace iteration_of_all_entities in order
            sample_index_selected = random.sample( range(0, counts[entity_index]), iteration_of_all_entities) 
            for iteration in range(iteration_of_all_entities):
                few_shot_examples.append(specific_entity_sentence_and_demon[entity_index][sample_index_selected[iteration]])

        print( "Duplicated few-shot examples: ", len(few_shot_examples))   
        few_shot_examples = np.asarray(few_shot_examples)
        few_shot_examples = np.unique(few_shot_examples, axis = 0) # eliminate duplicated rows
        
        sen = list(few_shot_examples[:,0] )
        dem = list(few_shot_examples[:,1] )#np.unique(few_shot_examples[:,1])
        print( "length of stratified_sentences_train: ", len(sen) )
    else:
        # randomly sample index for pop out demonstration examples
        values = list(range(0, len(sentences_dev)))
        #random.seed(44)
        random.shuffle(values)
        for i in range(num_examples):
            index = values.pop()
            sen.append(sentences_dev[index])
            dem.append(demonstrations_dev[index])

    return sen, dem, len(sen)


def tfidf_demonstration(input_sen, sentences_dev, demonstrations_dev, model, num_examples, num_tfidf_words = 5, stratify = True):
    v = TfidfVectorizer(min_df=0.001, max_df=0.8, analyzer='word')
    fitted_corpus = v.fit(sentences_dev)
    all_words = v.get_feature_names_out()
    xb = np.float32(  v.transform(sentences_dev).toarray() )   
    xq = np.float32(  v.transform([input_sen]).toarray() )  

    ## pick top N words with highest tfidf scores
    N = num_tfidf_words
    top_index_b = np.argsort( -xb, axis=1)
    xb_tfidf = all_words[top_index_b[:,:N]]
    xb_tfidf_joined = []
    for i, tfidf_words in enumerate(xb_tfidf):
        xb_tfidf_joined.append( " ".join(tfidf_words) ) 
    #print(xb_tfidf_joined)

    top_index_q = np.argsort( -xq, axis=1)
    xq_tfidf = all_words[top_index_q[:,:N]]
    xq_tfidf_joined = []
    for i, tfidf_words in enumerate(xq_tfidf):
        xq_tfidf_joined.append( " ".join(tfidf_words) ) 
    #print(xq_tfidf_joined)

    similarity_matrix = model.similarity(xq_tfidf_joined, xb_tfidf_joined)

    ### pick top k number of examples as few-shots
    k = num_examples
    similarity_index = np.argsort(-similarity_matrix, axis = 1)
    similarity_index_selected = similarity_index[0, :k]
    sen = []
    dem = [] 
    for i in range(num_examples):
        sen.append(sentences_dev[similarity_index_selected[i]])
        dem.append(demonstrations_dev[similarity_index_selected[i]])

    return sen, dem, len(sen)


def sentence_demonstration(sentences_dev, demonstrations_dev, input_sen, num_examples, faiss_index_list, model, stratify):
    sen = []
    dem = []  

    if stratify:
        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        unique, counts = np.unique(development_entity, return_counts=True)
        index = []
        specific_entity_sentence_and_demon = []

        ########### Handle few shot learning dataset
        ## get numpy array (type of entity, its corresponding sentence_and_demon_train)
        for i, entity in enumerate(unique):
            # get the index of a certain entity in a numpy array
            index.append( np.where(development_entity == entity) ) 
            # dimension 1 is number of unique entities, dimension 2 is all corresponding sentence_and_demon_train of an entity
            specific_entity_sentence_and_demon.append(sentence_and_demon[index[i]])
        
        iteration_of_all_entities = int( num_examples/len(unique) )  # how many total number of samples of one entity is needed for demon_size
        for entity_index in range( len(unique) ):
            # specific_entity_sentence_and_demon[entity_index] is a list of sentences of a one entity in development set
            specific_entity_sentence = [ specific_entity_sentence_and_demon[entity_index][i][0] for i in range(len(specific_entity_sentence_and_demon[entity_index]))]
            specific_entity_demon = [ specific_entity_sentence_and_demon[entity_index][i][1] for i in range(len(specific_entity_sentence_and_demon[entity_index]))]    
            # train faiss and add sentences_dev_encoded to the dataset
            index = faiss_index_list[entity_index]
            
            # input_sen is just one sentence
            input_sen_encoded = model.encode(input_sen)
            input_sen_encoded = np.asarray( input_sen_encoded )
            
            # D is the distance matrix and I is the index matrix of shape ( len of input_sentence, k_nearest) -> (1, num_examples)
            # iteration_of_all_entities is k, k_nearest 
            D, I = index.search(input_sen_encoded, iteration_of_all_entities)   
            for i in range(iteration_of_all_entities):
                index = I[0][i]
                sen.append(specific_entity_sentence[index])
                dem.append(specific_entity_demon[index])
        
        print( "Duplicated few-shot examples: ", len(sen),len(dem))   
        combined_sen_dem = np.asarray([sen,dem])
        combined_sen_dem = np.unique(combined_sen_dem, axis = 1)  
        sen = list(combined_sen_dem[0] )
        dem = list(combined_sen_dem[1] ) # np.unique(few_shot_examples[:,1])
        print( "length of stratified_sentences_train: ", len(sen), len(dem) )

    else: # not stratify the few-shot examples from the 300 development dataset
        # input_sen is just one sentence
        input_sen_encoded = model.encode(input_sen)
        input_sen_encoded = np.asarray( input_sen_encoded ).reshape(1,input_sen_encoded.shape[0])
        # D is the distance matrix and I is the index matrix of shape ( len of input_sentence, k_nearest) -> (1, num_examples)
        index = faiss_index_list[0]
        D, I = index.search(input_sen_encoded, num_examples)
        #print(I)
        for i in range(num_examples):
            index = I[0][i]
            sen.append(sentences_dev[index])
            dem.append(demonstrations_dev[index])
    return sen, dem, len(sen)



def entity_demonstration(response, sentences_dev, demonstrations_dev, num_examples, model, entity_list_dev, stratify):
    sen = []
    dem = [] 
    similarities_list = []
    for i in range(len(demonstrations_dev)):
        similarities = np.asarray(  model.similarity(response, entity_list_dev[i])  )
        if len(similarities.shape):
            similarities = np.asarray(similarities).max(axis=0).mean()
        else:
            similarities = np.asarray(similarities).max(axis=1).mean()

        similarities_list.append([i,similarities])
    sorted_similarities_list = sorted(similarities_list, key=lambda x:x[1], reverse=True)

    for i in range(num_examples):
        index = sorted_similarities_list[i][0]
        sen.append(sentences_dev[index])
        dem.append(demonstrations_dev[index])
    return sen, dem, len(sen)


def get_phrase_in_sentence(demonstrations_dev):
    entity_list = []
    for i, sin_response in enumerate(demonstrations_dev):
        sin_sentence_entity_list = []
        sin_item_entity_list = sin_response.split("\n\n")[0] # just to prevent \n split the \n\n
        sin_item_entity_list = sin_item_entity_list.split("\n") 
        for sin_item_entity in sin_item_entity_list:
            #print(sin_item_entity)
            if len(sin_item_entity.split(": ")) == 1:
                sin_sentence_entity_list.append("None")
            elif len(sin_item_entity.split(": ")) > 2:
                sin_item = sin_item_entity.split(": ")[:-1] 
                sin_sentence_entity_list.append(sin_item)    
            else:
                sin_item, sin_entity = sin_item_entity.split(": ")
                sin_sentence_entity_list.append(sin_item)
        entity_list.append(sin_sentence_entity_list)    
    
    return entity_list 


# def fit_bow_tfidf(text_dataset_array):
#     """
#     Fits a BoW model and uses a TF-IDF transformer on it to compute TF-IDF scores.
#     :param text_dataset_array: numpy array of strings, or pandas series. This means you can pass all values of a
#     pandas data frame like this df["column_name"]
#     :return: tuple(word count matrix, TF-IDF matrix, BoW model, TF-IDF transformer)
#     """
#     # Extracting features from text files
#     # max_df = 0.50 means "ignore terms that appear in more than 50% of the documents".
#     # min_df = 0.01 means "ignore terms that appear in less than 1% of the documents".
#     # vectorizer = StemmedTfidfVectorizer(min_df=0.0004, max_df=0.9983, analyzer="word")

#     # count_vectorizer = CountVectorizer(min_df=0.0004, max_df=0.9983, stop_words=ENGLISH_STOP_WORDS)
#     count_vectorizer = CountVectorizer(min_df=0.0004, max_df=0.9983, stop_words=ENGLISH_STOP_WORDS)

#     X_train_counts = count_vectorizer.fit_transform(text_dataset_array)

#     # TF-IDF
#     tfidf_transformer = TfidfTransformer()
#     X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#     return X_train_counts, X_train_tfidf, count_vectorizer, tfidf_transformer