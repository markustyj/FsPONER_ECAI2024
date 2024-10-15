import os
import json
import logging
import time
import random
import numpy as np
import pandas as pd
import warnings
import string
import tiktoken
import faiss
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
nltk.download('stopwords')
nltk.download('punkt')
from simcse import SimCSE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

from gpt_ner_api_codes.demonstration_samples_generation import * # import all three methods "random", "sentence" and "entity"
#from gpt_ner_api_codes.api_gpt3 import *
from gpt_ner_api_codes.api_gpt4 import *
 

logger = logging.getLogger(__name__)

def split_dev_dataset_input(path_original_sentence, path_entities, path_tokens_tags, num_input=2, demon_size=20, stratify = True):
    
    path_original_sentence = os.path.abspath(path_original_sentence)
    path_entities = os.path.abspath(path_entities)
    path_tokens_tags = os.path.abspath(path_tokens_tags)
    
    with open (path_original_sentence, encoding='utf-8') as f:
        sentences = f.readlines()
        #sentences = sentences[:-1]
    with open (path_entities, encoding='utf-8') as f:
        demonstrations = f.read()
        #demonstrations = demonstrations[:-1]
        demonstrations = demonstrations.split("\n\n")
        for i in range(len(demonstrations)):
            demonstrations[i] = demonstrations[i] + "\n\n"
    with open (path_tokens_tags, encoding='utf-8') as f: 
        token_tags = []
        for line in f:
            token_tag = json.loads(line)
            token_tags.append(token_tag)
        token_tags = token_tags[:-1]
    ## get sentences, demonstrations and token_tags    

    if stratify:
        # get entities of each sentence in the few-shot dataset
        entity_list =  get_entity_from_demonstrations(demonstrations) 
        #print(len(sentences), len(demonstrations), len(entity_list) )
        sentences_dev, sentences_input, sentences_train, demonstrations_dev, demonstrations_input_solutions, demonstrations_train = stratify_the_dataset(sentences, demonstrations, entity_list, demon_size, num_input)
    else:
        sentences_dev, sentences_input, demonstrations_dev, demonstrations_input_solutions = train_test_split(sentences, demonstrations, train_size=demon_size, test_size=num_input, random_state=42)
        
    return sentences_dev, sentences_input, sentences_train, demonstrations_dev, demonstrations_input_solutions, demonstrations_train


def stratify_the_dataset(sentences, demonstrations, entity_list, demon_size, num_input):
    # merge the original sentence and their entities to a large numpy array 
    x,y = get_combined_dataset( sentences, demonstrations, entity_list )
    print(np.unique(y,  return_counts=True))
    sentence_and_demon_train, sentence_and_demon_test, entity_train, entity_test = train_test_split(x, y, test_size = 0.05, random_state=42, stratify=y )
    # get the unique entities in a dataset and the number of samples in it                              train_size = 0.8,
    unique, counts = np.unique(entity_train, return_counts=True)
    index = []
    specific_entity_sentence_and_demon = []
    few_shot_examples = []

    ########### Handle few shot learning dataset
    ## get numpy array (type of entity, its corresponding sentence_and_demon_train)
    for i, entity in enumerate(unique):
        # get the index of a certain entity in a numpy array
        index.append( np.where(entity_train == entity) ) 
        # dimension 1 is number of unique entities, dimension 2 is all corresponding sentence_and_demon_train of an entity
        specific_entity_sentence_and_demon.append(sentence_and_demon_train[index[i]])

    ## demon_size = 300 < entity types * the rarest entity type
    if demon_size < len(unique) * counts.min():
        iteration_of_all_entities = int( demon_size/len(unique) ) + 1 # how many total round of entites is needed for demon_size
        for entity_index in range( len(unique) ):
            # a list of random int numbers to replace iteration_of_all_entities in order
            sample_index_selected = random.sample( range(0, counts[entity_index]), iteration_of_all_entities) 
            for iteration in range(iteration_of_all_entities):
                few_shot_examples.append(specific_entity_sentence_and_demon[entity_index][sample_index_selected[iteration]])
    else: ## demon_size = 300 > entity types * the rarest entity type
        for entity_index in range( len(unique) ):
            # a list of random int numbers to replace iteration_of_all_entities in order
            sample_index_selected = random.sample( range(0, counts[entity_index]), counts.min()) 
            for j in range( counts.min() ): # the min samples of the rarest entity
                few_shot_examples.append(specific_entity_sentence_and_demon[entity_index][sample_index_selected[j]])
        while len(few_shot_examples) < demon_size:
            random_index = random.randint(0,len(sentence_and_demon_train)-1)
            few_shot_examples.append(sentence_and_demon_train[random_index])
            #few_shot_examples = np.unique(few_shot_examples)        

    print( "Duplicated few-shot examples: ", len(few_shot_examples))   
    few_shot_examples = np.asarray(few_shot_examples)
    few_shot_examples = np.unique(few_shot_examples, axis = 0) # eliminate duplicated rows
    stratified_sentences_develop = list(few_shot_examples[:,0] )
    stratified_entities_develop = list(few_shot_examples[:,1] )#np.unique(few_shot_examples[:,1])
    print( "length of stratified_sentences_develop: ", len(stratified_sentences_develop) )
    #print( len(stratified_entities_train) ) 
    #print( stratified_entities_train )

    ########### Handle test input dataset, besides few shot learning dataset
    sentence_and_demon_test = np.asarray(sentence_and_demon_test)
    sentence_and_demon_test = np.unique(sentence_and_demon_test, axis = 0)
    stratified_sentences_test = list(sentence_and_demon_test[-num_input:,0]) # pick the last 10 samples to avoid "no entity in this sentence"
    stratified_entities_test = list(sentence_and_demon_test[-num_input:,1] )
    print( "length of stratified_sentences_test: ", len(stratified_sentences_test) )
    #print( len(stratified_entities_test) ) 
    #print( stratified_entities_test )

    ########### Except test data, the rest of data is for training. Handle training dataset, besides few shot learning dataset
    sentence_and_demon_train = np.asarray(sentence_and_demon_train)
    sentence_and_demon_train = np.unique(sentence_and_demon_train, axis = 0)
    stratified_sentences_train = list(sentence_and_demon_train[:,0]) # pick the last 10 samples to avoid "no entity in this sentence"
    stratified_entities_train = list(sentence_and_demon_train[:,1] )
    print( "length of stratified_sentences_train: ", len(stratified_sentences_train) )


    return stratified_sentences_develop, stratified_sentences_test, stratified_sentences_train, stratified_entities_develop, stratified_entities_test, stratified_entities_train


# Above is data preprocessing, stratify ...
################################################################################################################################
################################################################################################################################
# Below is code of generating prompt and instructions, and running them with GPT API
def get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset, num_examples): #, method = "random", dataset = "film", num_examples = 1, stratify = True):
    # The input parameter "film" here is used for prompt instruction of different dataset, e.g. in domain of, different domains
    """ 
    get the gpt response with stored sentences directly, without running get_textual_instruction_with_fewshot_examples()
    We do not need input parameters: method = "random", num_examples = 1, stratify = True
    generate prompts for thin film technology dataset"""

    # get the prompt with few-shot learning examples, before running GPT API

    response = []
    # iterate over the first 30 input sentences to the prompt
    for input_sen_index, input_sen in enumerate(sentences_input):
        
        # use the selected few-shot examples for each input sentence
        fewshot_sen_one_sentence = all_fewshot_sen[input_sen_index]
        fewshot_dem_one_sentence = all_fewshot_dem[input_sen_index]

        # update the num_examples, if there is duplicated items
        # num_examples_new is used in except in case that we exceed the maximum tokens
        #num_examples = len(fewshot_sen_one_sentence)
        num_examples_new = num_examples

        instruction = get_prompt_with_demonstration_examples_gpt4( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset)
        instruction = instruction + "Extract all the entities from the input sentence below:\n" + "Input sentence: {input}".format(input = input_sen) 
        instruction = instruction + "\nExtracted entities of the input sentence:\n"
        prompt_text = instruction
        #logger.info(prompt_text)
        
        ### get the maximun number of few-shot examples
        enc = tiktoken.encoding_for_model("text-davinci-003")
        for i in range(num_examples):
            num_tokens = len( enc.encode(prompt_text) )
            if num_tokens > 4096 - 300:  ## consider 300 token length for output
                num_examples_new = num_examples_new - 1
                instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples_new, dataset)
                instruction = instruction + "Extract all the entities from the input sentence below:\n" + "Input sentence: {input}".format(input = input_sen) + "\nExtracted entities of the input sentence:\n"
                prompt_text = instruction
                #print("exceed maximun token length, -{i}", i+1)
            else: 
                break
        if input_sen_index%10 == 9:      
            print(f"Instruction generated: running GPT API now. Input: {input_sen_index+1}")

        i = 0
        while i < 10:
            try:   
                response.append( gpt_complete(prompt_text=prompt_text) )
                #logger.debug(response)
                time.sleep(1)  # stop 1 second to avoid error
                break
        
            except: #InvalidRequestError: in case that the prompt and output reach the token limitation
                print("Connection error: try again automatically")
            i = i+1    
        
    return prompt_text , response


def get_direct_gpt4_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset, num_examples): #, method = "random", dataset = "film", num_examples = 1, stratify = True):
    # The input parameter "film" here is used for prompt instruction of different dataset, e.g. in domain of, different domains
    """ 
    get the gpt response with stored sentences directly, without running get_textual_instruction_with_fewshot_examples()
    We do not need input parameters: method = "random", num_examples = 1, stratify = True
    generate prompts for thin film technology dataset"""

    # get the prompt with few-shot learning examples, before running GPT API

    response = []
    # iterate over the first 30 input sentences to the prompt
    for input_sen_index, input_sen in enumerate(sentences_input):
        
        # use the selected few-shot examples for each input sentence
        fewshot_sen_one_sentence = all_fewshot_sen[input_sen_index]
        fewshot_dem_one_sentence = all_fewshot_dem[input_sen_index]

        # update the num_examples, if there is duplicated items
        # num_examples_new is used in except in case that we exceed the maximum tokens
        #num_examples = len(fewshot_sen_one_sentence)
        num_examples_new = num_examples

        instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset)
        instruction = instruction + "Extract all the entities from the input sentence below:\n" + "Input sentence: {input}".format(input = input_sen) 
        instruction = instruction + "\nExtracted entities of the input sentence:\n"
        prompt_text = instruction
        #logger.info(prompt_text)
        
        ### get the maximun number of few-shot examples
        enc = tiktoken.encoding_for_model("text-davinci-003")
        for i in range(num_examples):
            num_tokens = len( enc.encode(prompt_text) )
            if num_tokens > 32000 - 288:  ## consider 300 token length for output
                num_examples_new = num_examples_new - 1
                instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples_new, dataset)
                instruction = instruction + "Extract all the entities from the input sentence below:\n" + "Input sentence: {input}".format(input = input_sen) + "\nExtracted entities of the input sentence:\n"
                prompt_text = instruction
                #print("exceed maximun token length, -{i}", i+1)
            else: 
                break
        if input_sen_index%10 == 9:  
            print(f"Instruction generated: running GPT API now. Input: {input_sen_index+1}")

        i = 0
        while i < 10:
            try:   
                response.append( gpt_complete(prompt_text=prompt_text) )
                #logger.debug(response)
                time.sleep(1)  # stop 1 second to avoid error
                break
        
            except: #InvalidRequestError: in case that the prompt and output reach the token limitation
                print("Connection error: try again automatically")
            i = i+1    
        
    return prompt_text , response


def get_gpt_response(sentences_dev, demonstrations_dev, sentences_input, method = "random", dataset = "film", num_examples = 1, stratify = True):
    """generate prompts for thin film technology dataset"""

    # get the prompt with few-shot learning examples, before running GPT API
    all_fewshot_sen, all_fewshot_dem = get_textual_instruction_with_fewshot_examples(sentences_input, sentences_dev, demonstrations_dev, method, dataset, num_examples, stratify)

    response = []
    # iterate over the first 30 input sentences to the prompt
    for input_sen_index, input_sen in enumerate(sentences_input[:30]):
        
        # use the selected few-shot examples for each input sentence
        fewshot_sen_one_sentence = all_fewshot_sen[input_sen_index]
        fewshot_dem_one_sentence = all_fewshot_dem[input_sen_index]

        # update the num_examples, if there is duplicated items
        # num_examples_new is used in except in case that we exceed the maximum tokens
        num_examples = len(fewshot_sen_one_sentence)
        num_examples_new = len(fewshot_sen_one_sentence)

        instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset)
        #instruction = instruction + "Input: {input}".format(input = input_sen) + "Output:\n"
        instruction = instruction + "Extract all the entities from the input sentence below:\n" + "Input sentence: {input}".format(input = input_sen) 
        #instruction = instruction + "\nThere are multiple entities in the input sentence. Please extract them all. "
        instruction = instruction + "\nExtracted entities of the input sentence:\n"
        prompt_text = instruction
        #logger.info(prompt_text)
        
        ### get the maximun number of few-shot examples
        enc = tiktoken.encoding_for_model("text-davinci-003")
        for i in range(num_examples):
            num_tokens = len( enc.encode(prompt_text) )
            if num_tokens > 4096 - 300:  ## consider 300 token length for output
                num_examples_new = num_examples_new - 1
                instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples_new, dataset)
                instruction = instruction + "Extract all the entities from the input sentence below:\n" + "Input sentence: {input}".format(input = input_sen) + "\nExtracted entities of the input sentence:\n"
                prompt_text = instruction
                print("exceed maximun token length, -{i}", i+1)
            else: 
                break
            
        print(f"Instruction generated: running GPT API now. Input: {input_sen_index+1}")

        i = 0
        while i < 10:
            try:   
                response.append( gpt_complete(prompt_text=prompt_text, max_tokens=300) )
                #logger.debug(response)
                time.sleep(1)  # stop 1 second to avoid error
                break
        
            except: #InvalidRequestError: in case that the prompt and output reach the token limitation
                print("Connection error: try again automatically")
            i = i+1    
        
    return prompt_text , response

def get_textual_instruction_with_fewshot_examples(sentences_input, sentences_dev, demonstrations_dev, method, dataset, num_examples, stratify):
    """
    Half of the get_gpt_response, do not run GPT API, but get the dataset needed.
    """

    if method == "random":
        all_fewshot_sen, all_fewshot_dem = get_fewshot_sen_dem_random(sentences_input, sentences_dev, demonstrations_dev, num_examples, stratify, dataset)

    elif method == "sentence" or method == "entity":
        model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        faiss_index_list = get_index_from_faiss(sentences_dev, demonstrations_dev, model, stratify, dataset)
        all_fewshot_sen, all_fewshot_dem = get_fewshot_sen_dem_sentembedding(sentences_input, sentences_dev, demonstrations_dev, num_examples, faiss_index_list, model, stratify, dataset)
    
    elif method == "tfidf":
        # sentences_input: a list of input sentences, not one single sentence
        model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        all_fewshot_sen, all_fewshot_dem = get_fewshot_sen_dem_with_tfidf(sentences_input, sentences_dev, demonstrations_dev, model, num_examples, stratify, dataset, num_tfidf_words = 6 )
    
    elif method == "tfidf_half":
        model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
        all_fewshot_sen, all_fewshot_dem = get_fewshot_sen_dem_with_tfidf_random_half(sentences_input, sentences_dev, demonstrations_dev, model, num_examples, stratify, dataset, num_tfidf_words = 10)
    else:
        raise  ValueError ("You can only select method between 'random', 'sentence', 'entity', 'tfidf', 'tfidf_half'." )

    return all_fewshot_sen, all_fewshot_dem



def get_index_from_faiss(sentences_dev, demonstrations_dev, model, stratify, dataset):
    faiss_index_list = []
    
    if stratify:
        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        
        unique_no_order, counts_no_order = np.unique(development_entity, return_counts=True)
        # unique: the unique entity types; 
        # counts: the corresponding amounts of different entity types
        sorted_index = np.argsort(-counts_no_order)
        unique = unique_no_order[sorted_index]
        counts = sorted(counts_no_order, reverse = True)
        print(unique)
        index = []
        specific_entity_sentence_and_demon = []

        ########### Handle few shot learning dataset
        ## get numpy array (type of entity, its corresponding sentence_and_demon_train)
        for i, entity in enumerate(unique):
            # get the index of a certain entity in a numpy array
            index.append( np.where(development_entity == entity) ) 
            # dimension 1 is number of unique entities, dimension 2 is all corresponding sentence_and_demon_train of an entity
            specific_entity_sentence_and_demon.append(sentence_and_demon[index[i]])
        
        for entity_index, unique in enumerate(unique):
            start = time.time()
            # specific_entity_sentence_and_demon[entity_index] is a list of sentences and demonstrations of a one entity in development set
            # change the format from [[sentence,demom entities], ...] to [sentence, ...], [demonstrations, ...]
            specific_entity_sentence = [ specific_entity_sentence_and_demon[entity_index][i][0] for i in range(len(specific_entity_sentence_and_demon[entity_index]))]
            sentences_dev_encoded = np.asarray( model.encode( specific_entity_sentence) )
            # train faiss and add sentences_dev_encoded to the dataset
            index = faiss.IndexFlatIP( sentences_dev_encoded.shape[1])
            index.add(sentences_dev_encoded)
            faiss_index_list.append(index)
            print(unique+" takes time: ", time.time()-start )

    else: # stratify == false
        # sentences_dev is a list of sentences in development set
        # only pick 10 for "random", since we actually don't need it for random selection method.
        sentences_dev_encoded = np.asarray( model.encode(sentences_dev) )
        # train faiss and add sentences_dev_encoded to the dataset
        index = faiss.IndexFlatIP( sentences_dev_encoded.shape[1])
        index.add(sentences_dev_encoded)
        faiss_index_list.append(index)
    
    return faiss_index_list


def get_fewshot_sen_dem_sentembedding(sentences_input, sentences_dev, demonstrations_dev, num_examples, faiss_index_list, model, stratify, dataset):
    sen = []
    dem = []  
    all_fewshot_sen = [ [] for i in range(len(sentences_input))]
    all_fewshot_dem = [ [] for i in range(len(sentences_input))]

    if stratify:
        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        unique_no_order, counts_no_order = np.unique(development_entity, return_counts=True)
        # unique: the unique entity types; 
        # counts: the corresponding amounts of different entity types
        sorted_index = np.argsort(-counts_no_order)
        unique = unique_no_order[sorted_index]
        counts = sorted(counts_no_order, reverse = True)
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
        if iteration_of_all_entities < 1:
            iteration_of_all_entities = 1
        for entity_index in range( len(unique) ):
            # specific_entity_sentence_and_demon[entity_index] is a list of sentences of a one entity in development set
            specific_entity_sentence = [ specific_entity_sentence_and_demon[entity_index][i][0] for i in range(len(specific_entity_sentence_and_demon[entity_index]))]
            specific_entity_demon = [ specific_entity_sentence_and_demon[entity_index][i][1] for i in range(len(specific_entity_sentence_and_demon[entity_index]))]    
            # train faiss and add sentences_dev_encoded to the dataset
            index = faiss_index_list[entity_index]
            # input_sen is just one sentence
            input_sen_encoded = model.encode(sentences_input)
            input_sen_encoded = np.asarray( input_sen_encoded )
            
            # D is the distance matrix and I is the index matrix of shape ( len of input_sentence, k_nearest) -> (1, num_examples)
            # iteration_of_all_entities is k, k_nearest 
            D, I = index.search(input_sen_encoded, iteration_of_all_entities)    
            for i in range(len(sentences_input)): 
                for j in range(iteration_of_all_entities):  
                    index = I[i][j]
                    all_fewshot_sen[i].append(specific_entity_sentence[index])
                    all_fewshot_dem[i].append(specific_entity_demon[index])

        for i in range(len(sentences_input)):
            # rearrange the list --> all samples in each entity will be placed round after round in the list        
            all_fewshot_sen_copy = all_fewshot_sen[i]
            all_fewshot_dem_copy = all_fewshot_dem[i]
            all_fewshot_sen[i] = []
            all_fewshot_dem[i] = []
            for k in range(iteration_of_all_entities): 
                for j in range(len(unique)):
                    # considering that iteration_of_all_entities might be larger than the counts of a certain entity type
                    if iteration_of_all_entities > counts[j]:
                        continue
                    all_fewshot_sen[i].append(all_fewshot_sen_copy[k + iteration_of_all_entities*j])
                    all_fewshot_dem[i].append(all_fewshot_dem_copy[k + iteration_of_all_entities*j])


            print( "Duplicated few-shot examples: ", len(all_fewshot_sen[i]))   
            combined_sen_dem = np.asarray([all_fewshot_sen[i],all_fewshot_dem[i]])
            combined_sen_dem = np.unique(combined_sen_dem, axis = 1)
            all_fewshot_sen[i] = list(combined_sen_dem[0] )
            all_fewshot_dem[i] = list(combined_sen_dem[1] )
            print( "Length of stratified_sentences_train: ", len(all_fewshot_sen[i]) )  
            all_fewshot_sen[i] = all_fewshot_sen[i][:num_examples]
            all_fewshot_dem[i] = all_fewshot_dem[i][:num_examples]
            # print("Considering num_examples:",len(all_fewshot_sen[i]), "Num_examples: ", num_examples )


    else: # not stratify the few-shot examples from the 300 development dataset
        # input_sen is just one sentence
        input_sen_encoded = model.encode(sentences_input)
        input_sen_encoded = np.asarray( input_sen_encoded )
        # D is the distance matrix and I is the index matrix of shape ( len of input_sentence, k_nearest) -> (1, num_examples)
        index = faiss_index_list[0]
        D, I = index.search(input_sen_encoded, num_examples)
        #print(I)
        for i in range(len(sentences_input)):
            for j in range(num_examples):
                index = I[i][j]
                all_fewshot_sen[i].append(sentences_dev[index])
                all_fewshot_dem[i].append(demonstrations_dev[index])

    return all_fewshot_sen, all_fewshot_dem



def get_fewshot_sen_dem_random(sentences_input, sentences_dev, demonstrations_dev, num_examples, stratify, dataset):
    sen = [] 
    dem = []
    all_fewshot_sen = []
    all_fewshot_dem = []
    if stratify :
        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        
        unique_no_order, counts_no_order = np.unique(development_entity, return_counts=True)
        # unique: the unique entity types; 
        # counts: the corresponding amounts of different entity types
        sorted_index = np.argsort(-counts_no_order)
        unique = unique_no_order[sorted_index]
        counts = sorted(counts_no_order, reverse = True)

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
        if iteration_of_all_entities < 1:
            iteration_of_all_entities = 1

        """ all samples in each entity will be placed together in the list """
        # for entity_index in range( len(unique) ):
        #     # a list of random int numbers to replace iteration_of_all_entities in order
        #     if counts[entity_index] < iteration_of_all_entities:
        #         iteration_of_all_entities = counts[entity_index]
        #     sample_index_selected = random.sample( range(0, counts[entity_index]), iteration_of_all_entities) 
        #     for iteration in range(iteration_of_all_entities):
        #         few_shot_examples.append(specific_entity_sentence_and_demon[entity_index][sample_index_selected[iteration]])
        """ all samples in each entity will be placed round after round in the list """
        for iteration in range(iteration_of_all_entities):
            for entity_index in range( len(unique) ):
                # To avoid the case that iteration_of_all_entities > counts[entity_index]
                if iteration >= counts[entity_index]:
                    break
                sample_index_selected = random.sample( range(0, counts[entity_index]), 1)
                few_shot_examples.append(specific_entity_sentence_and_demon[entity_index][sample_index_selected[0]])

        print( "Duplicated few-shot examples: ", len(few_shot_examples))   
        few_shot_examples = np.asarray(few_shot_examples)
        few_shot_examples = np.unique(few_shot_examples, axis = 0) # eliminate duplicated rows
        print( "Length of stratified_sentences_train: ", len(few_shot_examples) )
        sen = list(few_shot_examples[:num_examples,0] )
        dem = list(few_shot_examples[:num_examples,1] )  # np.unique(few_shot_examples[:,1])
        print("Considering num_examples:",len(sen), "Num_examples: ", num_examples )

    else:
        # randomly sample index for pop out demonstration examples
        values = list(range(0, len(sentences_dev)))
        #random.seed(44)
        random.shuffle(values)
        for i in range(num_examples):
            index = values.pop()
            sen.append(sentences_dev[index])
            dem.append(demonstrations_dev[index])

    # sen and dem is the randomly selected few-shot examples for one input sentence
    # use the same random few-shot examples for all input sentences, broadcasting to ( N_input_sentences, N_few_shots_examples )
    for i in range( len(sentences_input) ):
        all_fewshot_sen.append(sen )
        all_fewshot_dem.append(dem )     

    return all_fewshot_sen, all_fewshot_dem 


def get_fewshot_sen_dem_with_tfidf(sentences_input, sentences_dev, demonstrations_dev, model, num_examples, stratify, dataset, num_tfidf_words = 10):
    if stratify:
        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        unique_no_order, counts_no_order = np.unique(development_entity, return_counts=True)
        # unique: the unique entity types; 
        # counts: the corresponding amounts of different entity types
        # place the unique and counts list in order frpm largest amount to the smallest amount
        sorted_index = np.argsort(-counts_no_order)
        unique = unique_no_order[sorted_index]
        counts = sorted(counts_no_order, reverse = True)

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


        preprocessed_documents_dev = [preprocess_document(document) for document in sentences_dev]
        preprocessed_documents_input = [preprocess_document(document) for document in sentences_input]
        corpus = preprocessed_documents_dev
        v = TfidfVectorizer(min_df=0.00001, max_df=0.2, analyzer='word')
        v.fit(corpus)
        X_input_v = v.transform(preprocessed_documents_input)
        xq = X_input_v.toarray()
        
        all_fewshot_sen = []
        all_fewshot_dem = []
        iteration_of_all_entities = int( num_examples/len(unique) )  # how many total number of samples of one entity is needed for demon_size
        # in case that num_examples is smaller than len(unique) -> avoid iteration_of_all_entities = 0
        if iteration_of_all_entities < 1:
            iteration_of_all_entities = 1

        #iterate for each input sentence
        for i in range(len(preprocessed_documents_input)): 
            fewshot_sen_one_sentence = []
            fewshot_dem_one_sentence = []
            combined_sen_dem_one_sentence = []
            # iterate for each entity for iteration_of_all_entities times
            for entity_index in range( len(unique) ):
                sen_one_entity = specific_entity_sentence_and_demon[entity_index][:,0] 
                dem_one_entity = specific_entity_sentence_and_demon[entity_index][:,1] 
                X_dev_v = v.transform(sen_one_entity)
                xb = X_dev_v.toarray()
                cos_sim = cosine_similarity( xq, xb) # shape(1000, len(xb)) -> 1000 input examples, len(xb) few-shot examples of a certain entity
                top_index_xb = np.argsort( -cos_sim, axis=1)
                # for iteration_of_all_entities times
                for top_index in top_index_xb[i, :iteration_of_all_entities]:
                    combined_sen_dem_one_sentence.append([sen_one_entity[top_index], dem_one_entity[top_index] ])

            # rearrange the list --> all samples in each entity will be placed round after round in the list        
            combined_sen_dem_one_sentence_copy = combined_sen_dem_one_sentence
            combined_sen_dem_one_sentence = []
            for i in range(iteration_of_all_entities) : 
                for j in range(len(unique)):
                    # considering that iteration_of_all_entities might be larger than the counts of a certain entity type
                    if iteration_of_all_entities > counts[j]:
                        continue
                    combined_sen_dem_one_sentence.append(combined_sen_dem_one_sentence_copy[i+ iteration_of_all_entities*j])

            print("Duplicated: ", len(combined_sen_dem_one_sentence))       
            combined_sen_dem_one_sentence = np.asarray( combined_sen_dem_one_sentence)   
            combined_sen_dem_one_sentence = np.unique(combined_sen_dem_one_sentence, axis = 0) # eliminate duplicated rows
            print("Unique: ", len(combined_sen_dem_one_sentence))
            fewshot_sen_one_sentence = combined_sen_dem_one_sentence[:num_examples,0]  # limit number of few-shot examples in case that num_examples < number of entities
            fewshot_dem_one_sentence = combined_sen_dem_one_sentence[:num_examples,1]  # limit number of few-shot examples in case that num_examples < number of entities
            #print("considering num_examples: ", len(fewshot_sen_one_sentence), "num_examples: ", num_examples )

            all_fewshot_sen.append(fewshot_sen_one_sentence)
            all_fewshot_dem.append(fewshot_dem_one_sentence)

    else: # not stratify when selecting 20 examples from the 300 development dataset
        preprocessed_documents_dev = [preprocess_document(document) for document in sentences_dev]
        preprocessed_documents_input = [preprocess_document(document) for document in sentences_input]

        corpus = preprocessed_documents_dev
        #corpus = ["She doesn't like cooking", "Thin film technology dataset is difficult to handle", "I hate NLP", "I am handsome.", "I am a pretty girl.", "I am a machine learning expert."]
        v = TfidfVectorizer(min_df=0.00001, max_df=0.2, analyzer='word')
        v.fit(corpus)
        X_dev_v = v.transform(preprocessed_documents_dev)
        xb = X_dev_v.toarray()
        X_input_v = v.transform(preprocessed_documents_input)
        xq = X_input_v.toarray()
        cos_sim = cosine_similarity( xq, xb) # shape(1000, 300) -> 1000 input examples, 300 few-shot examples
        # get index of the top K most similar few-shot sentences for each input sentence
        top_index_xb = np.argsort( -cos_sim, axis=1)

        all_fewshot_sen = []
        all_fewshot_dem = []
        for i in range(len(preprocessed_documents_input)):
            fewshot_sen_one_sentence = []
            fewshot_dem_one_sentence = []
            for top_index in top_index_xb[i, :num_examples]:
                fewshot_sen_one_sentence.append(sentences_dev[top_index])
                fewshot_dem_one_sentence.append(demonstrations_dev[top_index])
            all_fewshot_sen.append(fewshot_sen_one_sentence)
            all_fewshot_dem.append(fewshot_dem_one_sentence)

    return all_fewshot_sen, all_fewshot_dem 

def get_fewshot_sen_dem_with_tfidf_random_half(sentences_input, sentences_dev, demonstrations_dev, model, num_examples, stratify, dataset, num_tfidf_words = 10):
    
    """ 
    If the most similar sample or sentence induces the model in a negative way, i.e. with wrong labels, the performance will hurt.
    Therefore, we consider include some random examples in the few-shot examples.
    """

    if stratify:
        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        unique_no_order, counts_no_order = np.unique(development_entity, return_counts=True)
        # unique: the unique entity types; 
        # counts: the corresponding amounts of different entity types
        # place the unique and counts list in order frpm largest amount to the smallest amount
        sorted_index = np.argsort(-counts_no_order)
        unique = unique_no_order[sorted_index]
        counts = sorted(counts_no_order, reverse = True)

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


        preprocessed_documents_dev = [preprocess_document(document) for document in sentences_dev]
        preprocessed_documents_input = [preprocess_document(document) for document in sentences_input]
        corpus = preprocessed_documents_dev
        v = TfidfVectorizer(min_df=0.00001, max_df=0.2, analyzer='word')
        v.fit(corpus)
        X_input_v = v.transform(preprocessed_documents_input)
        xq = X_input_v.toarray()
        
        all_fewshot_sen = []
        all_fewshot_dem = []
        iteration_of_all_entities = int( num_examples/len(unique) )  # how many total number of samples of one entity is needed for demon_size
        # in case that num_examples is smaller than len(unique) -> avoid iteration_of_all_entities = 0
        if iteration_of_all_entities < 1:
            iteration_of_all_entities = 1

        #iterate for each input sentence
        for i in range(len(preprocessed_documents_input)): 
            fewshot_sen_one_sentence = []
            fewshot_dem_one_sentence = []
            combined_sen_dem_one_sentence = []
            # iterate for each entity for iteration_of_all_entities times
            for entity_index in range( len(unique) ):
                sen_one_entity = specific_entity_sentence_and_demon[entity_index][:,0] 
                dem_one_entity = specific_entity_sentence_and_demon[entity_index][:,1] 
                X_dev_v = v.transform(sen_one_entity)
                xb = X_dev_v.toarray()
                cos_sim = cosine_similarity( xq, xb) # shape(1000, len(xb)) -> 1000 input examples, len(xb) few-shot examples of a certain entity
                top_index_xb = np.argsort( -cos_sim, axis=1)
                # for iteration_of_all_entities times
                for top_index in top_index_xb[i, :iteration_of_all_entities]:
                    combined_sen_dem_one_sentence.append([sen_one_entity[top_index], dem_one_entity[top_index] ])

            # rearrange the list --> all samples in each entity will be placed round after round in the list        
            combined_sen_dem_one_sentence_copy = combined_sen_dem_one_sentence
            combined_sen_dem_one_sentence = []
            for i in range(iteration_of_all_entities) : 
                for j in range(len(unique)):
                    # considering that iteration_of_all_entities might be larger than the counts of a certain entity type
                    if iteration_of_all_entities > counts[j]:
                        continue
                    combined_sen_dem_one_sentence.append(combined_sen_dem_one_sentence_copy[i+ iteration_of_all_entities*j])

            print("Duplicated: ", len(combined_sen_dem_one_sentence))       
            combined_sen_dem_one_sentence = np.asarray( combined_sen_dem_one_sentence)   
            combined_sen_dem_one_sentence = np.unique(combined_sen_dem_one_sentence, axis = 0) # eliminate duplicated rows
            print("Unique: ", len(combined_sen_dem_one_sentence))
            fewshot_sen_one_sentence = combined_sen_dem_one_sentence[:num_examples,0]  # limit number of few-shot examples in case that num_examples < number of entities
            fewshot_dem_one_sentence = combined_sen_dem_one_sentence[:num_examples,1]  # limit number of few-shot examples in case that num_examples < number of entities
            #print("considering num_examples: ", len(fewshot_sen_one_sentence), "num_examples: ", num_examples )


            ### The codes above are for stratified if-idf
            ### The codes below we add random samples

            values = list(range(0, len(sentences_dev)))
            #random.seed(44)
            random.shuffle(values)
            index = values.pop()
            # iterate the few-shot examples for each input sentence
            fewshot_sen_one_sentence_copy = fewshot_sen_one_sentence
            fewshot_dem_one_sentence_copy = fewshot_dem_one_sentence
            for k, item in enumerate(fewshot_sen_one_sentence_copy):
                if k%2 == 0: ### even number, use the sample from TF-IDF
                    fewshot_sen_one_sentence[k] = fewshot_sen_one_sentence_copy[k//2]
                    fewshot_dem_one_sentence[k] = fewshot_dem_one_sentence_copy[k//2]
                else:  ### odd number, use the sample from random sampling
                    index = values.pop()
                    fewshot_sen_one_sentence[k] = sentences_dev[index]
                    fewshot_dem_one_sentence[k] = demonstrations_dev[index]

            all_fewshot_sen.append(fewshot_sen_one_sentence)
            all_fewshot_dem.append(fewshot_dem_one_sentence)

    else: # not stratify when selecting 20 examples from the 300 development dataset
        preprocessed_documents_dev = [preprocess_document(document) for document in sentences_dev]
        preprocessed_documents_input = [preprocess_document(document) for document in sentences_input]

        corpus = preprocessed_documents_dev
        #corpus = ["She doesn't like cooking", "Thin film technology dataset is difficult to handle", "I hate NLP", "I am handsome.", "I am a pretty girl.", "I am a machine learning expert."]
        v = TfidfVectorizer(min_df=0.00001, max_df=0.2, analyzer='word')
        v.fit(corpus)
        X_dev_v = v.transform(preprocessed_documents_dev)
        xb = X_dev_v.toarray()
        X_input_v = v.transform(preprocessed_documents_input)
        xq = X_input_v.toarray()
        cos_sim = cosine_similarity( xq, xb) # shape(1000, 300) -> 1000 input examples, 300 few-shot examples
        # get index of the top K most similar few-shot sentences for each input sentence
        top_index_xb = np.argsort( -cos_sim, axis=1)

        #######
        all_fewshot_sen = []
        all_fewshot_dem = []
        for i in range(len(preprocessed_documents_input)):
            fewshot_sen_one_sentence = []
            fewshot_dem_one_sentence = []
            values = list(range(0, len(sentences_dev)))
            #random.seed(44)
            random.shuffle(values)

            # iterate the few-shot examples for each input sentence
            for k, top_index in enumerate(top_index_xb[i, :num_examples]):
                if k%2 == 0: ### even number, use the sample from TF-IDF
                    fewshot_sen_one_sentence.append(sentences_dev[top_index])
                    fewshot_dem_one_sentence.append(demonstrations_dev[top_index])
                else:  ### odd number, use the sample from random sampling
                    index = values.pop()
                    fewshot_sen_one_sentence.append(sentences_dev[index])
                    fewshot_dem_one_sentence.append(demonstrations_dev[index])
                    
            all_fewshot_sen.append(fewshot_sen_one_sentence)
            all_fewshot_dem.append(fewshot_dem_one_sentence)

    return all_fewshot_sen, all_fewshot_dem 

def preprocess_document(document):
    """function of data preprocessing the textual data for tf-idf method"""
    # Tokenization
    tokens = word_tokenize(document)
    # Lowercase conversion
    tokens = [token.lower() for token in tokens]
    # Punctuation removal
    tokens = [token for token in tokens if token not in string.punctuation]
    # Stop word removal
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return " ".join(tokens)

def get_unique_entity_list(dataset):
    """ place the entity types in order for all methods, i.e. from the largest to the smallest"""
    if dataset == "film":
        unique_entity_list = ['Component', 'Location', 'Function', 'EnergyFlow','Attribution','Material','Effect','System','ScientificConcept','Shape','Value','InfoFlow','Consequence','PhysicsFlow','Measure','State','AnnotatorNotes']
    elif dataset == "fabner":
        unique_entity_list = ['Component', 'Location', 'Function', 'EnergyFlow','Attribution','Material','Effect','System','ScientificConcept','Shape','Value','InfoFlow','Consequence','PhysicsFlow','Measure','State','AnnotatorNotes']
    elif dataset == "assembly":
        unique_entity_list = ['Component', 'Location', 'Function', 'EnergyFlow','Attribution','Material','Effect','System','ScientificConcept','Shape','Value','InfoFlow','Consequence','PhysicsFlow','Measure','State','AnnotatorNotes']    
    else:
        raise  ValueError ("You can only select dataset between 'film', 'fabner', 'assembly'." ) 
    return  unique_entity_list

def get_fewshot_sen_dem_with_tfidf_and_simcse(sentences_input, sentences_dev, demonstrations_dev, model, num_examples, stratify, num_tfidf_words = 10):
    
    """ The old verison of tfidf """
    
    if stratify :
        # create empty list for final output, shape (N_input_sentences, N_selected_fewshot_examples)
        all_fewshot_sen = [ [] for i in range(len(sentences_input))]
        all_fewshot_dem = [ [] for i in range(len(sentences_input))]

        # use sentences in development dataset as corpus
        v = TfidfVectorizer(min_df=0.001, max_df=0.8, analyzer='word')
        fitted_corpus = v.fit(sentences_dev)
        all_words = v.get_feature_names_out()
        xq = np.float32(  v.transform(sentences_input).toarray() )  

        ## pick top N words with highest tfidf scores
        N = num_tfidf_words
        top_index_q = np.argsort( -xq, axis=1)
        xq_tfidf = all_words[top_index_q[:,:N]]
        xq_tfidf_joined = []
        for i, tfidf_words in enumerate(xq_tfidf):
            xq_tfidf_joined.append( " ".join(tfidf_words) ) 
        #print(xq_tfidf_joined)

        # the sentences_dev and demonstration_dev are in the same order, combine them
        entity_list = get_entity_from_demonstrations(demonstrations_dev)
        sentence_and_demon, development_entity = get_combined_dataset( sentences_dev, demonstrations_dev, entity_list )
        # get the unique entities and how many time they appear in the development dataset
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
            # get the specific_entity_sentence and entities separately from specific_entity_sentence_and_demon
            specific_entity_sentence = [ specific_entity_sentence_and_demon[entity_index][i][0] for i in range(len(specific_entity_sentence_and_demon[entity_index]))]
            specific_entity_demon = [ specific_entity_sentence_and_demon[entity_index][i][1] for i in range(len(specific_entity_sentence_and_demon[entity_index]))]    

            # transform all original sentences containing a specific entity into xb
            xb = np.float32(  v.transform( specific_entity_sentence ).toarray() )   
            top_index_b = np.argsort( -xb, axis=1)
            xb_tfidf = all_words[top_index_b[:,:N]]
            xb_tfidf_joined = []
            for i, tfidf_words in enumerate(xb_tfidf):
                xb_tfidf_joined.append( " ".join(tfidf_words) ) 

            # calculate similarity    
            similarity_matrix = model.similarity(xq_tfidf_joined, xb_tfidf_joined)

            ### pick top k number of examples as few-shots
            k = iteration_of_all_entities
            similarity_index = np.argsort(-similarity_matrix, axis = 1)
            similarity_index_selected = similarity_index[:, :k]
            #print(similarity_index_selected.shape, similarity_index_selected)

            # save the few-shot examples for each input sentence, ( N_input_sentences, N_few_shots_examples )
            # iterate over all input sentence, number of input sentences = len(sentences_input)
            for i in range( len(sentences_input)):
                for j in range(iteration_of_all_entities):
                    #print(similarity_index_selected.shape, i, j)
                    #print(similarity_index_selected[i,j])
                    # We have to use specific_entity_sentence instead of sentences_input here, 
                    # the length of specific_entity_sentence is much larger in some cases, because some 
                    # entities appear several time in one sentence.
                    all_fewshot_sen[i].append(specific_entity_sentence[similarity_index_selected[i,j]])
                    all_fewshot_dem[i].append(specific_entity_demon[similarity_index_selected[i,j]])
            
        # eliminate duplicated few-shot examples for all input sentences        
        for i in range(len(sentences_input)):
            print( "Duplicated few-shot examples: ", len(all_fewshot_sen[i]))   
            combined_sen_dem = np.asarray([all_fewshot_sen[i],all_fewshot_dem[i]])
            combined_sen_dem = np.unique(combined_sen_dem, axis = 1)
            all_fewshot_sen[i] = list(combined_sen_dem[0] )
            all_fewshot_dem[i] = list(combined_sen_dem[1] )
            print( "length of stratified_sentences_train: ", len(all_fewshot_sen[i]) ) 

    else:  # do not use stratify in Tfidf  
        v = TfidfVectorizer(min_df=0.001, max_df=0.8, analyzer='word')
        fitted_corpus = v.fit(sentences_dev)
        all_words = v.get_feature_names_out()
        xb = np.float32(  v.transform(sentences_dev).toarray() )   
        xq = np.float32(  v.transform(sentences_input).toarray() )  

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
        similarity_index_selected = similarity_index[:, :k]

        # save the few-shot examples for each input sentence, ( N_input_sentences, N_few_shots_examples )
        all_fewshot_sen = []
        all_fewshot_dem = [] 
        for i in range(similarity_matrix.shape[0]):
            sen = []
            dem = []
            for j in range(num_examples):
                sen.append(sentences_dev[similarity_index_selected[i,j]])
                dem.append(demonstrations_dev[similarity_index_selected[i,j]])
            all_fewshot_sen.append(sen)
            all_fewshot_dem.append(dem)

    return all_fewshot_sen, all_fewshot_dem    


def change_output_format_to_tokens_tags(sentences_input, response):
    total_list = []
    #print(response[-1] + "\n")
    for i, sin_response in enumerate(response):
        #print( i, response[i], sentences_input[i] )
        dict_one_sentence = {}
        dict_one_sentence["tokens"] = sentences_input[i].split(" ")
        dict_one_sentence["ner_tags"] = ["O" for k in range(len(dict_one_sentence["tokens"])+1)]
        ## print(dict_one_sentence)
        sin_item_entity_list = sin_response.split("\n\n")[0] # just to prevent \n split the \n\n
        sin_item_entity_list = sin_item_entity_list.split("\n")
        
        for sin_item_entity in sin_item_entity_list:
            #print(sin_item_entity)
            if len( sin_item_entity.split(": ") ) == 1:
                # ignore the case the "no entity in this sentence" "O" is assigned before.
                continue
            else:
                # prevent Nd : YAG: MATE
                sin_entity = sin_item_entity.split(": ")[-1]
                sin_item = sin_item_entity.split(": " + sin_entity)[0]
            #print(sin_item, sin_entity)

            if sin_item == "": # dolly output empty entities
                sin_item = "Scheisse"
            len_sin_item = len(sin_item.split())
                
            except_item = sentences_input[i].split(sin_item) 
            if len(except_item) == 1:
                # in case that sim_item is the first word of the sentence, set position_begin_item = 0
                position_begin_item = 0
                # update the ner_tags for len(except_item) == 1 and 2      
                for k in range(len_sin_item):
                    if position_begin_item+k >= len(dict_one_sentence["ner_tags"]):
                    # this if condition is used to make sure that we can at least get the results from evaluation
                    # Sometimes sin_items can not be separated correctly    
                        break
                    dict_one_sentence["ner_tags"][position_begin_item+k] = sin_entity


            elif len(except_item) == 2: 
                # pick the index of this sin_item as position_begin_item
                position_begin_item = len (   word_tokenize(  except_item[0].strip() )    )      
                # update the ner_tags for len(except_item) == 1 and 2    
                for k in range(len_sin_item):
                    dict_one_sentence["ner_tags"][position_begin_item+k] = sin_entity     
                         
            else:
                # in case that sim_items happen twice in the original sentence, len(except_item) > 2
                for sin_item_index in range(len(except_item)-1):
                    position_begin_item = len (   word_tokenize(  except_item[sin_item_index].strip() )    )   
                    for k in range(len_sin_item):
                        if dict_one_sentence["ner_tags"][position_begin_item+k] != "O":
                            dict_one_sentence["ner_tags"][position_begin_item+k] = sin_entity
                #continue 

            # update the ner_tags for len(except_item) == 1 and 2      
            #for k in range(len_sin_item):
            #    dict_one_sentence["ner_tags"][position_begin_item+k] = sin_entity

        total_list.append(dict_one_sentence)
    return total_list


def get_classification_report(transformed_solution, transformed_output, target_names = None):
    y_true = []
    y_pred = []
    for i in range(len(transformed_solution)):
        y_true.append( transformed_solution[i]["ner_tags"] )
        y_pred.append( transformed_output[i]["ner_tags"] )    
    y_true = [entity  for sublist in y_true for entity in sublist]
    y_pred = [entity  for sublist in y_pred for entity in sublist]
    #print(y_true)
    #print(y_pred)
    if target_names == None:
        print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0) )
    else:
        print(classification_report(y_true=y_true, y_pred=y_pred, zero_division=0, target_names=target_names) )


def get_evaluation_without_o(transformed_solution, transformed_output, target_names = None):
    warnings.filterwarnings("ignore")
    y_true = []
    y_pred = []
    for i in range(len(transformed_solution)):
        for element in transformed_solution[i]["ner_tags"]:
            y_true.append( element )
        for element in transformed_output[i]["ner_tags"]:
            y_pred.append( element )    
    extracted_entities = list(set(y_true))
    # ignore O entities.
    extracted_entities.remove("O")
    print(extracted_entities)

    entities_score = np.zeros((len(extracted_entities),5))
    columns = ['entity', 'precision', 'recall', 'F1 score', 'support']
    entities_score = pd.DataFrame(entities_score, columns=columns)

    num_true_positive_total = 0
    denominator_precision_total = 0
    denominator_recall_total = 0
    # iterate over all entities except "O"
    for i, entity in enumerate(extracted_entities): 
        entities_score['entity'][i] = entity
        denominator_precision = y_pred.count(entity)
        denominator_recall = y_true.count(entity)
        num_true_positive = 0 
        # get all true positive entities, i.e. the entity is in y_pred and y_true
        for j, item in enumerate(y_pred):
            if item == entity and item == y_true[j]:
                num_true_positive = num_true_positive + 1
        # precision
        if denominator_precision == 0: # prevent 0 division
            entities_score['precision'][i] = 0
        else:    
            entities_score['precision'][i] = num_true_positive / denominator_precision * 100
        # recall
        if denominator_recall == 0:  # prevent 0 division
            entities_score['recall'][i] = 0 
        else:    
            entities_score['recall'][i] = num_true_positive / denominator_recall * 100
        # F1 score
        if entities_score['precision'][i] + entities_score['recall'][i] == 0:  # prevent 0 division
            entities_score['F1 score'][i] = 0
        else:    
            entities_score['F1 score'][i] = 2*entities_score['precision'][i]*entities_score['recall'][i] / (entities_score['precision'][i]+entities_score['recall'][i]) / 100
        # update support value
        entities_score['support'][i] = int(denominator_recall)
        # for accuracy calculation
        num_true_positive_total = num_true_positive_total + num_true_positive
        denominator_precision_total = denominator_precision_total + denominator_precision
        denominator_recall_total =  denominator_recall_total + denominator_recall 

    ## calculate the average
    average_score = np.zeros((3,5))
    columns = ['entity', 'precision', 'recall', 'F1 score', 'support']
    average_score = pd.DataFrame(average_score, columns=columns)    
    average_score["entity"][:3] = ["accuracy","macro avg","weighted avg"]
    average_score["support"][:3] = int(entities_score['support'].sum())
    
    # accuracy
    average_score["precision"][0] = "-"
    average_score["recall"][0] = "-"
    average_score["F1 score"][0] = "-"

    # macro mean
    average_score["precision"][1] = entities_score['precision'].mean()
    average_score["recall"][1] = entities_score['recall'].mean()
    average_score["F1 score"][1] = entities_score['F1 score'].mean()

    # weighted mean
    for i, entity in enumerate(extracted_entities): 
        average_score["precision"][2] = average_score["precision"][2] + entities_score['precision'][i]*entities_score['support'][i]/average_score["support"][0]
        average_score["recall"][2] = average_score["recall"][2] + entities_score['recall'][i]*entities_score['support'][i]/average_score["support"][0]
        average_score["F1 score"][2] = average_score["F1 score"][2] + entities_score['F1 score'][i]*entities_score['support'][i]/average_score["support"][0]

    # update accuracy
    # in multi-class classification cases where each observation has a single label, 
    # the micro-F1, micro-precision, micro-recall, and accuracy share the same value.
    # In fact, the number of FN and FP should be same, but we transform the format containing entites O.
    # This makes the number of FN and FP not identical. 
    # FP - 1 in a certain class will not lead to TN -1  in the rest of classes.
     
    average_score["F1 score"][0] = 2*num_true_positive_total / (denominator_recall_total + denominator_recall_total) * 100

    return pd.concat([ entities_score, average_score])