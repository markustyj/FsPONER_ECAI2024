import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import pickle
import tiktoken
import sys
import torch
import transformers
from transformers import pipeline, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
from gpt_ner_api_codes.prompts_all import *
from timeit import default_timer
import nltk
nltk.download('punkt')


from huggingface_hub import login
login(token = "")


model = "NousResearch/Llama-2-7b-chat-hf"
#model = "meta-llama/Llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model,  trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model,
    #load_in_8bit=True,
    #load_in_4bit=True,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True,
    #max_memory={0: "14.5GIB", "cpu": "14GIB"},
    #max_memory = {0: '22.5GB', 1: '22.5GB', 2: '1MB', 3: '1MB'}

)

# Use a pipeline as a high-level helper
pipe = pipeline("text-generation", model=base_model, tokenizer = tokenizer, 
                return_full_text=False, max_new_tokens=4096,  #max_length=4096,
              trust_remote_code = True, device_map = "auto")

######################################################################################################
## Load the model with the code above
## Load the 2 functions below
######################################################################################################

def get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset):
    """The old instruction and prompt for running GPT API 3.5"""
    if dataset == "film":
        instruction = "You are an excellent linguist in the domain of thin film head technology. The task is to label the entities in the given sentence. " \
            "The optional entities include Component, Location, Function, EnergyFlow (Energy Flow), Attribution, Material, Effect, System, ScientificConcept (Scientific Concept), "\
            "Shape,  Value, InfoFlow (Information Flow), Consequence, PhysicsFlow (Physics Flow), Measure, State, AnnotatorNotes. "\
    #        "Below are some examples.\n\n"
    elif dataset == "fabner":
        instruction = "You are an excellent linguist in manufacturing domain. The task is to label the entities in the given sentence. " \
            "The optional entities include CONPRI (Concept/principles), MATE (Material), MANP (Manufacturing process), PRO (Mechanical properties), PARA (Process parameters), CHAR (Process characterization), "\
            "MACEQ (Machine/equipment), APPL (Application), FEAT (Engineering features), ENAT (Enabling technology), MANS (Manufacturing standards), BIOP (Biomedical). "\
    #        "Below are some examples.\n\n"  
    elif dataset == "assembly":
        instruction = "You are an excellent linguist in assembly instructions. The task is to label the entities in the given sentence. " \
            "The optional entities include PART (parts), RPOS (relative positions), OPER (operations), TOOL (tools), ID (identification numbers), QTY (quantity), "\
            "DIM (dimensions), WGT (weights), PROP (general properties). "\
    #        "Below are some examples.\n\n"      
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

        instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset)
        instruction = instruction + "Extract the entities from the input sentence below:\n" + "Input sentence: {input}".format(input = input_sen) 
        instruction = instruction + "\nExtracted entities of the input sentence:\n"
        prompt_text = instruction
        #logger.info(prompt_text)
        
        ### get the maximun number of few-shot examples
        #enc = tiktoken.encoding_for_model("text-davinci-003")
        for i in range(100):
            num_tokens = len( tokenizer.encode(prompt_text) )
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

        # i = 0
        # output = ""
        # while i < 10:
        #     start = default_timer()
        #     try:   
        #         output = pipe(prompt_text) # , max_new_tokens=4096, return_full_text=False, #temperature = 0.1, #do_sample=True,#top_k=10,#num_return_sequences=1)
        #         break       
        #     except: #InvalidRequestError: in case that the prompt and output reach the token limitation
        #         print("Connection error: try again automatically")
        #     i = i+1    
        
        start = default_timer()
        output = pipe(prompt_text) #, max_new_tokens=4096, return_full_text=False, temperature = 0.1, do_sample=True,#top_k=10,#num_return_sequences=1,                         
        print("running time for this input sentence ", default_timer() - start)
        response.append( output )

    return prompt_text , response


######################################################################################################
## load the data, run inference, save the result
######################################################################################################


def run_inference(dataset, method, stratified, num_examples):

    if dataset == "forget":

        with open("data/immutable_data_formal/sen_thf_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_thf_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)         
        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 
        #### zero-shot
        #### example = 0
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 0)
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # #### example = 1 
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 0)
        # with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)    

        with open("data/immutable_data_formal/sen_assembly_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_assembly_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   
        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 
        #### zero-shot
        #### example = 0
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 0)
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 


        with open("data/immutable_data_formal/sen_fabner_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_fabner_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   
        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 
        #### zero-shot
        #### example = 0
        print("fabner zero-shot")
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 0)
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
    




    if dataset == "film":
        with open("data/immutable_data_formal/sen_thf_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_thf_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   

#if method == "random":
    #if stratified ==  "false":
        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 

        #### zero-shot
        #### example = 0
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 0)
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)

        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_1_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_thf_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5)
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10)
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 20  
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20)
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 


#elif method == "sentence":
    #if stratified ==  "false":
        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_1_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_thf_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5)
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10)
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 20  
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20)
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 


#elif method == "tfidf":
    #if stratified == "false":
        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_1_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)       
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_thf_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


#elif method == "tfidf_half":
    #if stratified == "false":
        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_5_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_10_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/thf_not_stratified_input_30_example_20_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_thf_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_5_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_10_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/thf_stratified_input_30_example_20_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)        
#else: ## 'random', 'sentence', 'tfidf', 'tfidf_half'.
    #raise  ValueError ("You can only select dataset between 'random', 'sentence', 'tfidf', 'tfidf_half'." )  










    elif dataset =="assembly":
        with open("data/immutable_data_formal/sen_assembly_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_assembly_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   


#if method == "random":
#    if stratified ==  "false":
        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 

        #### zero-shot
        #### example = 0
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 0)
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)  

        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_1_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


#    else: # stratified == True
        with open("data/immutable_data_formal/sen_assembly_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5)
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10)
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 20  
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20)
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 


#elif method == "sentence":
#    if stratified ==  "false":
        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_1_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


#    else: # stratified == True
        with open("data/immutable_data_formal/sen_assembly_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5)
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10)
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 20  
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20)
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 


#elif method == "tfidf":
#    if stratified == "false":
        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_1_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)    
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


#    else: # stratified == True
        with open("data/immutable_data_formal/sen_assembly_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


#elif method == "tfidf_half":
#    if stratified == "false":
        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_5_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_10_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/assembly_not_stratified_input_30_example_20_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


#    else: # stratified == True
        with open("data/immutable_data_formal/sen_assembly_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_5_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_10_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/assembly_stratified_input_30_example_20_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)        
#else: ## 'random', 'sentence', 'tfidf', 'tfidf_half'.
#    raise  ValueError ("You can only select dataset between 'random', 'sentence', 'tfidf', 'tfidf_half'." )

    




    elif dataset == "fabner":    
        with open("data/immutable_data_formal/sen_fabner_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_fabner_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   


#if method == "random":
    #if stratified ==  "false":
        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 

        #### zero-shot
        #### example = 0
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 0)
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)

        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_1_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)    
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_fabner_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5)
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10)
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 20  
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20)
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 


#elif method == "sentence":
    #if stratified ==  "false":
        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_1_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)    
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_fabner_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5)
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10)
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # num_examples = 20  
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20)
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 


#elif method == "tfidf":
    #if stratified == "false":
        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 1 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_1_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)    
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_fabner_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


#elif method == "tfidf_half":
    #if stratified == "false":
        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_5_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_10_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/fabner_not_stratified_input_30_example_20_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)


    #else: # stratified == True
        with open("data/immutable_data_formal/sen_fabner_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_stratified_input_30_example_20_tfidf_half.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 5
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_5_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)   
        # num_examples = 10
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_10_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)
        # num_examples = 20
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        with open("formal_eva_results_llama2chat7b/fabner_stratified_input_30_example_20_tfidf_half.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)        
#else: ## 'random', 'sentence', 'tfidf', 'tfidf_half'.
    #raise  ValueError ("You can only select dataset between 'random', 'sentence', 'tfidf', 'tfidf_half'." )
    else:
         raise  ValueError ("You can only select dataset between 'film', 'fabner', 'assembly'." )      
    
    return response    

#print(  run_inference(dataset="film", method="random", stratified=True, num_examples=5)  )
if __name__ == "__main__":
    dataset = str(sys.argv[1])
    method = str(sys.argv[2])
    stratified = str(sys.argv[3])
    num_examples = int(sys.argv[4])
    response = run_inference(dataset, method, stratified, num_examples)

    # processed_response = []
    # for item in response:
    #     dem = item[0]["generated_text"]
    #     print('\nitem[0]["generated_text"]\n', item[0]["generated_text"])
    #     dem = dem.replace("* ", "")
    #     dem = dem.split("Note: ")[0]
    #     dem = dem.split("\n\n")[0]
    #     print( '\nafter processing\n'+ dem + "\n\n" )
    #     processed_response.append ( dem)
        
    # transformed_output_random_exam20 = change_output_format_to_tokens_tags(sentences_input, processed_response)
    # transformed_solution = change_output_format_to_tokens_tags(sentences_input, demonstrations_input_solutions)
    # print(len(transformed_output_random_exam20), len(transformed_solution) )
    # df = get_evaluation_without_o(transformed_solution = transformed_solution[:3], transformed_output = transformed_output_random_exam20)    
    # print(df)
    print(response)