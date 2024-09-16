import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  #"1"  or "0,1" for multiple GPUs

from datasets import load_dataset
from transformers import pipeline, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding, GenerationConfig, TrainingArguments, Trainer
import torch
import time
#import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig
from timeit import default_timer
import sys
import pickle

##########################################
# load the model and tokenizer
##########################################
model_name = "NousResearch/Llama-2-7b-chat-hf"
#new_model_path = "./finetuned_models/finetuned_llama-2-7b-chat_thf_full_training_16bit"
new_model_path = "./finetuned_models/finetuned_llama-2-7b-chat_fabner_full_training_16bit"



# device_map = {
#         'transformer.word_embeddings': 0,
#         'transformer.word_embeddings_layernorm': 0,
#         'lm_head': 0,
#         'transformer.h': 0,
#         'transformer.ln_f': 0,
#         'model.embed_tokens': 0,
#         'model.layers': 0,
#         'model.norm': 0
#         }
peft_model = AutoModelForCausalLM.from_pretrained(
    new_model_path,
    torch_dtype=torch.bfloat16,
    device_map = "auto"
    )
tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                          device_map = "auto"
                                          )

#########################################
## load the pretrained model
#########################################
# peft_model_base = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.bfloat16,
#     device_map="auto"
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# peft_model = PeftModel.from_pretrained(peft_model_base, 
#                                        new_model_path, 
#                                        torch_dtype=torch.bfloat16,
#                                        is_trainable=False)
# lora_config = LoraConfig.from_pretrained(new_model_path)
# peft_model = get_peft_model(original_model, lora_config)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"
print(print_number_of_trainable_model_parameters(peft_model))


## show the results:
#index = 88
#input = dataset['train'][index]['input']
#baseline_extracted_entities = dataset['train'][index]['output']

# def get_output (prompt):
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(device_map = "auto", max_new_tokens=100, num_beams=1, return_full_text=False))
#     peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

#     return peft_model_text_output

pipe = pipeline("text-generation", model=peft_model, tokenizer = tokenizer, 
            return_full_text=False, max_new_tokens=100,  #max_length=4096,
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
            "Below is an example.\n\n"
        
    elif dataset == "fabner":
        instruction = "You are an excellent linguist in manufacturing domain. The task is to label the entities in the given sentence. " \
            "The optional entities include CONPRI (Concept/principles), MATE (Material), MANP (Manufacturing process), PRO (Mechanical properties), PARA (Process parameters), CHAR (Process characterization), "\
            "MACEQ (Machine/equipment), APPL (Application), FEAT (Engineering features), ENAT (Enabling technology), MANS (Manufacturing standards), BIOP (Biomedical). "\
            "Below is an example.\n\n"  
        
    elif dataset == "assembly":
        instruction = "You are an excellent linguist in assembly instructions. The task is to label the entities in the given sentence. " \
            "The optional entities include PART (parts), RPOS (relative positions), OPER (operations), TOOL (tools), ID (identification numbers), QTY (quantity), "\
            "DIM (dimensions), WGT (weights), PROP (general properties). "\
            "Below is an example.\n\n"    
        
    else:
        raise  ValueError ("You can only select dataset between 'film', 'fabner', 'assembly'." )   

    # for each input sentence, iterate over the index of all few shot examples
    for i in range( num_examples ):      
        sen_element = fewshot_sen_one_sentence[i]
        dem_element = fewshot_dem_one_sentence[i]
        # instruction = instruction + "Input{i}:{sen_element}".format(i=i+1, sen_element=sen_element) \
        #       + "Output{i}:\n{dem_element}".format(i = i+1, dem_element = dem_element) 
        instruction = instruction + "Example {i}:\n".format(i=i+1) + "Sentence: {sen_element}".format(sen_element=sen_element) \
              + "Extracted entities of the example sentence:\n{dem_element}".format(i = i+1, dem_element = dem_element) 
        
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

        # instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples, dataset)
        # instruction = instruction + "Input sentence: {input}".format(input = input_sen)  + "\nExtracted entities of the input sentence: \n"
        # prompt_text = instruction 
        prompt_text = "Input sentence: {input}".format(input = input_sen) + "\nExtracted entities of the above sentence: \n"

        #logger.info(prompt_text)
        
        ### get the maximun number of few-shot examples
        #enc = tiktoken.encoding_for_model("text-davinci-003")
        for i in range(100):
            num_tokens = len( tokenizer.encode(prompt_text) )
            if num_tokens > 4096 - 300:  ## consider 300 token length for output
                num_examples_new = num_examples_new - 1
                instruction = get_prompt_with_demonstration_examples( fewshot_sen_one_sentence, fewshot_dem_one_sentence, num_examples_new, dataset)
                instruction = instruction + "Input sentence: {input}".format(input = input_sen) + "\nExtracted entities of the input sentence: \n"
                prompt_text = instruction
                #print("exceed maximun token length, -{i}", i+1)
            else: 
                break
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
        if input_sen_index%20 == 0:
            print(output)

    return prompt_text , response



######################################################################################################
## load the data, run inference, save the result
######################################################################################################

def run_inference(dataset, method, stratified, num_examples):

    if dataset == "film":
        with open("data/immutable_data_formal/sen_thf_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_thf_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   

        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 
        # num_examples = 0
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 0 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data) 
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 0 )
        with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_0_random_3_epochs.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)     
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_1_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)      

#elif method == "sentence":
    #if stratified ==  "false":
        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_1_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)     
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)

#elif method == "tfidf":
    #if stratified == "false":
        with open("data/immutable_data_formal/sen_thf_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_thf_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_1_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)       
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "film", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/thf_not_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)




    elif dataset =="assembly":
        with open("data/immutable_data_formal/sen_assembly_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_assembly_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   

        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 
        # num_examples = 0
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 0 )
        with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data) 
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 0 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_0_random_3_epochs.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data) 
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_1_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)    
        # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)  
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)      
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)    


#elif method == "sentence":
#    if stratified ==  "false":
        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_1_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)    
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)

#elif method == "tfidf":
#    if stratified == "false":
        with open("data/immutable_data_formal/sen_assembly_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_assembly_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_1_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "assembly", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/assembly_not_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)





    elif dataset == "fabner":    
        with open("data/immutable_data_formal/sen_fabner_input_30.txt", "rb") as data:   #Pickling
            sentences_input = pickle.load(data)  
        with open("data/immutable_data_formal/dem_fabner_input_30.txt", "rb") as data:   #Pickling
            demonstrations_input_solutions = pickle.load(data)   

        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_random.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data) 
        # # num_examples = 0
        prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 0 )
        with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_0_random.txt", "wb") as data:   #Pickling
            pickle.dump(response, data)  
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 0 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_0_random_3_epochs.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)      
        #num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_1_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5)
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_stratified_input_30_example_5_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data) 
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10)
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_stratified_input_30_example_10_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)     
        #num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_20_random.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)          
               

        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_sentence.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_1_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_5_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_10_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_20_sentence.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)


        with open("data/immutable_data_formal/sen_fabner_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_sen = pickle.load(data)    
        with open("data/immutable_data_formal/dem_fabner_not_stratified_input_30_example_20_tfidf.txt", "rb") as data:   # Unpickling
            all_fewshot_dem = pickle.load(data)  
        # num_examples = 1
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 1 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_1_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)     
        # # num_examples = 5
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 5 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_5_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)   
        # # num_examples = 10
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 10 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_10_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)
        # num_examples = 20
        # prompt_text , response = get_direct_gpt_response(all_fewshot_sen, all_fewshot_dem, sentences_input, dataset = "fabner", num_examples = 20 )
        # with open("formal_eva_results_finetuned_llama2chat7b/fabner_not_stratified_input_30_example_20_tfidf.txt", "wb") as data:   #Pickling
        #     pickle.dump(response, data)

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