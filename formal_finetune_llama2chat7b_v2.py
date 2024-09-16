from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding, GenerationConfig, TrainingArguments, Trainer
import torch
import time
#import evaluate
import pandas as pd
import numpy as np
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig, get_peft_model
from trl import SFTTrainer

from timeit import default_timer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  #"1"  or "0,1" for multiple GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:1024"



##########################################
# load the model and tokenizer
model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model_path = "./finetuned_models/finetuned_llama-2-7b-chat_assembly_full_training_16bit"

original_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
#original_model.config.use_cache = False
#original_model.config.pretraining_tp = 1
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")


####################################
#load dataset 
train_file = "./data/immutable_data_formal/llama2chat7b_assembly_train.jsonl"
raw_dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
        },
    )
print( raw_dataset )




#print(f"Shapes of the datasets:")
#print(f"Training: {tokenized_datasets['train'].shape}, \n{tokenized_datasets['train']}")

def tokenize_function(example):
    if train_file == "./data/immutable_data_formal/llama2chat7b_thf_train.jsonl":
        start_prompt = """### instruction:
                You are an excellent linguist in the domain of thin film head technology. The task is to label the entities in the given sentence. 
                The optional entities include Component, Location, Function, EnergyFlow (Energy Flow), Attribution, Material, Effect, System, ScientificConcept (Scientific Concept),
                Shape,  Value, InfoFlow (Information Flow), Consequence, PhysicsFlow (Physics Flow), Measure, State, AnnotatorNotes.\n"""\
                """### Input:
                Input sentence: """
    elif train_file == "./data/immutable_data_formal/llama2chat7b_assembly_train.jsonl":
        start_prompt = """### instruction:
                You are an excellent linguist in assembly instructions. The task is to label the entities in the given sentence. 
                The optional entities include PART (parts), RPOS (relative positions), OPER (operations), TOOL (tools), ID (identification numbers), QTY (quantity), 
                DIM (dimensions), WGT (weights), PROP (general properties). """\
                """### Input:
                Input sentence: """
    elif train_file == "./data/immutable_data_formal/llama2chat7b_fabner_train.jsonl":
        start_prompt = """### instruction:
                You are an excellent linguist in manufacturing domain. The task is to label the entities in the given sentence.
                The optional entities include CONPRI (Concept/principles), MATE (Material), MANP (Manufacturing process), PRO (Mechanical properties), PARA (Process parameters), CHAR (Process characterization),
                MACEQ (Machine/equipment), APPL (Application), FEAT (Engineering features), ENAT (Enabling technology), MANS (Manufacturing standards), BIOP (Biomedical). """\
                """### Input:
                Input sentence: """

    end_prompt = """\nExtracted entities of the above sentence: \n
                ### Output: \n"""
    prompt = [start_prompt + input_sentence + end_prompt for input_sentence in example["input"]]
    #prompt_output = [prompt[i] + example["input"][i] for i in range(len(prompt))] 
    example['input_ids'] = tokenizer(prompt, padding="max_length", max_length = 512, truncation=True, return_tensors="pt").input_ids
    example['labels'] = tokenizer(example["output"], padding="max_length", max_length = 512, truncation=True, return_tensors="pt").input_ids
    
    return example
dataset = raw_dataset.map(tokenize_function, batched=True )
tokenized_datasets = dataset.remove_columns(['instruction', 'input', 'output'])
print(tokenized_datasets)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# dataset = raw_dataset['train']
# combined_columns = []
# print(len(dataset["instruction"]))
# for i in range(len(dataset["instruction"])):
#     row = start_prompt + dataset["input"][i] + end_prompt + dataset["output"][i]
#     combined_columns.append(row)
# dataset = dataset.add_column("text", combined_columns)    
# tokenized_datasets = dataset.remove_columns(['instruction', 'input', 'output'])
# print(tokenized_datasets)

# raw_datasets_train = raw_datasets["train"]
# combined_columns = []
# print(len(raw_datasets_train["instruction"]))
# for i in range(len(raw_datasets_train["instruction"])):
#     row = "### Instruction: " + raw_datasets_train["instruction"][i] + "### Input: " + raw_datasets_train["input"][i] + "### Output: " + raw_datasets_train["output"][i]
#     combined_columns.append(row)
# dataset = raw_datasets_train.add_column("text", combined_columns)   
# dataset = raw_datasets_train.remove_columns(['instruction', 'input', 'output'])

############################### 
#### finetuning process

#### peft model
lora_config = LoraConfig(
    r=64, # Rank
    lora_alpha=32,
    #target_modules=["q", "v"],
    lora_dropout=0.1,
    bias="none",
    #task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    task_type="CAUSAL_LM",

)

def print_number_of_trainable_model_parameters(model):
    trainable_model_params = 0
    all_model_params = 0
    for _, param in model.named_parameters():
        all_model_params += param.numel()
        if param.requires_grad:
            trainable_model_params += param.numel()
    return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

print(print_number_of_trainable_model_parameters(original_model))


# training_args = TrainingArguments(
#     output_dir=new_model,
#     learning_rate=1e-5,
#     num_train_epochs=3,
#     weight_decay=0.01,
#     logging_steps=1,
#     max_steps=1
# )
peft_training_args = TrainingArguments(
    output_dir= "./results" , #f'./peft-thin-film-training-{str(int(time.time()))}',
    auto_find_batch_size=True,
    #per_device_train_batch_size=8,
    learning_rate=1e-4, # Higher learning rate than full fine-tuning.
    weight_decay=0.001,
    num_train_epochs=1,
    logging_steps=1,
    max_steps=-1, # Number of training steps (overrides num_train_epochs)

    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    #save_steps=save_steps,
    #logging_steps=logging_steps,
    #fp16=False,
    #bf16=False,
    max_grad_norm=0.3, # Maximum gradient normal (gradient clipping)
    warmup_ratio=0.03, # Ratio of steps for a linear warmup (from 0 to learning rate)
    group_by_length=True, # Group sequences into batches with same length  # Saves memory and speeds up training considerably
    lr_scheduler_type="cosine", # Learning rate schedule
    report_to="tensorboard",
)

### start training
# trainer = Trainer(
#     model=original_model,
#     args=training_args,
#     train_dataset=tokenized_datasets['train'],
#     #eval_dataset=tokenized_datasets['validation']
# )
peft_trainer = Trainer(
    model=original_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets['train'],
    data_collator=data_collator,
    tokenizer=tokenizer,
)


start = default_timer()
peft_trainer.train()
peft_trainer.model.save_pretrained(new_model_path)
tokenizer.save_pretrained(new_model_path)
print("training time: ", default_timer() - start)


############################### Short evaluation

## load the pretrained model
# below is the original_model and tokenizer
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

peft_model = AutoModelForCausalLM.from_pretrained (new_model_path)
#lora_config = LoraConfig.from_pretrained(new_model_path)
#peft_model = get_peft_model(original_model, lora_config)
print(print_number_of_trainable_model_parameters(peft_model))


## show the results:
index = 88
input = raw_dataset['train'][index]['input']
baseline_extracted_entities = raw_dataset['train'][index]['output']

if train_file == "./data/immutable_data_formal/llama2chat7b_thf_train.jsonl":
    prompt = f"""
        You are an excellent linguist in the domain of thin film head technology. The task is to label the entities in the given sentence.
        The optional entities include Component, Location, Function, EnergyFlow (Energy Flow), Attribution, Material, Effect, System, ScientificConcept (Scientific Concept),
        Shape,  Value, InfoFlow (Information Flow), Consequence, PhysicsFlow (Physics Flow), Measure, State, AnnotatorNotes.
        Input sentence: {input}
        Extracted entities of the above sentence: \n """
elif train_file == "./data/immutable_data_formal/llama2chat7b_assembly_train.jsonl":
    prompt = f"""
        You are an excellent linguist in assembly instructions. The task is to label the entities in the given sentence. 
        The optional entities include PART (parts), RPOS (relative positions), OPER (operations), TOOL (tools), ID (identification numbers), QTY (quantity), 
        DIM (dimensions), WGT (weights), PROP (general properties). 
        Input sentence: {input}
        Extracted entities of the above sentence: \n """\

elif train_file == "./data/immutable_data_formal/llama2chat7b_fabner_train.jsonl":
    start_prompt = """### instruction:
        You are an excellent linguist in manufacturing domain. The task is to label the entities in the given sentence.
        The optional entities include CONPRI (Concept/principles), MATE (Material), MANP (Manufacturing process), PRO (Mechanical properties), PARA (Process parameters), CHAR (Process characterization),
        MACEQ (Machine/equipment), APPL (Application), FEAT (Engineering features), ENAT (Enabling technology), MANS (Manufacturing standards), BIOP (Biomedical). 
        Input sentence: {input}
        Extracted entities of the above sentence: \n """\

input_ids = tokenizer(prompt, return_tensors="pt").input_ids

# original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300, num_beams=1))
# original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)

peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=300, num_beams=1))
peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

dash_line = '-'.join('' for x in range(100))
print(dash_line)
print(f'BASELINE HUMAN SUMMARY:\n{baseline_extracted_entities}')
# print(dash_line)
# print(f'ORIGINAL MODEL:\n{original_model_text_output}')
print(dash_line)
print(f'PEFT MODEL: {peft_model_text_output}')