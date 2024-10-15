# FsPONER -- ECAI2024
FsPONER: Few-shot Prompt Optimization for Named Entity Recognition in Domain-specific Scenarios

Link to the paper: https://arxiv.org/abs/2407.08035


### data 
data/assembly_dataset, data/fabNER, data/thin-film-technology-dataset store the original data of the three industrial datasets

data/immutable_data_formal stores the corresponding few-shot examples for each input sentence in the test dataset

### eva_results
contain the generated completions from LLMs

### gpt_api_codes
the code to set up the OpenAI LLMs and construct the prompt with selected few-shot examples 

If the code in the 2 .py files are unreadable, please check the script for few-shot selection methods in 

### requirements
The requirements: please see requirements_finetune_llama2.txt and requirements_gpt_prompting_hf38.txt


##### notebooks with eva prefix are evaluation results
##### notebooks with formal_finetune are the scripts to fine-tune LLaMA 2
##### notebooks with get_results are the scripts to get completions from LLaMA 2-chat, Vicuna...
