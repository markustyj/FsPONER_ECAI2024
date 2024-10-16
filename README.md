# FsPONER -- ECAI2024
FsPONER: Few-shot Prompt Optimization for Named Entity Recognition in Domain-specific Scenarios

Link to the paper: https://arxiv.org/abs/2407.08035

#### The optimized prompt structure.
<figure>
  <div style="text-align: center;">
  <img src=prompt_structure.png alt="Description" width="420">
  <figcaption></figcaption>
</figure>


## folder - data 
***data/assembly_dataset***, ***data/fabNER***, ***data/thin-film-technology-dataset*** store the original data of the three industrial datasets

***data/immutable_data_formal*** stores the corresponding few-shot examples for each input sentence in the test dataset

## folder - eva_results
contain the generated completions from LLMs, based on the proposed few-shot prompting methods

## folder - gpt_api_codes
the code to set up the OpenAI LLMs and construct the prompt with selected few-shot examples 

Please check the clean script for few-shot selection methods (random, embedding-based, TFIDF-based) in https://github.com/markustyj/FsPONER_ECAI2024/blob/main/few_shot_list_creation.py

## requirements
The requirements: please see ***requirements_finetune_llama2.txt*** and ***requirements_gpt_prompting_hf38.txt***

## Some notes
notebooks with ***eva_*** prefix are evaluation results --> F1 score, precision, recall

notebooks with ***formal_finetune_*** are the scripts to fine-tune LLaMA 2

notebooks with ***get_results_*** are the scripts to get completions from LLaMA 2-chat, Vicuna...

<br>

### A short overview of evaluation results
<figure>
  <div style="text-align: center;">
  <img src=eva_overview.png alt="Description" width="666">
  <figcaption></figcaption>
</figure>
