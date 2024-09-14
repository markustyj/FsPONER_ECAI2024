from salmon.pre_training.pre_train_masked_lm import transformer_masked_lm_pretrain
from salmon.downstream.text_classification_finetune import transformer_finetune_textcat_downstream
from salmon.downstream.ner_finetune_noprefix import transformer_finetune_ner_downstream
#from salmon.downstream.ner_finetune import transformer_finetune_ner_downstream
from salmon.models import Model

"""
################# pretraining
ppl_before, ppl_after = transformer_masked_lm_pretrain(

    train_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/lex_glue/train_lm_scotus_small.jsonl',
    test_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/lex_glue/test_lm_scotus_small.jsonl',
    #train_file='/home/z004r5cc/Documents/master_thesis/requirements-ner-id/requirement_train.jsonl',
    #test_file='/home/z004r5cc/Documents/master_thesis/requirements-ner-id/requirement_test.jsonl',

    base_model='sshleifer/tiny-distilroberta-base',
    output_model='models/cli_lm_test_pythonapi',
    logging_dir='logs_cli_test_pythonapi',
    tokenizer_model='distilroberta-base',
    num_train_epochs=10,
    save_strategy='no'
)

print(f'Perplexity before: {ppl_before}')
print(f'Perplexity after:  {ppl_after}')

"""

"""
################ classification
from salmon.downstream.text_classification_finetune import transformer_finetune_textcat_downstream

eval_before, eval_after = transformer_finetune_textcat_downstream(
        train_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/lex_glue/train_ds_tc_scotus_small.jsonl',
        val_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/lex_glue/val_ds_tc_scotus_small.jsonl',
        test_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/lex_glue/test_ds_tc_scotus_small.jsonl',
        base_model='models/cli_lm_test_pythonapi',
        output_model='models/cli_lm_test_ds_pythonapi',
        logging_dir='logs_cli_test_pythonapi',
        tokenizer_model='distilroberta-base',
        save_strategy='no',
        num_train_epochs=1,
        training_data_fraction=1.0
    )

print(eval_before)
print(eval_after)

"""


################ NER
eval_before, eval_after = transformer_finetune_ner_downstream(

        #train_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/un_ner/un_train_ner_small.jsonl',
        #val_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/un_ner/un_val_ner_small.jsonl',
        #test_file='/home/z004r5cc/Documents/master_thesis/salmon_v025/examples/data/un_ner/un_test_ner_small.jsonl',

        #train_file='/home/z004r5cc/Documents/master_thesis/fabNER/fabner_simple_train.jsonl',
        #val_file='/home/z004r5cc/Documents/master_thesis/fabNER/fabner_simple_val.jsonl',
        #test_file='/home/z004r5cc/Documents/master_thesis/fabNER/fabner_simple_test.jsonl',
        
        #train_file='/home/z004r5cc/Documents/master_thesis/al-nlp-req,requirements-ner-id/requirement_train.jsonl',
        #val_file='/home/z004r5cc/Documents/master_thesis/al-nlp-req,requirements-ner-id/requirement_val.jsonl',
        #test_file='/home/z004r5cc/Documents/master_thesis/al-nlp-req,requirements-ner-id/requirement_test.jsonl',
        
        #train_file = ("/home/z004r5cc/Documents/master_thesis/Assembly-Stanford-NER/dataset/alternators-engines-gearboxes/train.jsonl"),
        #val_file = ("/home/z004r5cc/Documents/master_thesis/Assembly-Stanford-NER/dataset/alternators-engines-gearboxes/val.jsonl"),
        #test_file = ("/home/z004r5cc/Documents/master_thesis/Assembly-Stanford-NER/dataset/alternators-engines-gearboxes/test.jsonl"),

        train_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/train.jsonl"),
        val_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/val.jsonl"),
        test_file = ("/home/z004r5cc/Documents/master_thesis/thin-film-processed/finetune_entity_model/test.jsonl"),

        #base_model='models/cli_lm_test_pythonapi',
        base_model='"/home/z004r5cc/Documents/models/thin-film-finetuned"',
        #output_model='models/thin-film-finetuned',
        logging_dir='logs_ner_test_pythonapi',
        tokenizer_model='distilroberta-base',
        save_strategy='no',
        num_train_epochs=5,
        training_data_fraction=1.0,
        tokenizer_add_prefix_space = True
    )

print(eval_before)
print(eval_after)
