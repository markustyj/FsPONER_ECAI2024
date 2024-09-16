import pandas as pd
import numpy as np
import json
from textblob import TextBlob
import os
import nltk

def read_files(path):
    data = pd.read_csv(path, sep = ' ', header=None, names = ["tokens","ner_tags"])
    return data

def read_files_assembly(path):
    data = pd.read_csv(path, sep = '\t', header=None, names = ["tokens","ner_tags"])
    return data


#read the fabner dataset to format {"tokens":[] , "ner_tags":[]}
def get_jsonl(df, dataset_path):
    with open(dataset_path, 'w', encoding='utf-8') as f:
        dic_seq = {}
        token_list = []
        ner_tags_list = []
        for index, row in df.iterrows():
            token_list.append(row["tokens"])
            ner_tags_list.append(row["ner_tags"])
            if row['tokens'] == '.':
                dic_seq["tokens"] = token_list
                dic_seq["ner_tags"] = ner_tags_list
                f.write(json.dumps(dic_seq) + "\n")
                #json.dump(dic_seq, f, ensure_ascii=False)#, indent=1)
                #print(dic_seq)
                dic_seq = {}
                token_list = []
                ner_tags_list = []


#change the entity, delete the prefix BIO
def get_jsonl_fabner_simple(df, dataset_path):
    with open(dataset_path, 'w', encoding='utf-8') as f:
        mask = df["ner_tags"] != "O"
        # eliminate the prefix in the entities
        df.loc[mask,"ner_tags"] = df[mask]["ner_tags"].apply(lambda x: x.split("-")[1])
        dic_seq = {}
        token_list = []
        ner_tags_list = []
        for index, row in df.iterrows():
            token_list.append(row["tokens"])
            ner_tags_list.append(row["ner_tags"])
            if row['tokens'] == '.':
                dic_seq["tokens"] = token_list
                dic_seq["ner_tags"] = ner_tags_list
                f.write(json.dumps(dic_seq) + "\n")
                #json.dump(dic_seq, f, ensure_ascii=False)#, indent=1)
                #print(dic_seq)
                dic_seq = {}
                token_list = []
                ner_tags_list = []


def get_jsonl_fabner_simple_with_textblob(df, dataset_path):
    """ 1) transform the csv dataframe to json 
        2) eliminate the prefix B I O 
        3) use TextBlob to separate the sentences"""
    
    with open(dataset_path, 'w', encoding='utf-8') as f:
        mask = df["ner_tags"] != "O"
        df.loc[mask,"ner_tags"] = df[mask]["ner_tags"].apply(lambda x: x.split("-")[1])
        dic_seq = {}
        tokens = ""
        token_list = []
        ner_tags_index = 0
        ner_tags_list = []
        for index, row in df.iterrows():
            # consider all tokens into the string context
            tokens = tokens + row["tokens"] + " " 
            # write all ner_tags in one long list
            ner_tags_list.append(row["ner_tags"])

        #separate to sentences    
        textblober = TextBlob(tokens)
        sentences = textblober.sentences    

        for sentence in sentences:
            token_list = sentence.split(" ")
            dic_seq["tokens"] = token_list
            dic_seq["ner_tags"] = ner_tags_list[ner_tags_index:(ner_tags_index+len(token_list)) ]
            f.write(json.dumps(dic_seq) + "\n")
            # update all parameters
            ner_tags_index = ner_tags_index + len(token_list)
            dic_seq = {}
            token_list = []
            
              

def get_original_sentences_from_tokens_nertags(path_in, path_out):
    with open(path_in, 'r', encoding='utf-8') as f:
        sentence_list = []
        entities_list = []
        for line in f:
            tokens_ner_tags = json.loads(line)
            sentence = ""
            entity = ""
            for item in tokens_ner_tags["tokens"]:
                sentence = sentence + item + " "
            sentence_list.append(sentence)

    with open(path_out, 'w', encoding='utf-8') as e:                
        for sentence in sentence_list:
            e.write(sentence + '\n')   

def get_entities_from_tokens_nertags(path_in, path_out):
    #read the json file and change to dictionary
    with open(path_in,'r', encoding='utf-8') as f:
        tokens_tags = []
        for line in f:
            token_tag = json.loads(line) 
            tokens_tags.append(token_tag)
    #tokens_tags = tokens_tags[:-1]  
    with open(path_out,'w', encoding='utf-8') as f:
        for line in tokens_tags:
            words = ""
            words_tags_list = ""
            #words_tags_list = []
            for index, tag in enumerate(line["ner_tags"]):
                # take entity, if it's not "O"
                if tag != "O":
                    if index > 0 and line["ner_tags"][index] == line["ner_tags"][index-1]: # index>0 -> index not out of range
                        words = words + " " +  line["tokens"][index]
                    else:
                        words = line["tokens"][index]

                    # the next tag different from the current one        
                    if index < len(line["ner_tags"])-1: # make sure index not out of range
                        if line["ner_tags"][index] != line["ner_tags"][index+1] :
                            words_tags_list =  words_tags_list    +   (words + ": " + line["ner_tags"][index]) + "\n" 
                    
                    # the last token is always full stop sign -> not even enter the if tag != "O":
                    # thus, the next line of code is meaningless
                    else:   
                        words_tags_list = words_tags_list   +   (words + ": " + line["ner_tags"][index])  + "\n"
                        
            if words_tags_list == "":
                words_tags_list = "no entity in this sentence \n"
                
            f.write(  (words_tags_list) + "\n"  )             


# we do not use the requirement dataset
def get_jsonl_requirement(path_input, path_output):
    requirement_train = pd.read_json(path_input, lines = True)
    requirement_train.drop(columns = ["id","ner_tags"], inplace = True)
    requirement_train.rename(columns = { "tags" : "ner_tags"}, inplace = True)
    with open(path_output, 'w', encoding='utf-8') as f:
        for _, row in requirement_train.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")


def get_jsonl_assembly(df, dataset_path):
    with open(dataset_path, 'w', encoding='utf-8') as f:
        dic_seq = {}
        token_list = []
        ner_tags_list = []
        for index, row in df.iterrows():
            token_list.append(row["tokens"])
            ner_tags_list.append(row["ner_tags"])
            if row['tokens'] == '0':
                # change "0" to "."
                token_list[-1] = "."
                dic_seq["tokens"] = token_list
                dic_seq["ner_tags"] = ner_tags_list
                f.write(json.dumps(dic_seq) + "\n")
                #json.dump(dic_seq, f, ensure_ascii=False)#, indent=1)
                #print(dic_seq)
                dic_seq = {}
                token_list = []
                ner_tags_list = []            


#read the dataset paragraph by paragraph, contain some errors
def get_jsonl_thin_film_technology(path_in, path_out):

    nltk.download('punkt') 
    with open(path_out, 'w', encoding='utf-8') as e:
        for filename in filter(lambda p: p.endswith("txt"), os.listdir(path_in)):
            #initialization
            files_txt = []
            files_ann = []
            files = {}

            #for txt, get the list for all textual tokens
            filepath = os.path.join(path_in, filename)
            with open(filepath, mode='r',encoding='utf-8') as f:
                files_txt = f.read()   
                f.close()
            files["tokens"] = nltk.word_tokenize(files_txt)

            #for ann, get the correct format for annotated dataset    
            filename_ann = filename[:7] + ".ann"
            filepath_ann = os.path.join(path_in, filename_ann)
            files_ann = pd.read_csv(filepath_ann, sep = '\s+', header=None, 
                                    names = ["T-id","ner_tags","span1", "span2", "tokens1","tokens2","tokens3", "tokens4","tokens5","tokens6",  
                                             "tokens7","tokens8","tokens9","tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], 
                                    engine='python')
            files_ann = files_ann.replace([None, np.nan], "")
            files_ann['tokens'] = (files_ann['tokens1'] + " " + files_ann['tokens2'] + " " + files_ann['tokens3'] + " " + 
                                files_ann['tokens4'] + " " + files_ann['tokens5'] + " " + files_ann['tokens6'] + " " + 
                                files_ann['tokens7'] + " " + files_ann['tokens8'] + " " + files_ann['tokens9'] + " " +
                                files_ann['tokens10'] + " " + files_ann['tokens11'] + " " + files_ann['tokens12'] + " " +
                                files_ann['tokens13'] + " " + files_ann['tokens14'] + " " + files_ann['tokens15'])  
            files_ann.drop(columns = ['tokens1','tokens2','tokens3','tokens4','tokens5','tokens6','tokens7','tokens8','tokens9',
                                      "tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], inplace = True)

            # get the jsonl format that salmon can take as input
            ner_tags = []  
            for index, single_token in enumerate(files['tokens']):
                for tag_index, row in enumerate(files_ann['tokens']):
                    if single_token in row:
                        ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann
                        if single_token == ".":
                            ner_tags[-1] = "O"
                        break
                if len(ner_tags) < index+1:    
                    ner_tags.append('O')
            files['ner_tags'] = ner_tags

        
            e.write(json.dumps(files) + "\n")    

#the following 2 functions contain some errors                        
def get_jsonl_thin_film_technology_sentence(path, path_out):

    nltk.download('punkt') 
    with open(path_out, 'w', encoding='utf-8') as e:
        for filename in filter(lambda p: p.endswith("txt"), os.listdir(path)):
            #initialization
            files_txt = []
            files_ann = []
            files = {}

            #for txt, get the list for all textual tokens
            filepath = os.path.join(path, filename)
            with open(filepath, mode='r',encoding='utf-8') as f:
                files_txt = f.read()   
                f.close()
            files["tokens"] = nltk.word_tokenize(files_txt)

            #for ann, get the correct format for annotated dataset    
            filename_ann = filename[:7] + ".ann"
            filepath_ann = os.path.join(path, filename_ann)
            files_ann = pd.read_csv(filepath_ann, sep = '\s+', header=None, 
                                    names = ["T-id","ner_tags","span1", "span2", "tokens1","tokens2","tokens3", "tokens4","tokens5","tokens6",  
                                             "tokens7","tokens8","tokens9","tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], 
                                    engine='python')
            files_ann = files_ann.replace([None, np.nan], "")
            files_ann['tokens'] = (files_ann['tokens1'] + " " + files_ann['tokens2'] + " " + files_ann['tokens3'] + " " + 
                                files_ann['tokens4'] + " " + files_ann['tokens5'] + " " + files_ann['tokens6'] + " " + 
                                files_ann['tokens7'] + " " + files_ann['tokens8'] + " " + files_ann['tokens9'] + " " +
                                files_ann['tokens10'] + " " + files_ann['tokens11'] + " " + files_ann['tokens12'] + " " +
                                files_ann['tokens13'] + " " + files_ann['tokens14'] + " " + files_ann['tokens15'])  
            files_ann.drop(columns = ['tokens1','tokens2','tokens3','tokens4','tokens5','tokens6','tokens7','tokens8','tokens9',
                                      "tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], inplace = True)

            # get the jsonl format that salmon can take as input
            ner_tags = []  
            for index, single_token in enumerate(files['tokens']):
                for tag_index, row in enumerate(files_ann['tokens']):
                    if single_token in row:
                        ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann
                        if single_token == ".":
                            ner_tags[-1] = "O"
                        break
                if len(ner_tags) < index+1:    
                    ner_tags.append('O')
            files['ner_tags'] = ner_tags

            files = pd.DataFrame(files)
            dic_seq = {}
            token_list = []
            ner_tags_list = []
            for index, row in files.iterrows():
                token_list.append(row["tokens"])
                ner_tags_list.append(row["ner_tags"])
                if row['tokens'] == '.':
                    dic_seq["tokens"] = token_list
                    dic_seq["ner_tags"] = ner_tags_list
                    e.write(json.dumps(dic_seq) + "\n")
                    dic_seq = {}
                    token_list = []
                    ner_tags_list = []                    
def get_jsonl_thin_film_technology_sentence_new(path, path_out):

    nltk.download('punkt') 
    with open(path_out, 'w', encoding='utf-8') as e:
        for filename in filter(lambda p: p.endswith("txt"), os.listdir(path)):
            #initialization
            files_txt = []
            files_ann = []
            files = {}

            #for txt, get the list for all textual tokens
            filepath = os.path.join(path, filename)
            with open(filepath, mode='r',encoding='utf-8') as f:
                files_txt = f.read()   
                f.close()
            files["tokens"] = nltk.word_tokenize(files_txt)

            #for ann, get the correct format for annotated dataset    
            filename_ann = filename[:7] + ".ann"
            filepath_ann = os.path.join(path, filename_ann)
            files_ann = pd.read_csv(filepath_ann, sep = '\s+', header=None, 
                                    names = ["T-id","ner_tags","span1", "span2", "tokens1","tokens2","tokens3", "tokens4","tokens5","tokens6",  
                                             "tokens7","tokens8","tokens9","tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], 
                                    engine='python')
            files_ann = files_ann.replace([None, np.nan], "")
            '''
            files_ann['tokens'] = (files_ann['tokens1'] + " " + files_ann['tokens2'] + " " + files_ann['tokens3'] + " " + 
                                files_ann['tokens4'] + " " + files_ann['tokens5'] + " " + files_ann['tokens6'] + " " + 
                                files_ann['tokens7'] + " " + files_ann['tokens8'] + " " + files_ann['tokens9'] + " " +
                                files_ann['tokens10'] + " " + files_ann['tokens11'] + " " + files_ann['tokens12'] + " " +
                                files_ann['tokens13'] + " " + files_ann['tokens14'] + " " + files_ann['tokens15'])  
            files_ann.drop(columns = ['tokens1','tokens2','tokens3','tokens4','tokens5','tokens6','tokens7','tokens8','tokens9',
                                      "tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], inplace = True)
            '''
            # get the jsonl format that salmon can take as input
            columns_token = ['tokens1','tokens2','tokens3','tokens4','tokens5','tokens6','tokens7','tokens8','tokens9',
                                      "tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"]
            ner_tags = []  
            for index, single_token in enumerate(files['tokens']):
                #for column in columns_token:   
                for column in columns_token:
                    for tag_index, row in enumerate(files_ann[column]):
                        if single_token == row :  # only for the last one
                            ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann`
                            files_ann[column][tag_index] = ""
                            if ner_tags[-1] == "Others":
                                ner_tags[-1] = "O" 
                            break    
                    if len(ner_tags) == index+1:    
                        break                         

                if len(ner_tags) < index+1:    # if no equal string text is found in the columns of tokens
                    ner_tags.append('O')
            files['ner_tags'] = ner_tags

            files = pd.DataFrame(files)
            dic_seq = {}
            token_list = []
            ner_tags_list = []
            for index, row in files.iterrows():
                token_list.append(row["tokens"])
                ner_tags_list.append(row["ner_tags"])
                if row['tokens'] == '.':
                    dic_seq["tokens"] = token_list
                    dic_seq["ner_tags"] = ner_tags_list
                    e.write(json.dumps(dic_seq) + "\n")
                    dic_seq = {}
                    token_list = []
                    ner_tags_list = []                      



#The latest function to read thin film technology, can be improved with the same method in get_chatgpt_ner_output_format()
def get_jsonl_thin_film_technology_sentence_new_new(path_in, path_out):

    nltk.download('punkt') 
    with open(path_out, 'w', encoding='utf-8') as e:
        for filename in filter(lambda p: p.endswith("txt"), os.listdir(path_in)):
            #initialization
            files_txt = []
            files_ann = []
            files = {}

            #for txt, get the list for all textual tokens
            filepath = os.path.join(path_in, filename)
            with open(filepath, mode='r',encoding='utf-8') as f:
                files_txt = f.read()   
                f.close()
            files["tokens"] = nltk.word_tokenize(files_txt)

            #for ann, get the correct format for annotated dataset    
            filename_ann = filename[:7] + ".ann"
            filepath_ann = os.path.join(path_in, filename_ann)
            files_ann = pd.read_csv(filepath_ann, sep = '\s+', header=None, 
                                    names = ["T-id","ner_tags","span1", "span2", "tokens1","tokens2","tokens3", "tokens4","tokens5","tokens6",  
                                             "tokens7","tokens8","tokens9","tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], 
                                    engine='python')
            files_ann = files_ann.replace([None, np.nan], "")
            
            # get the jsonl format that salmon can take as input
            columns_token = ['tokens1','tokens2','tokens3','tokens4','tokens5','tokens6','tokens7','tokens8','tokens9',
                                      "tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"]
            ner_tags = []  
            for index, single_token in enumerate(files['tokens']):
                #for column in columns_token:   
                for i in range(1,15):
                    column = "tokens" + str(i)
                    column2 = "tokens" + str(i+1)
                    for tag_index, row in enumerate(files_ann[column]):
                        row = row.strip(".")
                        if index < len(files["tokens"])-1: ## not reaching the last token of the original sentence 
                            if files_ann[column2][tag_index] != "":
                                if single_token == row and files_ann[column2][tag_index] == files['tokens'][index+1]:
                                    ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann`
                                    files_ann[column][tag_index] = "" # after finding the word in the original sentence, set it to '' to avoid double assignment
                                    if ner_tags[-1] == "Others":
                                        ner_tags[-1] = "O" 
                                    break
                            else:  ## the second word of a phase of a certain entity is not empty, 
                                if single_token == row :  #not consider the second word in the condition
                                    ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann`
                                    files_ann[column][tag_index] = ""
                                    if ner_tags[-1] == "Others":
                                        ner_tags[-1] = "O" 
                                    break    
                        else: ## have reached the last token of the original sentence 
                            if single_token == row :  # only for the last one
                                ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann`
                                files_ann[column][tag_index] = ""
                                if ner_tags[-1] == "Others":
                                    ner_tags[-1] = "O" 
                                break    
                    if len(ner_tags) == index+1:    # if the word is found, do not go though all token columns
                        break                         

                if len(ner_tags) < index+1:    # if no equal string text is found in the columns of tokens
                    ner_tags.append('O')
            files['ner_tags'] = ner_tags

            files = pd.DataFrame(files)
            dic_seq = {}
            token_list = []
            ner_tags_list = []
            for index, row in files.iterrows():
                token_list.append(row["tokens"])
                ner_tags_list.append(row["ner_tags"])
                if row['tokens'] == '.':
                    dic_seq["tokens"] = token_list
                    dic_seq["ner_tags"] = ner_tags_list
                    e.write(json.dumps(dic_seq) + "\n")
                    dic_seq = {}
                    token_list = []
                    ner_tags_list = []         
   

# same method as get_jsonl_thin_film_technology_sentence_new_new, but use the textblob() to separate the sentences
# The effect is even not so good as without textblob(), because we use nltk to get tokens of the sentence, 
# , so some examples like e.g. will not be separated by "."
def get_jsonl_thin_film_technology_version2_textblob(path_in, path_out):
    nltk.download('punkt') 
    with open(path_out, 'w', encoding='utf-8') as e:
        for filename in filter(lambda p: p.endswith("txt"), os.listdir(path_in)):
            #initialization
            files_txt = []
            files_ann = []
            files = {}

            #for txt, get the list for all textual tokens
            filepath = os.path.join(path_in, filename)
            with open(filepath, mode='r',encoding='utf-8') as f:
                files_txt = f.read()   
                f.close()
            files["tokens"] = nltk.word_tokenize(files_txt)

            #for ann, get the correct format for annotated dataset    
            filename_ann = filename[:7] + ".ann"
            filepath_ann = os.path.join(path_in, filename_ann)
            files_ann = pd.read_csv(filepath_ann, sep = '\s+', header=None, 
                                    names = ["T-id","ner_tags","span1", "span2", "tokens1","tokens2","tokens3", "tokens4","tokens5","tokens6",  
                                             "tokens7","tokens8","tokens9","tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"], 
                                    engine='python')
            # too many empty columns for tokens, replace None value or np.nan value with ""
            files_ann = files_ann.replace([None, np.nan], "")

            # get the jsonl format that salmon can take as input
            columns_token = ['tokens1','tokens2','tokens3','tokens4','tokens5','tokens6','tokens7','tokens8','tokens9',
                                      "tokens10","tokens11","tokens12","tokens13","tokens14","tokens15"]
            ner_tags = []  
            for index, single_token in enumerate(files['tokens']):
                #for column in columns_token:   
                for i in range(1,15):
                    column = "tokens" + str(i)  # string value of tokens1, tokens2 ...
                    column2 = "tokens" + str(i+1)
                    for tag_index, row in enumerate(files_ann[column]):
                        row = row.strip(".")
                        if index < len(files["tokens"])-1: ## not reaching the last token of the original sentence 
                            if files_ann[column2][tag_index] != "":
                                if single_token == row and files_ann[column2][tag_index] == files['tokens'][index+1]:
                                    ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann`
                                    files_ann[column][tag_index] = "" # after finding the word in the original sentence, set it to '' to avoid double assignment
                                    if ner_tags[-1] == "Others":
                                        ner_tags[-1] = "O" 
                                    break
                            else:  ## the second word of a phase of a certain entity is not empty, 
                                if single_token == row :  #not consider the second word in the condition
                                    ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann`
                                    files_ann[column][tag_index] = ""
                                    if ner_tags[-1] == "Others":
                                        ner_tags[-1] = "O" 
                                    break    
                        else: ## have reached the last token of the original sentence 
                            if single_token == row :  # only for the last one
                                ner_tags.append(files_ann['ner_tags'][tag_index])  # assign the tag to ner_tags according to files_ann`
                                files_ann[column][tag_index] = ""
                                if ner_tags[-1] == "Others":
                                    ner_tags[-1] = "O" 
                                break    
                    if len(ner_tags) == index+1:    # if the word is found, do not go though all token columns
                        break                         

                if len(ner_tags) < index+1:    # if no equal string text is found in the columns of tokens
                    ner_tags.append('O')
            files['ner_tags'] = ner_tags

            df = pd.DataFrame(files)
            dic_seq = {}
            tokens = ""
            token_list = []
            ner_tags_index = 0
            ner_tags_list = []
            for index, row in df.iterrows():
                # consider all tokens into the string context
                tokens = tokens + row["tokens"] + " " 
                # write all ner_tags in one long list
                ner_tags_list.append(row["ner_tags"])

            #separate to sentences    
            textblober = TextBlob(tokens)
            sentences = textblober.sentences   
            
            for sentence in sentences:
                token_list = sentence.split(" ")
                dic_seq["tokens"] = token_list
                dic_seq["ner_tags"] = ner_tags_list[ner_tags_index:(ner_tags_index+len(token_list)) ]
                e.write(json.dumps(dic_seq) + "\n")
                # update all parameters
                ner_tags_index = ner_tags_index + len(token_list)
                dic_seq = {}
                token_list = []  


#get the input sentence for GPT (not the entire paragraph) from the dataset in format {"tokens":[] , "ner_tags":[]},
def get_short_sentence(path_in, path_out):
    """ get the input sentence for GPT, not the entire paragraph
    input: jsonl, {"tokens":  , "ner_tags": }
    output: string short sentence """
    #read the json file and change to dictionary
    with open(path_in,'r', encoding='utf-8') as f:
        tokens_tags = []
        for line in f:
            token_tag = json.loads(line) 
            tokens_tags.append(token_tag)
    #tokens_tags    
    with open(path_out,'w', encoding='utf-8') as f:
        for line in tokens_tags:
            for index, token in enumerate(line["tokens"]):
                if index == 0:
                    words_tags_list = token 
                elif token == "." or token ==",":    
                    words_tags_list = words_tags_list + token 
                else:
                    words_tags_list = words_tags_list + " " + token    
            f.write(  (words_tags_list) + "\n"  )    


