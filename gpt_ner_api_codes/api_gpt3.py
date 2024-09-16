# Note: The openai-python library support for Azure OpenAI is in preview.
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://openai-aiattack-000518-eastus-01.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key =    "1f4b80860e134ab4be99a0766dbd8813" # os.getenv("OPENAI_API_KEY") #


def gpt_complete(prompt_text ):
    
    openai.api_type = "azure"
    openai.api_base = "https://openai-aiattack-000518-eastus-01.openai.azure.com/"
    openai.api_version = "2022-12-01"
    openai.api_key =    "1f4b80860e134ab4be99a0766dbd8813" # os.getenv("OPENAI_API_KEY") #
    # does not have access to ChatCompletion as well
    response = openai.Completion.create(  
        engine='text-davinci-003',
        prompt= prompt_text,
        # engine = 'gpt-3.5-turbo-0613',  have no access to this with this account
        # messages=[
        #     {"role": "system", "content": "You are a helpful assistant."},
        #     {"role": "user", "content": "how are you"}
        # ],
        temperature=0,
        max_tokens=200,
        top_p=0.1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    # print(response)
    choices = response["choices"]
    if len(choices) > 0:
        result = choices[0]
        text = result["text"]
        return text
    else:
        return None

