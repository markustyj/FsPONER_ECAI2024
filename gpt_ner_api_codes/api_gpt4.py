
import os
import openai

# {
#   "deployment_name": "gpt-4-32k",
#   "openai_api_key": "d702a80d85b64465b5c40d726d95fc88",
#   "openai_api_base": "https://openai-aiattack-000518-canadaeast-02-smrattack.openai.azure.com/",
#   "openai_api_version": "2023-07-01-preview"
# }

openai.api_type = "azure"
#openai.api_base = "https://smrattack-openai-msa000898-francecentral-04.openai.azure.com/"
openai.api_base = "https://openai-aiattack-000518-canadaeast-02-smrattack.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
#openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_key = "d702a80d85b64465b5c40d726d95fc88"


def gpt_complete(prompt_text):
    
    openai.api_type = "azure"
    openai.api_base = "https://openai-aiattack-000518-canadaeast-02-smrattack.openai.azure.com/"
    openai.api_version = "2023-07-01-preview"
    openai.api_key = "d702a80d85b64465b5c40d726d95fc88"

    response = openai.ChatCompletion.create(
        #engine="gpt-4",
        deployment_id = "gpt-4-32k",
        #prompt= prompt_text,
        messages=[
            # {"role": "system", "content": "You are a helpful assistant."},
            {"role": "system", "content": "You are an excellent linguist in the domain of thin film head technology. The task is to label the entities in the given sentence. "},
            {"role": "user", "content": "{prompt_text}".format(prompt_text=prompt_text) }
        ],
        temperature=0.1,
        max_tokens=200,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    choices = response["choices"]

    if len(choices) > 0:
        result = choices[0]
        text = result["message"]["content"]
        #print("###################")
        #print(text)
        return text
        # if "Los Angeles Dodgers" in text:
        #     assert True
        # else:
        #     assert False
    else:
        assert False
   
