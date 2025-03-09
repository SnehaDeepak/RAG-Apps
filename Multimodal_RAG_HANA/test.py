#source /Users/C5380034/Documents/LLM_ACCESS_SAP/.venv/bin/activate
#from gen_ai_hub.proxy import set_proxy_version
#set_proxy_version('btp') # 'gen-ai-hub'/

'''
from gen_ai_hub.proxy.native.openai import completions

response = completions.create(
  model_name="tiiuae--falcon-40b-instruct",
  prompt="The Answer to the Ultimate Question of Life, the Universe, and Everything is",
  max_tokens=7,
  temperature=0
)
print(response)
'''
'''
from gen_ai_hub.proxy.native.openai import chat

messages = [ {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
            {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
            {"role": "user", "content": "Do other Azure Cognitive Services support this too?"} ]

kwargs = dict(model_name='gpt-35-turbo', messages=messages)
response = chat.completions.create(**kwargs)
print("**************")
print(response)'''

from gen_ai_hub.proxy.native.google.clients import GenerativeModel
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')
kwargs = dict({'model_name': 'gemini-1.0-pro'})
model = GenerativeModel(proxy_client=proxy_client, **kwargs)
content = [{
                "role": "user",
                "parts": [{
                    "text": "Write a story about a magic backpack."
                    }]
                }]
model_response = model.generate_content(content)
print(model_response.text)

