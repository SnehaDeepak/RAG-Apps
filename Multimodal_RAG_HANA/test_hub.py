from gen_ai_hub.proxy.langchain.openai import OpenAIEmbeddings
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

proxy_client = get_proxy_client('gen-ai-hub')

#embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002', proxy_client=proxy_client)

#emd = embedding_model.embed_query('Every decoding is another encoding.')

#call without passing proxy_client

#embedding_model = OpenAIEmbeddings(proxy_model_name='text-embedding-ada-002')

#response = embedding_model.embed_query('Every decoding is another encoding.')
#print("****** embeddings ********")
#print(emd)

#from langchain.chains import LLMChain
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from gen_ai_hub.proxy.langchain.google_gemini import ChatGoogleGenerativeAI
import PIL.Image

chat_model = ChatGoogleGenerativeAI(proxy_model_name='gemini-1.0-pro')

'''template = 'You are a helpful assistant that translates english to pirate.'
system_message_prompt = SystemMessagePromptTemplate.from_template(template)
example_human = HumanMessagePromptTemplate.from_template('Hi')
example_ai = AIMessagePromptTemplate.from_template('Ahoy!')
human_template = '{text}'
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_human, example_ai, human_message_prompt])

chain = LLMChain(llm=chat_model, prompt=chat_prompt)
response = chain.invoke('I love programming')
print("&&&&&&&&&&& LLM response &&&&&&&&&&&&")
print(response)'''

img = PIL.Image.open('fetched_images/frame_4249142809_2af1437517504cf3b9a748973e61cce7-290524-1446-554.pdf.png')
response = chat_model.invoke(["illustrate the  image"],stream=False)
#response.resolve()
print(response)