from dotenv import load_dotenv
import os, re

from chatgpt_chain.llm_chain import get_bot_chain, update_bot_chain

import logging

# logging.getLogger().setLevel(logging.INFO) #fg19

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')


# chain = update_bot_chain(
#     index="chain_link_index", 
#     embeddings="openai",
#     embedding_api_key=OPENAI_KEY,
#     elasticsearch_url=ELASTICSEARCH_URL
# )

import time 

t0 = time.time()

chain = get_bot_chain(
    index="chain_link_index", 
    embeddings="openai",
    embedding_api_key=OPENAI_KEY,
    model_api_key=OPENAI_KEY,
    elasticsearch_url=ELASTICSEARCH_URL,
    history=True,
    # rescrape=True,
)



t1 = time.time() - t0
print('chain load time', t1)


for i in range(10):
    text = input("\nUser:")
    chain_response = chain(text)

    print("\nResponse:", chain_response)