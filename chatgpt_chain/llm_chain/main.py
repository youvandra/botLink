from langchain.vectorstores import ElasticVectorSearch
from langchain.llms import OpenAI, HuggingFaceHub

from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain.embeddings.openai import OpenAIEmbeddings

from gptcache.adapter.langchain_models import LangChainLLMs
from gptcache import cache

from langchain.document_loaders import SeleniumURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from chatgpt_chain.llm_chain.prompts import CUSTOM_STUFF_ANSWER_PROMPT
from chatgpt_chain.url_index import load_index_url


import requests
import json
import logging
import tqdm


def format_history(messages):
    """Format message history into chatgpt chain format
    
    The chain receives messages in pairs (1 user message & 1 assistant message). In order to
    do that, We will append every consecutive user maessage into 1 message.
    Args:
        messages (dict): message history
    """
    formatted_message = []
    message_pair = {"user": "",
                    "assistant": ""}
    
    for message in messages:
        if message['role'] == "user":
            message_pair["user"] += "\n" + message["content"]
        elif message["role"] == "assistant":
            message_pair["assistant"] += message["content"]
            formatted_message.append((message_pair["user"], message_pair["assistant"]))
            
            # reset message pair
            message_pair = {"user": "",
                            "assistant": ""}
    
    return formatted_message


### LLMs loader

def get_openai_llm(temperature=0, model_name="text-davinci-003", api_key=None):
    logging.info(f"Loading OpenAI LLM (temperature = {temperature})")
    llm = OpenAI(
        model_name=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        max_tokens=1000,
    )
    return llm

def get_huggingface_llm(model_name="google/flan-t5-base", api_key=None):
    logging.info(f"Loading HuggingFace LLM (model_name = {model_name})")
    llm = HuggingFaceHub(repo_id=model_name, huggingfacehub_api_token=api_key, model_kwargs={"temperature": 0.9, "max_length": 2048})
    return llm


# Embedding Models Loader
def get_huggingface_embeddings():
    logging.info(f"Loading Huggingface Embedding")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings


def get_openai_embeddings(api_key):
    logging.info(f"Loading OpenAI Embedding")
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    return embeddings


# Chain Loaders
def load_answer_generator(llm, prompt=CUSTOM_STUFF_ANSWER_PROMPT):
    logging.info(f"Loading answer gnerator")
    answer_generator = load_qa_chain(llm, verbose=False) #fg19 verbose True
    answer_generator.llm_chain.prompt = prompt
    return answer_generator


def load_question_generator(llm, prompt=CONDENSE_QUESTION_PROMPT):
    logging.info(f"Loading question gnerator")
    question_generator = LLMChain(llm=llm, prompt=prompt)
    return question_generator


# Documents Loader
def load_documents(source):
    logging.info(f"Loading documents (source = {source})")
    loader = TextLoader(source, encoding="utf8")
    documents = loader.load()
    return documents


def load_urls(index):

    urls = load_index_url(index=index)

    logging.info(f"Loading urls (urls = {urls})")
    loader = SeleniumURLLoader(urls=urls)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 700,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )

    documents = text_splitter.split_documents(documents)
    return documents


def split_documents(documents, chunk_size=1000, chunk_oeverlap=0):
    logging.info(f"Splitting documents")
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_oeverlap
    )
    documents = text_splitter.split_documents(documents)
    return documents


def get_vectorstore(index_name, embeddings, elasticsearch_url=None):
    logging.info(f"Loading vectorstore")
    db = ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name=index_name,
        embedding=embeddings,
    )
    return db

# get the content(only question) form the prompt to cache
def get_content_func(data, **_):
    return data.get("prompt").split("Question")[-1]


def create_vectorstore(
    documents,
    index_name,
    embeddings,
    elasticsearch_url=None,
):
    logging.info(f"Creating vectorstore")

    db = ElasticVectorSearch(
        elasticsearch_url=elasticsearch_url,
        index_name=index_name,
        embedding=embeddings,
    )

    # Remove existing embedding indices
    db.client.indices.delete(
        index=index_name,
        ignore_unavailable=True,
        allow_no_indices=True,
    )

    num_chunks = len(documents)
    logging.info(f"{num_chunks} chunks of documents are loaded")

    db.add_documents(
        documents, 
        bulk_kwargs={
            "chunk_size": 50000,
            "max_chunk_bytes": 200000000
        })
    return db


def get_conversational_chain(db, question_generator, answer_generator):
    logging.info(f"Loading conversational qa chain")

    qa = ConversationalRetrievalChain(
        retriever=db.as_retriever(),
        question_generator=question_generator,
        combine_docs_chain=answer_generator,
        # verbose=True # fg19
    )

    return qa

def get_retrieval_chain(db, answer_generator):
    logging.info(f"Loading retrieval chain")

    qa = RetrievalQA(combine_documents_chain=answer_generator, retriever=db.as_retriever(), verbose=True)

    return qa

def update_bot_chain(index, embeddings, embedding_api_key=None, elasticsearch_url=None):
    """ Rescrape documents and update embedding store """

    
    if embeddings == "huggingface":
        embeddings = get_huggingface_embeddings(embedding_api_key)
    elif embeddings == "openai":
        embeddings = get_openai_embeddings(embedding_api_key)

    # documents = load_pdf_document(index=index, ner_index=ner_index)
    documents = load_urls(index)
    logging.INFO("============ lendoc len documents {}".format(len(documents)))

    db = create_vectorstore(
        documents,
        index,
        embeddings=embeddings,
        elasticsearch_url=elasticsearch_url,
    )

def get_bot_chain(index, embeddings, embedding_api_key, model_api_key, elasticsearch_url, history=False):
    # cache.init(pre_embedding_func=get_msg_func)
    # cache.set_openai_key(openai_api_key)

    llm = get_openai_llm(
        model_name="text-davinci-003", temperature=0, api_key=model_api_key
    )
    # llm = get_huggingface_llm(model_name="MBZUAI/LaMini-Flan-T5-248M", api_key=model_api_key)

    # llm = LangChainLLMs(llm)

    if embeddings == "huggingface":
        embeddings = get_huggingface_embeddings()
    elif embeddings == "openai":
        embeddings = get_openai_embeddings(embedding_api_key)

    question_generator = load_question_generator(llm)
    answer_generator = load_answer_generator(llm)

    db = get_vectorstore(
        index,
        embeddings=embeddings,
        elasticsearch_url=elasticsearch_url,
    )

    if history:
        qa = get_conversational_chain(db, question_generator, answer_generator)
    else:
        qa = get_retrieval_chain(db, answer_generator)

    chain = CustomLLMChain(
        chain=qa,
        model_name="text-davinci-003",
        use_history=history
    )
    

    return chain

class CustomLLMChain:
    def __init__(self, chain, model_name, use_history):
        self.chain = chain
        self.model_name = model_name
        self.use_history = use_history
        self.chat_history = []
        
    def __call__(self, text):
        if self.use_history:
            response = self.chain(({"question": text, "chat_history": self.chat_history}))
            response = response['answer']
            self.chat_history.append((text, response))
        else:
            response = self.chain(text)
        return response 

    def get_model_name(self):
        return self.model_name