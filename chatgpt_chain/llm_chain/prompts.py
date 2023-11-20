from langchain.prompts import PromptTemplate

# prompt_template = """You are professional customer service of Otoritas Jasa Keuangan (OJK) Indonesia. You help customers by answering their question in concise manner. 

# Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Helpful Answer:"""
prompt_template = """You are professional customer service of Chain Link. You help people by answering questions in concise manner. 
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.


{context}

Question: {question}

"""
CUSTOM_STUFF_ANSWER_PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
