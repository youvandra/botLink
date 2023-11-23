from flask import Flask, Response, request
from dotenv import load_dotenv
import json
import os
import requests
import time
import uuid


from langchain.callbacks.manager import get_openai_callback
from chatgpt_chain.llm_chain import get_bot_chain, update_bot_chain

load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')



chain = get_bot_chain(
    index="chain_link_index", 
    embeddings="openai",
    embedding_api_key=OPENAI_KEY,
    model_api_key=OPENAI_KEY,
    elasticsearch_url=ELASTICSEARCH_URL,
    history=True,
    # rescrape=True,
)



load_dotenv()
OPENAI_KEY = os.getenv('OPENAI_KEY')
ELASTICSEARCH_URL = os.getenv('ELASTICSEARCH_URL')

app = Flask(__name__)

@app.route('/')
def main():
    response = Response(
        json.dumps(
            {
                "webhook-health": "ok"
            }
        ),
        status=200,
        mimetype='application/json'
    )
    return response

    
@app.route('/bot_interaction', methods=['POST'])
def webhook_openai_docs():
    # Get Message Send
    data = request.get_json()

    try:

        # Bot config payload
        query = data['query']
        user_id = data["user_id"]
    except:
        response = Response(
            json.dumps(
                {
                    "error": "invalid json"
                }
            ),
            status=400,
            mimetype='application/json'
        )
        return response

    try:
        with get_openai_callback() as cb:
            chain_response = chain(query, user_id=user_id)
        bot_response = {
            "recipient_id": user_id,
            "text": chain_response,
            "metadata": {
                "openai-api": {
                    "created":int(time.time()),
                    "model": chain.get_model_name(),
                    "usage": {
                        "prompt_tokens": cb.prompt_tokens,
                        "completion_tokens": cb.completion_tokens,
                        "total_tokens": cb.total_tokens,
                        "total_cost": cb.total_cost,
                    }
                }
            }
        }
    except requests.exceptions.RequestException:
        bot_response = {
            "recipient_id": sender_id,
            "text": "(ID) Mohon maaf sedang terjadi gangguan, silahkan coba kembali beberapa saat lagi. (EN) We apologize for the interruption, please try again in a few moments."
        }

    response = Response(
        json.dumps([bot_response]),
        status=200,
        mimetype='application/json'
    )

    return response

  
# @app.route('/openai_docs/update_embeddings', methods=['POST'])
# def update_embeddings():
#     """
#     Update document embeddings
#     """
#     # Get Message Send
#     data = request.get_json()

#     try:
#         # Bot config payload
#         index = data['bot_config']['index']
#         ner_index = data['bot_config']['ner_index']
#         embeddings = data['bot_config']['embeddings']
#     except:
#         response = Response(
#             json.dumps(
#                 {
#                     "error": "invalid json"
#                 }
#             ),
#             status=400,
#             mimetype='application/json'
#         )
#         return response

#     try:
#         chain = chain_loader.update_embedding(index=index, ner_index=ner_index, embeddings=embeddings)
#         bot_response = {
#             "text": 'Sukses update embedding dokumen',
#             "bot_config": data['bot_config']
#         }
#     except requests.exceptions.RequestException:
#         bot_response = {
#             "text": "(ID) Mohon maaf sedang terjadi gangguan, silahkan coba kembali beberapa saat lagi. (EN) We apologize for the interruption, please try again in a few moments."
#         }

#     response = Response(
#         json.dumps([bot_response]),
#         status=200,
#         mimetype='application/json'
#     )

#     return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6099, debug=False)