from pathlib import PurePath
import json 


def load_index_url(index):
    url_json_file = PurePath(f"chatgpt_chain/url_index/{index}.json")

    with open(url_json_file) as f:
        urls = json.load(f)["urls"]

    return urls
