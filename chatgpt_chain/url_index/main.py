from pathlib import PurePath
from collections import defaultdict 
import json 

from chatgpt_chain.scrapper import get_child_url


def load_index_url(index):
    url_json_file = PurePath(f"chatgpt_chain/url_index/{index}.json")

    with open(url_json_file) as f:
        urls = json.load(f)["urls"]

    final_scrapped_urls = []

    for url_detail in urls:
        url = url_detail['url']
        child_url_identifier = url_detail.get("child_url_identifier", None)
        subdomain = url_detail.get("subdomain", None)
        limit = url_detail.get("limit", None)


        if child_url_identifier is None or subdomain is None:
            final_scrapped_urls.append(url)
        else:
            child_urls = get_child_url(
                url=url, 
                subdomain=subdomain,
                child_url_identifier=child_url_identifier
            )

            if limit is not None:
                child_urls = child_urls[:limit]
            final_scrapped_urls += child_urls        

    final_scrapped_urls = list(set(final_scrapped_urls))
    print(final_scrapped_urls)
    print("============ len all urls:{} ========= ".format(len(final_scrapped_urls)))


    return final_scrapped_urls
