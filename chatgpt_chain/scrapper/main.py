import requests
import re
from bs4 import BeautifulSoup

# Replace 'url_of_documentation' with the actual URL of the documentation site

def get_child_url(url, subdomain, child_url_identifier):

    # Send a GET request to the documentation site
    response = requests.get(url)

    # Parse the HTML content of the page
    soup = BeautifulSoup(response.text, 'html.parser')

    matches = []

    for identifier in child_url_identifier:
        # Use a regular expression to extract URLs from href attributes
        # url_pattern = re.compile(r'href=["\'](?!.*remix\.ethereum\.org)(https?://\S*docs\.chain\.link\S+?)["\']')
        url_pattern = re.compile(r'href=["\']({}\S+?)["\']'.format(identifier))

        # If too much, exclude links that has parents= parameters
        # Example: /resources/contributing-to-chainlink?parent=chainlinkFunctions

        # Find all matches in the HTML content
        matches += url_pattern.findall(response.text)

    # remove parameters (/resources/contributing-to-chainlink?parent=chainlinkFunctions -> /resources/contributing-to-chainlink)
    # this should remove duplicated urls
    urls = list(set(match.split('?')[0] for match in matches))

    # add subdomain (/resources/contributing-to-chainlink -> https://docs.chain.link/resources/contributing-to-chainlink)
    urls = [subdomain + url for url in urls]
    return urls


if __name__ == "__main__":
    urls = get_child_url(
        url='https://docs.chain.link/chainlink-automation', 
        subdomain='https://docs.chain.link',
        child_url_identifier=['/resources', '/chainlink-automation', '/architecture-overview']
    )

    urls = get_child_url(
        url="https://chain.link/press", 
        subdomain= "",
        child_url_identifier=[ "https://"]
    )



    # Print the list of URLs
    for url in urls:
        print(url)