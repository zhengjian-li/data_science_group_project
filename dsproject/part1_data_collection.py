""" Data Collection Module """
import json
import time
from typing import NamedTuple
import requests
import wikipedia
import wptools
from bs4 import BeautifulSoup
from SPARQLWrapper import JSON
from SPARQLWrapper import SPARQLWrapper
from tqdm import tqdm
import os
import warnings

class DataCollectorConfig(NamedTuple):
    """
    DataCollectorConfig defines the configuration for a data collection process.

    Attributes:
        category (str): The category of data to be collected, 'sculptor' or 'computer_scientist'.
        data_limt (int): The maximum number of data items to be collected.
        language (str): The language of the data to be collected, e.g., 'en' for English.
    """

    category: str
    data_limit: int
    language: str

def get_bio_kg(config: DataCollectorConfig, sleep_time:int = 65) -> None:
    """
    Get and save the biography and the Knowledge Graph of people in a given category

    Args:
        config (DataCollectorConfig): The configuration for the data collection process.
        sleep_time (int): The time to sleep between requests in seconds.
    
    Raises:
        ValueError: If the category is not 'sculptor' or 'computer_scientist'.
    """

    # Check the category
    try:
        assert config.category in ['sculptor', 'computer_scientist']
    except AssertionError as e:
        raise ValueError("category must be either 'sculptor' or 'computer_scientist'") from e
    
    name_ids = get_name_id(config) # Get the list of names

    for name, page_id in tqdm(name_ids, desc="Collecting data"):
        name_camel = name.replace(" ", "")
        category_camel = config.category.replace("_", " ").title().replace(" ","") + "s"
        # Get and save the biography
        
        wikipedia.set_lang(config.language)
        try:
            page = wikipedia.page(pageid=page_id) # Try to get the page by page ID
        except:
            try:
                page = wikipedia.page(name) # Try to get the page by name
            except:
                name = wikipedia.search(name +' '+ config.category)[0] # If the name is not found, search for it
                page = wikipedia.page(name) # Get the page by the new name
        name = page.title
        summary = page.summary #only the first paragraph, for the whole article use page.content
        with open(f'data/part1/{name_camel}_{category_camel}_{config.language}.txt',
                  mode='w+', encoding="utf-8") as f:
            f.write(summary)
        # Get and save the triples of the Knowledge Graph
        triples = get_kg_triples(name, config.language)
        with open(f'data/part1/{name_camel}_{category_camel}_{config.language}.json',
                  mode='w+', encoding="utf-8") as f:
            json.dump(triples, f)

        time.sleep(sleep_time) # Sleep for 60 seconds to avoid being blocked

def get_not_en_title(language: str, title:str) -> str:
    """
    Get the title of the Wikipedia page in a given language.
    
    Args:
        language (str): The language code for the desired language.
        title (str): The title of the Wikipedia page in English.
    
    Returns:
        str: The title of the Wikipedia page in the desired language.
    
    Raises:
        ValueError: If the language is not found.
    """
    new_title = None # The title in the desired language

    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "titles": title,
        "prop": "langlinks",
        "format": "json",
    }
    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    # Get the language links
    langlinks = list(DATA['query']['pages'].values())[0]['langlinks']
    
    # Find the title in the desired language
    for link in langlinks:
        if link['lang'] == language:
            new_title = link['*']
            break
    
    if not new_title: # If the language is not found
        raise ValueError('No language found')
    
    return new_title

def get_name_id(config: DataCollectorConfig) -> list:
    """
    Get the list of names of people in a given category from Wikipedia
    
    Args:
        config (DataCollectorConfig): The configuration for the data collection process.

    Returns:
        list: The list of names and their Wikipedia page IDs, e.g., [(name, page_id), ...]
    """

    # The mapping from the category to the title of the Wikipedia page
    mapping = {'sculptor': 'List of sculptors', 'computer_scientist': 'List of computer scientists'}
    # Map the category to the title of the Wikipedia page
    title = mapping[config.category]
    
    # Get the title in the desired language if it is not English
    title = get_not_en_title(config.language, title) if config.language != 'en' else title

    # The URL of the Wikipedia page
    url = f"https://{config.language}.wikipedia.org/wiki/{title}"

   
    # Use BeautifulSoup to get the list of names
    page= requests.get(url=url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
    soup= BeautifulSoup(page.content, 'html.parser')
    body_content = soup.find_all(id='bodyContent') #get the body content
    lis = body_content[0].find_all('li')#get the list items

    results = [] # The list of names
    for li in tqdm(lis, desc="Getting names"):
        links = li.find_all('a')
        if not links: # If the list item does not contain a link
            continue
        
        title = links[0].get('title')
            
        if not title: # If the list item does not contain a title
            continue
        # Add the first title (the Name) to the list
        idinqury_url = f"https://{config.language}.wikipedia.org/w/api.php?action=query&format=json&titles={title}"
        data = requests.get(idinqury_url, timeout= 10).json()
        page_id = list(data['query']['pages'].keys())[0]
        results.append((title, page_id))

        # Break if the number of names reach the data limit
        if len(results) >= config.data_limit:
            break

    return results


def get_kg_triples(name:str, language:str)-> dict:
    """
    Get the triples of the Knowledge Graph of a person from Wikidata

    Returns:
        dict: The triples of the Knowledge Graph.
    """
    #  Get the Wikidata
    page = wptools.page(name, lang=language)
    page.get_wikidata(show=False)
    wikidata = page.data['wikibase'] 
    # Get the Knowledge Graph
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    query_string = f"""DESCRIBE wd:{wikidata}"""
    sparql.setQuery(query_string)
    sparql.setReturnFormat(JSON)
    try:
        triples = sparql.query().convert()
    except:
        time.sleep(100) # Sleep for 100 seconds to avoid being blocked
        triples = sparql.query().convert()

    return triples