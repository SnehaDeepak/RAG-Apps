import time
import os
import shutil
from bs4 import BeautifulSoup
from langchain_community.document_loaders import ConfluenceLoader
from langchain_core.documents import Document
from atlassian import Confluence

def scrape_wiki_page(space_key, loaded_wiki):
    # max limit is 100. We need to loop over the start values.
    limit = 100
    start = 0
     
    all_pages = []
     
    while True:
        pages = loaded_wiki.get_all_pages_from_space(
            space_key,
            start,
            limit,
            status=None,
            expand="body.storage",
            content_type='page'
        )
 
        if not pages:
            break
         
        all_pages += pages
        start += limit
 
    return all_pages

def get_all_documents(all_pages, wiki_url):
    documents = []
 
    for page in all_pages:
        content = page["body"]["storage"]["value"]
        text = BeautifulSoup(content, "html").get_text(" ", strip=True)
 
        if len(text) > 0:
            documents.append(
                Document(
                    metadata={
                        "title": page["title"],
                        "source": wiki_url + page["_links"]["webui"],
                        "pg_id": page['id']
                    },
                    page_content=text,
                )
            )
 
    return documents

def get_all_space():
    wiki_url = 'URL'
    personal_access_token = ''
    
    # Initialize the Confluence client
    loaded_wiki = Confluence(
        url=wiki_url,
        token=personal_access_token
    )
    spaces = loaded_wiki.get_all_spaces(start=0, limit=1000, expand=None)
    slist = spaces['results']
    space_keys, space_names=[],[]
    for s in slist:
        space_keys.append(s['key'])
        space_names.append(s['name'])
    #print(space_keys)
    #print(space_names)
    return space_keys

def main(space_key):
    wiki_url = 'URL'
    personal_access_token = 'n'
    
    # Initialize the Confluence client
    loaded_wiki = Confluence(
        url=wiki_url,
        token=personal_access_token
    )

    all_pages = scrape_wiki_page(space_key, loaded_wiki)
    pg_data = get_all_documents(all_pages, wiki_url)

    shutil.rmtree(space_key, ignore_errors=True)
    os.makedirs(space_key, exist_ok=True)

    for i in range(0,2): #no. of pages to be downloaded from wiki space
        document = pg_data[i]
        title = document.metadata['title']
        id = document.metadata['pg_id']
        #page_title='VEG 7.1 Integration'
        #space = '3DVE'
        page_id = loaded_wiki.get_page_id(space_key, title)
        content = loaded_wiki.export_page(page_id=id)
        
        with open(os.path.join(space_key, title) + ".pdf", "wb") as pdf_file:
            pdf_file.write(content)
            pdf_file.close()
            print("Completed")

#main(space_key='3DVE')