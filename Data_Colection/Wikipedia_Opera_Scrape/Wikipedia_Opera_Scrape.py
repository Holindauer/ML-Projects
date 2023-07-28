import requests
from bs4 import BeautifulSoup
import os

'''
This script scrapes the Wikipedia category page for opera, and saves the text of each article in the category to a text file.
It does not handle subcategories, so it only scrapes the articles in the main category page.
'''

def get_links(soup):
    # Find the part of the page where the links are
    # WP stores categoriy links in a div (a div is a section of a page) with id="mw-pages"
    content_div = soup.find("div", id="mw-pages")

    # html 'a' elements represent links - find and extract them into links variable
    links = content_div.find_all("a")
    return links


def get_articles(links):
    article_text = {}

    # Loop through all the links
    for link in links:
        # The href attribute of the link is the end of the Wikipedia URL -- specifying the specific page
        href = link.get('href')

        # Concatenate the base Wikipedia URL with this href
        # This is the URL for this page
        url = "https://en.wikipedia.org" + href

        # Get the page content
        r = requests.get(url)
        page_soup = BeautifulSoup(r.content, 'html.parser')

        # Print the title of the page, which is contained in the title tag
        article_title = page_soup.title.string

        paragraphs = page_soup.find_all('p')
        
        collect_paragraphs = [] # list to store paragraphs
        for p in paragraphs: # loop through paragraphs
            p_graph = p.get_text() # get text
            collect_paragraphs.append(p_graph) # add to list
        article = ' '.join(collect_paragraphs) # join list into unified string

        article_text[article_title] = article #add article text to dictionary, with title as key

    return article_text

def clean_title(title):
    # list of invalid characters for Windows file names
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']

    # replace each invalid character with an underscore
    for char in invalid_chars:
        title = title.replace(char, '_')
    return title

def save_articles(article_text):
    os.chdir("C:\\Users\\hunte\\OneDrive\\Documents\\Coding Projects\\Web Scraping") # change directory to new directory
    print(f"current working directory: {os.getcwd()}")

    dir_name = "Opera Articles" # name of directory to store articles
    os.mkdir(dir_name) # make directory to store articles
    print(f"saving articles in {os.getcwd()}")
    os.chdir(dir_name) # change directory to new directory

    for title, article in article_text.items():
        clean_title_name = clean_title(title) # clean title before creating file
        file_name = clean_title_name + '.txt'
        with open(file_name, 'w', encoding='utf-8') as f: # open file in write mode with utf-8 encoding to handle special characters
            f.write(article)

def main():
    # make GET request to main category page for opera, store its contents in soup object
    print("making request to Wikipedia...")
    r = requests.get("https://en.wikipedia.org/wiki/Category:Opera")
    soup = BeautifulSoup(r.content, 'html.parser')

    print("getting links...")
    links = get_links(soup) # get links from soup object

    print("getting articles...")
    article_text = get_articles(links) # get article text from links --> dictionary

    print("saving articles...")
    save_articles(article_text) # save articles to directory

if __name__ == "__main__":
    main()