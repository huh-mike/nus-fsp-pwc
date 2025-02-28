### ISSUES: WHEN CRAWLING ALL THE URLS, SOME OF THEM ARE PDF, USE PDF PROCESSOR. For some of the websites they contain mainly pdfs, use pdf processor to process them. For some urls, there are mainly pdfs, use pdf processor to process all the urls. after the main process.

import json
import asyncio
import os
import dotenv
import requests
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

from PdfProcessor import download_pdf as dp
from PdfProcessor import extract_text_from_pdf as ep

from GPTServices import gpt_generate_single_response
from db import upload_to_mongo  # Import upload function

dotenv.load_dotenv()


def get_urls_in_iras_updates(depth):
    '''
    :param depth: number of pages to crawl.
    :return: Dictionary that contains the titles with their urls.
    '''
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    driver = webdriver.Chrome(options=chrome_options)

    base_url = "https://www.iras.gov.sg"
    latest_updates_url = f"{base_url}/latest-updates/{depth}"
    driver.get(latest_updates_url)

    links_dict = {}
    try:
        container_selector = "#Main_C005_Col00 > div:nth-child(2) > div > div > article > section > div.eyd-listing-results__articles"
        wait = WebDriverWait(driver, 10)
        articles_container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, container_selector)))
        wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, f"{container_selector} article")))

        articles = articles_container.find_elements(By.CSS_SELECTOR, "article")
        for article in articles:
            try:
                a_tag = article.find_element(By.CSS_SELECTOR, "section.eyd-article-item__text > h3 > a")
                title = a_tag.text.strip()
                link = a_tag.get_attribute("href")
                if title and link:
                    links_dict[title] = link
            except Exception as inner_err:
                print("Error extracting data from an article:", inner_err)
    except Exception as e:
        print("Error finding the articles container:", e)
    finally:
        driver.quit()

    return links_dict


async def crawl_4_ai_contents_paragraph_break(url):
    browser_config = BrowserConfig(headless=True, verbose=True)
    extraction_instruction = "Please extract the text contents of the articles. Exclude navigation menus, advertisements, or unrelated footer content."

    llm_strategy = LLMExtractionStrategy(
        provider="openai/gpt-4o-mini",
        api_token=os.getenv("OPENAI_API_KEY"),
        extraction_type="text",
        instruction=extraction_instruction,
    )

    run_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS,
    )

    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(url=url, config=run_config)
        llm_strategy.show_usage()
        return {"url": url, "text": result.extracted_content}


def get_scraped_data_with_pages(depth):
    '''

    Added GPT Summarizations

    :param depth: number of pages to scrape
    :return: scraped dataset (untagged)
    '''

    def is_pdf_link(input_url):
        try:
            response = requests.head(input_url, allow_redirects=True, timeout=5)
            content_type = response.headers.get('Content-Type', '')
            return 'application/pdf' in content_type
        except requests.RequestException:
            return False

    urls_dict = {}
    for i in range(depth):
        urls_dict.update(get_urls_in_iras_updates(i))

    scraped_data = []

    for title, url in urls_dict.items():
        print(f"Crawling: {url}")

        if is_pdf_link(url):
            try:
                extracted_text = ep(dp(url,title))
                system_prompt = (
                    "By analyzing the given content downloaded from a pdf, generate the actual content of it in plain English. Only use the original text written in the content given. "
                    "Do not include formatting elements, structural placeholders that do not contribute meaning."
                    "The output should be just a regular string with paragraphing."
                    "Ensure there's no summarization or any variation of the original text.")
                processed_text = gpt_generate_single_response(user_prompt=extracted_text,system_prompt=system_prompt,model="gpt-4o-mini",temperature=0,token_limit=16000)
                scraped_data.append({"title": title, "url": url, "text": processed_text, "tags": ["PDF"]})
                continue

            except Exception as e:
                print(f"Error processing the following pdf: {url}, {e}")
                continue

        content = asyncio.run(crawl_4_ai_contents_paragraph_break(url))

        system_prompt = ("By analyzing the given content, generate the actual content of the website article in plain English. Only use the original text written in the website. Do not include formatting elements, structural placeholders that do not contribute meaning."
                         "If there is a link to a secondary page or a PDF that is directly relevant to the main content of the article, list those links under the key 'urls'. If no such links exist, 'urls' should contain an empty set. "
                         "Please ensure the urls are in a correct format by detecting and extracting the actual target URLs with identifying any embedded URLs inside angle brackets (<>). If an embedded URL is found, return only that valid URL. Otherwise, return the original URL."
                         "For example: for an url like https://www.iras.gov.sg/news-events/singapore-budget/<https:/www.mof.gov.sg/docs/librariesprovider3/budget2025/download/pdf/annexb2.pdf>, you should return https:/www.mof.gov.sg/docs/librariesprovider3/budget2025/download/pdf/annexb2.pdf only."
                         "The output should be in JSON format with two keys:"
                         "	•	'content': article content in plain English."
                         "  •	'urls': A list of only the relevant external links."
                         "Ensure that only the core, meaningful content of the article is retained.")

        cleaned_text = gpt_generate_single_response(user_prompt=content["text"],system_prompt=system_prompt,model="gpt-4o-mini",temperature=0,token_limit=16000)

        if isinstance(cleaned_text, str):
            try:
                cleaned_text = json.loads(cleaned_text)  # Convert JSON string to list
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                cleaned_text = []  # Default to an empty list if parsing fails

        # Check and crawl pdf files within a webpage
        if cleaned_text:
            try:
                urls = cleaned_text["urls"]
                pdf_content = []
                for url in urls:
                    if is_pdf_link(url):
                        filename = urlparse(url).path.split("/")[-1]
                        filename_without_ext = os.path.splitext(filename)[0]
                        extracted_text = ep(dp(url, filename_without_ext))
                        system_prompt = (
                            "By analyzing the given content downloaded from a pdf, generate the actual content of it in plain English. Only use the original text written in the content given. "
                            "Do not include formatting elements, structural placeholders that do not contribute meaning."
                            "The output should be just a regular string with paragraphing."
                            "Ensure there's no summarization or any variation of the original text.")
                        processed_text = gpt_generate_single_response(user_prompt=extracted_text,
                                                                      system_prompt=system_prompt, model="gpt-4o-mini",
                                                                      temperature=0, token_limit=16000)
                        pdf_content.append(processed_text)
                        print(f"pdf appended: {url}")
                    else:
                        print(f"skipped (not a pdf): {url}")
                        continue
                cleaned_text["pdf"] = pdf_content

            except KeyError as e:
                print(f"ERROR processing pdf from webpage: {e}")


        # Dummy Tags: will replace later
        scraped_data.append({"title": title, "url": url, "text": cleaned_text, "tags": ["IRAS"]})


    # Here is to parse the string list format into actual list format.

    print(scraped_data)
    upload_to_mongo(scraped_data, "FSPDatabase",
                    "untaggeddatabase")

    for item in scraped_data:
        text_data = item.get("text", "")

        if isinstance(text_data, str):
            try:
                parsed_text = json.loads(text_data)
            except json.JSONDecodeError:
                parsed_text = text_data
        else:
            parsed_text = text_data

        if isinstance(parsed_text, list):
            flattened_text = " ".join(
                " ".join(filter(None, inner_list))  # Remove None values
                if isinstance(inner_list, list) else str(inner_list)
                for inner_list in parsed_text
            )
        else:
            flattened_text = str(parsed_text)

        item["text"] = flattened_text

    # Print the modified JSON
    print(scraped_data)
    return scraped_data


    # upload_to_mongo(scraped_data, "TaggedDatabase", "TaggedCollection")
    # print("Data uploaded successfully.")


if __name__ == "__main__":
    get_scraped_data_with_pages(1)