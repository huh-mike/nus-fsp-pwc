import json
import asyncio
import os
import dotenv

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy

dotenv.load_dotenv()


def get_urls_in_iras_updates(depth):
    # Set up headless Chrome
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')

    # Initialize the driver (make sure chromedriver is in your PATH)
    driver = webdriver.Chrome(options=chrome_options)

    base_url = "https://www.iras.gov.sg"
    latest_updates_url = f"{base_url}/latest-updates/{depth}"

    driver.get(latest_updates_url)

    links_dict = {}

    try:
        # Define the container's CSS selector
        container_selector = "#Main_C005_Col00 > div:nth-child(2) > div > div > article > section > div.eyd-listing-results__articles"

        # Wait until the container is present in the DOM
        wait = WebDriverWait(driver, 10)  # waits up to 10 seconds
        articles_container = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, container_selector))
        )

        # Optionally, wait until at least one article is located within the container
        wait.until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, f"{container_selector} article"))
        )

        # Find all article elements within the container
        articles = articles_container.find_elements(By.CSS_SELECTOR, "article")

        # Loop through each article element to extract the title and link
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

async def crawl_4_ai_contents_paragraph_break(url: str) -> dict:
    """
    Crawls the given URL using Crawl4AI and extracts the main text content
    (excluding navigation, ribbon bars, etc.), then returns a JSON-like dictionary.

    Args:
        url (str): The URL to crawl.

    Returns:
        dict: A dictionary in JSON format with the URL and its extracted text content.
              Example:
              {
                  "url": "https://example.com",
                  "text": "The extracted main content..."
              }
    """

    browser_config = BrowserConfig(headless=True, verbose=True)
    extraction_instruction = """
    Please extract the exact content of the pdf provided.
    Exclude any navigation menus, advertisements, or unrelated footer content.
    """

    llm_strategy = LLMExtractionStrategy(
        provider="openai/gpt-4o-mini",  # LLM provider
        api_token=os.getenv("OPENAI_API_KEY"),
        extraction_type="text",
        instruction=extraction_instruction,
    )

    run_config = CrawlerRunConfig(
        extraction_strategy=llm_strategy,
        cache_mode=CacheMode.BYPASS, # This allows crawler to always fetch fresh data.
    )
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url=url,
            config=run_config
        )
        output = result.extracted_content
        llm_strategy.show_usage()

        return output

# For testing the module
if __name__ == "__main__":
    urls_dict = {}

    # 5 means to crawl 5 pages (of URLs)
    for i in range(5):
        urls_dict.update(get_urls_in_iras_updates(i+1))

    # This chunk only crawls the latest one URL.
    first_key = next(iter(urls_dict))
    print(f"Now Crawling: {urls_dict[first_key]}")
    print(asyncio.run(crawl_4_ai_contents_paragraph_break(urls_dict[first_key])))