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

from db import upload_to_mongo  # Import upload function

dotenv.load_dotenv()


def get_urls_in_iras_updates(depth):
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
    extraction_instruction = "Please extract the exact content of the PDF provided. Exclude navigation menus, advertisements, or unrelated footer content."

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
        return {"url": url, "text": result.extracted_content}


def main():
    urls_dict = get_urls_in_iras_updates(1)  # Crawl the first page of updates
    scraped_data = []

    for title, url in urls_dict.items():
        print(f"Crawling: {url}")
        content = asyncio.run(crawl_4_ai_contents_paragraph_break(url))
        scraped_data.append({"title": title, "url": url, "text": content["text"], "tags": ["IRAS", "tax update"]})

    upload_to_mongo(scraped_data, "TaggedDatabase", "TaggedCollection")
    print("Data uploaded successfully.")


if __name__ == "__main__":
    main()
