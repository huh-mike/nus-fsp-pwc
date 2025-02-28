import json
import asyncio
import os
import dotenv
import ast
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from sklearn.metrics.pairwise import cosine_similarity
from db import upload_to_mongo
from GPTServices import gpt_generate_embedding, gpt_generate_single_response
import numpy as np
from pymongo import MongoClient


dotenv.load_dotenv()


def get_urls_in_iras_updates(depth):
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')

    with webdriver.Chrome(options=chrome_options) as driver:
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
        return links_dict


async def crawl_4_ai_contents_paragraph_break(url):
    browser_config = BrowserConfig(headless=True, verbose=True)
    extraction_instruction = "Extract the exact content of the article. Exclude navigation menus, ads, and footers."

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


def summarize_content(content: str):
    summary_response = gpt_generate_single_response(content, system_prompt="Please summarize this text.")
    return summary_response


def scrape_and_tag_data(depth):
    urls_dict = {}
    for i in range(depth):
        urls_dict.update(get_urls_in_iras_updates(i))

    scraped_data = []
    for title, url in urls_dict.items():
        print(f"Crawling: {url}")
        content = asyncio.run(crawl_4_ai_contents_paragraph_break(url))
        summary = summarize_content(content["text"])
        scraped_data.append({"title": title, "url": url, "text": summary, "tags": []})

    tagged_data = create_embeddings_and_get_relevant_tags(scraped_data)

    for item in tagged_data:
        print(f"Title: {item['title'][:30]}... Tags: {item['tags']}")

    upload_to_mongo(tagged_data, "TaggedDatabase", "TaggedCollection")
    print("Data uploaded successfully.")

def reduce_embedding_dimension(embedding, target_dim=1536):
    embedding = np.array(embedding)

    if len(embedding) == target_dim:
        return embedding

    if len(embedding) < target_dim:
        padded = np.zeros(target_dim)
        padded[:len(embedding)] = embedding
        return padded

    if len(embedding) > target_dim:
        return embedding[:target_dim]

    return embedding

def generate_best_tags(top_n, article, tag_embeddings, valid_tags):

    article_embedding = np.array(gpt_generate_embedding(article), dtype=np.float32)

    if len(article_embedding) != 1536:
        article_embedding = reduce_embedding_dimension(article_embedding, 1536)

    similarities = {}
    for tag, tag_embedding in tag_embeddings.items():
        if len(tag_embedding) != 1536:
            tag_embedding = reduce_embedding_dimension(tag_embedding, 1536)

        article_2d = article_embedding.reshape(1, -1)
        tag_2d = tag_embedding.reshape(1, -1)

        similarity = cosine_similarity(article_2d, tag_2d)[0][0]
        similarities[tag] = similarity

    sorted_tags = sorted(similarities.items(), key=lambda x: x[1], reverse=True)

    print(f"Similarities for article: {similarities}")

    best_tags = []
    for tag, score in sorted_tags:
        if tag in valid_tags:
            best_tags.append(tag)
            if len(best_tags) >= top_n:
                break

    print(f"Best Tags for article: {article[:30]}... -> {best_tags}")
    return best_tags


def create_embeddings_and_get_relevant_tags(raw_dataset):
    try:

        df = pd.read_csv("tag_data_with_embeddings.csv")
        valid_tags = set(df["tag"])

        print(f"Number of valid tags: {len(valid_tags)}")
        print(f"Sample of valid tags: {list(valid_tags)[:5]}")

        if "embedding" not in df.columns or df["embedding"].isna().any():
            df["embedding"] = df["tag"].apply(gpt_generate_embedding)
            df["embedding"] = df["embedding"].apply(lambda x: str(x))
            df.to_csv("tag_data_with_embeddings.csv", index=False)
            print("Embeddings saved to CSV!")

        df["embedding"] = df["embedding"].apply(ast.literal_eval)
        tag_embeddings = {row["tag"]: np.array(row["embedding"], dtype=np.float32) for _, row in df.iterrows()}

        for item in raw_dataset:
            text_content = item["text"]
            print(f"\nProcessing content: {text_content[:50]}...")

            tag_results = generate_best_tags(5, text_content, tag_embeddings, valid_tags)

            item["tags"] = tag_results
            print(f"Added tags: {tag_results}")

        print(f"\nTagging complete. Sample of tagged items:")
        for i, item in enumerate(raw_dataset[:2]):
            print(f"Item {i + 1}: Title: {item.get('title', '')[:30]}... Tags: {item['tags']}")

        return raw_dataset
    except Exception as e:
        print(f"Error in create_embeddings_and_get_relevant_tags: {e}")
        return raw_dataset  # Return original dataset in case of error



def fetch_relevant_documents(user_embedding, top_n=1):
    MONGO_URI = os.getenv("MONGO_URI")
    mongo_client = MongoClient(os.getenv("MONGO_URI"),tls=True, tlsAllowInvalidCertificates=True)
    db = mongo_client["FSPDatabase"]
    collection = db["TaggedCollection"]

    try:
        # Retrieve all documents from the collection
        documents = list(collection.find({}, {"text": 1, "embedding": 1}))

        if not documents:
            return "No relevant documents found."

        # Extract embeddings and texts
        embeddings = np.array([doc["embedding"] for doc in documents])
        texts = [doc["text"] for doc in documents]

        # Compute cosine similarity
        user_embedding = np.array(user_embedding).reshape(1, -1)
        similarities = cosine_similarity(user_embedding, embeddings)[0]

        # Get top N most similar documents
        top_indices = similarities.argsort()[-top_n:][::-1]

        # Return the most relevant document
        best_match = texts[top_indices[0]]
        return best_match

    except Exception as e:
        print(f"Error fetching relevant documents: {e}")
        return "Error retrieving reference materials."

if __name__ == "__main__":
    scrape_and_tag_data(depth=1)


