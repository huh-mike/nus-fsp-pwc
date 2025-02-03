from datetime import datetime
import time
from dotenv import load_dotenv
import os
from openai import OpenAI
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

load_dotenv()

# Set-up OpenAI Client here once imported this file
openai_client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)


def scrape_latest_updates(threshold_date_str="01 Jan 2025"):
    # Convert the threshold date string to a date object
    threshold_date = datetime.strptime(threshold_date_str, "%d %b %Y").date()

    # Set up Chrome options (run headless)
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    # Initialize the driver (ensure chromedriver is installed and in your PATH)
    driver = webdriver.Chrome(options=chrome_options)

    try:
        driver.get("https://www.iras.gov.sg/latest-updates")
        time.sleep(3)  # In production, use WebDriverWait for better control

        # Find all update articles based on their specific class
        articles = driver.find_elements(By.CSS_SELECTOR, "article.eyd-article-item--updates")
        updates = []

        for article in articles:
            try:
                # Extract title and link
                title_element = article.find_element(By.CSS_SELECTOR, "section.eyd-article-item__text h3 a")
                title = title_element.text.strip()
                link = title_element.get_attribute("href")

                # Extract the publication date
                date_element = article.find_element(By.CSS_SELECTOR, "span.eyd-article-item__meta--date")
                date_str = date_element.text.strip()

                # Parse the date
                article_date = datetime.strptime(date_str, "%d %b %Y").date()

                # Check if the article meets our threshold for being "latest"
                if article_date >= threshold_date:
                    updates.append({
                        "title": title,
                        "link": link,
                        "date": article_date
                    })
            except Exception as e:
                print(f"Error processing an article: {e}")
                continue

        return updates
    finally:
        if driver:
            driver.quit()
            print("Exited with quitting driver.")
        else:
            print("No driver detected. Exited without quitting driver.")


def analyze_with_gpt4(scraped_data):
    """
    Uses GPT‑4o (via OpenAI API) to analyze the scraped data.
    Args:
        scraped_data (list): List of dictionaries with keys 'title' and 'summary'.
    Returns:
        The analysis response from GPT‑4.
    """

    # Build a prompt that includes the scraped data.
    # For a real implementation, consider summarizing or chunking if the data is too long.
    prompt = "Please analyze the following tax update information from IRAS and summarize the key points:\n\n"
    for item in scraped_data:
        prompt += f"Title: {item['title']}\nContent: {item['link']}\n\n"

    # Make the call to GPT‑4 using the ChatCompletion API.
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert analyst specialized in tax regulations."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500  # Adjust as needed for the desired response length
    )

    return response.choices[0].message.content