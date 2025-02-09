
'''
def main():
    print("Scraping IRAS Latest Updates...\n")
    recent_updates = scrape_latest_updates("01 Jan 2025")

    if not recent_updates:
        print("No updates were found. Please check the CSS selectors or the page structure.")
        return

    # Display the scraped data for debugging purposes
    print("Scraped Data:")
    for update in recent_updates:
        print(f"Title: {update['title']}")
        print(f"Date: {update['date']}")
        print(f"Link: {update['link']}\n")

    print("Sending data to GPT‑4o for analysis...\n")
    analysis = analyze_with_gpt4(recent_updates)

    print("GPT‑4o Analysis:\n")
    print(analysis)
'''


if __name__ == "__main__":
    pass

