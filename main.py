from ChatBotGUI import run_chatbotgui
from DataScraper import get_datascraped_depth
from DataTagger import create_embeddings_and_get_relevant_tags
from db import upload_to_mongo
if __name__ == "__main__":
    # Scrape Data from the first x pages:
    raw_article_dataset = get_datascraped_depth(1)

    # Tag each of the Data entries using vector similarity
    tagged_article_dataset = create_embeddings_and_get_relevant_tags(raw_article_dataset)


    upload_to_mongo(tagged_article_dataset)

    print(tagged_article_dataset)
    run_chatbotgui()