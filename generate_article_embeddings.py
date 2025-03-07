"""
Script to generate summaries and embeddings for the articles database.
This is a separate step from generating tag embeddings, as it involves using GPT
to create concise summaries of articles before embedding them.

This script:
1. Reads the scraped articles database
2. For each article:
   a. Generates a concise summary using GPT-4o-mini
   b. Creates an embedding for the summary
   c. Stores the embedding in a separate JSON file
3. Saves a new CSV with summaries and references to embedding files
"""

from ArticleEmbeddingGenerator import generate_embeddings_for_articles
import os
import shutil

def main():
    print("Starting summary and embedding generation for articles database...")
    
    # Input and output file paths
    input_file = 'scraped_data.csv'
    output_file = 'articles_with_embeddings.csv'
    embeddings_dir = 'article_embeddings'
    
    # Check if the articles database file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found in the current directory.")
        print("Please ensure the articles database file is present before running this script.")
        return
    
    # Ensure embeddings directory exists
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Generate summaries and embeddings
    generate_embeddings_for_articles(
        input_file=input_file,
        output_file=output_file,
        embeddings_dir=embeddings_dir
    )
    
    print("\nProcessing complete.")
    print(f"1. Original articles database: {input_file}")
    print(f"2. Enhanced database with summaries and embedding references: {output_file}")
    print(f"3. Article embeddings stored in: {embeddings_dir}/")
    print("\nNow you need to update the RAGServices.py to use these article embeddings for similarity search.")

if __name__ == "__main__":
    main()