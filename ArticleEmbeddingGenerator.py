import pandas as pd
import numpy as np
import json
import os
import ast
from typing import List, Dict, Any
from GPTServices import gpt_generate_embedding, gpt_generate_single_response

def create_article_summary(title: str, text: str, max_tokens: int = 3000) -> str:
    """
    Create a concise summary of an article using GPT-4o-mini.
    
    Args:
        title: Article title
        text: Article text content
        max_tokens: Maximum tokens for the summary
        
    Returns:
        Summarized article content
    """
    # Parse text content if it's stored as a dictionary string
    article_content = text
    if isinstance(text, str):
        try:
            text_dict = ast.literal_eval(text)
            if isinstance(text_dict, dict) and 'content' in text_dict:
                article_content = text_dict['content']
        except (ValueError, SyntaxError):
            article_content = text
    
    # Create system prompt for summary generation
    system_prompt = (
        "You are a tax information summarizer. Given an article title and content, "
        "create a concise summary that captures the key tax information. "
        f"Keep your summary under {max_tokens} tokens. Focus on factual information "
        "that would be most relevant for tax-related queries."
    )
    
    # Construct user prompt with title and content
    user_prompt = f"Title: {title}\n\nContent: {article_content}\n\nPlease summarize this tax article."
    
    # Generate summary using GPT
    try:
        summary = gpt_generate_single_response(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model="gpt-4o-mini",
            temperature=0.3,
            token_limit=max_tokens
        )
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        # Return a truncated version of the original content as fallback
        return article_content[:2000] + "..."

def generate_embeddings_for_articles(input_file: str = 'scraped_data.csv',
                                    output_file: str = 'articles_with_embeddings.csv',
                                    embeddings_dir: str = 'article_embeddings'):
    """
    Generate summaries and embeddings for articles in the database.
    
    Args:
        input_file: Path to the input CSV file containing article data
        output_file: Path to save the CSV file with summaries and embedding references
        embeddings_dir: Directory to store embedding JSON files
    """
    print(f"Loading articles database from {input_file}...")
    
    # Create embeddings directory if it doesn't exist
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Read the CSV file
    try:
        # Ensure all columns are read as strings to avoid type issues
        df = pd.read_csv(input_file, dtype=str)
        total_rows = len(df)
        print(f"Loaded {total_rows} articles from {input_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Add summary column if it doesn't exist
    if 'summary' not in df.columns:
        df['summary'] = None
        
    # Add embedding_reference column if it doesn't exist
    if 'embedding_reference' not in df.columns:
        df['embedding_reference'] = None
    
    # Find rows that need processing
    rows_to_process = []
    for idx, row in df.iterrows():
        needs_summary = pd.isna(row.get('summary')) or row.get('summary') == ''
        needs_embedding = pd.isna(row.get('embedding_reference')) or row.get('embedding_reference') == ''
        
        if needs_summary or needs_embedding:
            rows_to_process.append(idx)
    
    if not rows_to_process:
        print("All articles already have summaries and embeddings. No work needed.")
        return
    
    print(f"Processing {len(rows_to_process)} articles...")
    
    # Process each article
    for idx in rows_to_process:
        title = df.loc[idx, 'title']
        text = df.loc[idx, 'text']
        
        # Generate summary if needed
        if pd.isna(df.loc[idx, 'summary']) or df.loc[idx, 'summary'] == '':
            print(f"Generating summary for article {idx+1}/{total_rows}: {title}")
            summary = create_article_summary(title, text)
            df.at[idx, 'summary'] = summary
        else:
            summary = df.loc[idx, 'summary']
        
        # Generate embedding if needed
        if pd.isna(df.loc[idx, 'embedding_reference']) or df.loc[idx, 'embedding_reference'] == '':
            print(f"Generating embedding for article {idx+1}/{total_rows}")
            
            # Combine title and summary for embedding
            embedding_text = f"{title}\n\n{summary}"
            
            try:
                # Generate embedding
                embedding = gpt_generate_embedding(embedding_text)
                
                # Save embedding to a JSON file
                embedding_file = f"{embeddings_dir}/article_{idx}.json"
                with open(embedding_file, 'w') as f:
                    json.dump(embedding, f)
                
                # Store reference to embedding file
                df.at[idx, 'embedding_reference'] = f"article_{idx}.json"
                
                print(f"  → Embedding saved to {embedding_file}")
            except Exception as e:
                print(f"Error generating embedding for article {idx+1}: {e}")
    
    # Save the updated dataframe
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully saved articles with summaries and embedding references to {output_file}")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    generate_embeddings_for_articles()