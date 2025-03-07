import csv
import pandas as pd
import numpy as np
import json
import os
from GPTServices import gpt_generate_embedding
from typing import List, Dict, Any

def generate_embeddings_for_tags_database(input_file: str = 'tags_vector_database.csv', 
                                         output_file: str = 'tags_vector_database_with_embeddings.csv'):
    """
    Generates embeddings for each row in the tags database and saves them to a new CSV file.
    
    Args:
        input_file: Path to the input CSV file containing tag data
        output_file: Path to save the CSV file with embeddings
    """
    print(f"Loading tags database from {input_file}...")
    
    # Read the CSV file
    try:
        # Explicitly set dtype to ensure the Embeddings column is treated as object type
        df = pd.read_csv(input_file, dtype={'Embeddings': object})
        total_rows = len(df)
        print(f"Loaded {total_rows} rows from {input_file}")
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return
    
    # Create a new column for embeddings if it doesn't exist
    if 'Embeddings' not in df.columns:
        # Explicitly create column as object type to prevent the FutureWarning
        df['Embeddings'] = pd.Series(dtype=object)
        
    # Find rows where embeddings are empty
    embeddings_to_generate = df[df['Embeddings'].isna() | 
                               (df['Embeddings'] == '') | 
                               (df['Embeddings'].astype(str) == 'nan')].index.tolist()
    
    if not embeddings_to_generate:
        print("All rows already have embeddings. No work needed.")
        return
    
    print(f"Generating embeddings for {len(embeddings_to_generate)} rows...")
    
    # Create a separate directory to store raw embeddings
    embeddings_dir = "embeddings_data"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Generate embeddings for each row
    for idx in embeddings_to_generate:
        # Combine tag information into a single text to generate embedding
        tag_text = f"{df.loc[idx, 'Primary Category']} - {df.loc[idx, 'Secondary Category']} - " \
                  f"{df.loc[idx, 'Tertiary Category']} - {df.loc[idx, 'Tags Explanation']}"
        
        try:
            # Generate embedding using OpenAI API
            embedding = gpt_generate_embedding(tag_text)
            
            # Store the embedding in a separate file to avoid CSV formatting issues
            embedding_file = f"{embeddings_dir}/embedding_{idx}.json"
            with open(embedding_file, 'w') as f:
                json.dump(embedding, f)
            
            # Store a reference to the embedding file in the CSV
            df.at[idx, 'Embeddings'] = f"embedding_{idx}.json"
            
            print(f"Generated embedding for row {idx+1}/{total_rows}")
        except Exception as e:
            print(f"Error generating embedding for row {idx+1}: {e}")
    
    # Save the updated dataframe to CSV
    try:
        df.to_csv(output_file, index=False)
        print(f"Successfully saved embeddings references to {output_file}")
        print(f"Raw embeddings are stored in the '{embeddings_dir}' directory")
    except Exception as e:
        print(f"Error saving CSV file: {e}")

if __name__ == "__main__":
    generate_embeddings_for_tags_database()