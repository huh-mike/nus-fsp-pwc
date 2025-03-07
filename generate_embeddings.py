"""
Script to generate embeddings for the tags database.
Run this script before using the RAG system for the first time.

This script:
1. Reads the original tags database CSV
2. Generates embeddings for each tag using OpenAI's API
3. Stores embeddings in separate JSON files in 'embeddings_data' directory
4. Creates a new CSV with references to these embedding files
"""

from TagEmbeddingGenerator import generate_embeddings_for_tags_database
import os
import shutil

def main():
    print("Starting embedding generation for tags database...")
    
    # Input and output file paths
    input_file = 'tags_vector_database.csv'
    output_file = 'tags_vector_database_with_embeddings.csv'
    
    # Check if the tags database file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found in the current directory.")
        print("Please ensure the tags database file is present before running this script.")
        return
    
    # Check if there's a corrupted embeddings file from previous attempts and back it up
    if os.path.exists(input_file) and os.path.getsize(input_file) > 500000:  # If file is suspiciously large
        backup_file = f"{input_file}.backup"
        print(f"The existing file {input_file} appears to be corrupted (too large).")
        print(f"Creating a backup at {backup_file}")
        shutil.copy2(input_file, backup_file)
    
    # Ensure embeddings directory exists
    embeddings_dir = "embeddings_data"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Generate embeddings and save to separate files
    generate_embeddings_for_tags_database(
        input_file=input_file, 
        output_file=output_file
    )
    
    print("Embedding generation complete.")
    print(f"1. Original database: {input_file}")
    print(f"2. Updated database with embedding references: {output_file}")
    print(f"3. Actual embeddings stored in: {embeddings_dir}/")
    print("\nYou can now run the main.py file to use the RAG system.")

if __name__ == "__main__":
    main()