import numpy as np
import pandas as pd
import ast
import os
import json
from typing import List, Dict, Any, Tuple
from GPTServices import gpt_generate_embedding

class RAGSystem:
    def __init__(self, 
                tags_database_path: str = 'tags_vector_database_with_embeddings.csv',
                articles_database_path: str = 'articles_with_embeddings.csv',
                tags_embeddings_dir: str = 'embeddings_data',
                articles_embeddings_dir: str = 'article_embeddings',
                fallback_articles_path: str = 'scraped_data.csv'):
        """
        Initialize the RAG system with tag and article databases.
        
        Args:
            tags_database_path: Path to the CSV file containing tag embeddings references
            articles_database_path: Path to the CSV file containing article embeddings
            tags_embeddings_dir: Directory containing the tag embeddings JSON files
            articles_embeddings_dir: Directory containing the article embeddings JSON files
            fallback_articles_path: Fallback path to original articles if enhanced version not found
        """
        self.tags_db_path = tags_database_path
        self.articles_db_path = articles_database_path
        self.fallback_articles_path = fallback_articles_path
        self.tags_embeddings_dir = tags_embeddings_dir
        self.articles_embeddings_dir = articles_embeddings_dir
        
        self.tags_df = None
        self.articles_df = None
        self.embeddings_cache = {}  # Cache to store loaded embeddings
        
        # Flag to indicate if we're using the enhanced articles database
        self.using_enhanced_articles = os.path.exists(articles_database_path)
        
        self.load_databases()
    
    def load_databases(self) -> None:
        """Load tag and article databases from CSV files."""
        try:
            # Set dtype for Embeddings column to avoid warnings
            self.tags_df = pd.read_csv(self.tags_db_path, dtype={'Embeddings': object})
            print(f"Loaded tags database with {len(self.tags_df)} entries")
        except Exception as e:
            print(f"Error loading tags database: {e}")
            self.tags_df = pd.DataFrame()
        
        # Try to load the enhanced articles database with embeddings first
        try:
            if self.using_enhanced_articles:
                self.articles_df = pd.read_csv(self.articles_db_path, dtype=str)
                print(f"Loaded enhanced articles database with {len(self.articles_df)} entries")
            else:
                # Fall back to the original articles database
                self.articles_df = pd.read_csv(self.fallback_articles_path, dtype=str)
                print(f"Using original articles database with {len(self.articles_df)} entries")
                print("Note: For better retrieval, run generate_article_embeddings.py")
        except Exception as e:
            print(f"Error loading articles database: {e}")
            self.articles_df = pd.DataFrame()
    
    def get_embedding_vector(self, embedding_reference: str, is_article: bool = False) -> np.ndarray:
        """
        Load embedding vector from a file reference.
        
        Args:
            embedding_reference: Reference to the embedding file
            is_article: Whether this is an article embedding (vs tag embedding)
            
        Returns:
            Embedding vector as numpy array
        """
        # Use cache key that distinguishes between article and tag embeddings
        cache_key = f"{'article' if is_article else 'tag'}:{embedding_reference}"
        
        # Check if the embedding is already in the cache
        if cache_key in self.embeddings_cache:
            return self.embeddings_cache[cache_key]
        
        try:
            # Check if the reference is a file path
            if isinstance(embedding_reference, str) and embedding_reference.endswith('.json'):
                # Choose the right embeddings directory
                embeddings_dir = self.articles_embeddings_dir if is_article else self.tags_embeddings_dir
                filepath = os.path.join(embeddings_dir, embedding_reference)
                
                if os.path.exists(filepath):
                    with open(filepath, 'r') as f:
                        embedding = json.load(f)
                    
                    # Convert to numpy array and cache it
                    embedding_vector = np.array(embedding)
                    self.embeddings_cache[cache_key] = embedding_vector
                    return embedding_vector
                
            # Try parsing as a direct embedding string (for backward compatibility)
            if isinstance(embedding_reference, str) and embedding_reference.startswith('[') and embedding_reference.endswith(']'):
                embedding_vector = np.array(ast.literal_eval(embedding_reference))
                self.embeddings_cache[cache_key] = embedding_vector
                return embedding_vector
                
        except Exception as e:
            print(f"Error loading embedding: {e}")
        
        return None
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if vec1 is None or vec2 is None:
            return 0.0
            
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        
        # Avoid division by zero
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    
    def find_top_tags(self, query: str, top_n: int = 3) -> List[Dict[str, Any]]:
        """
        Find the top N tags that match the query based on embedding similarity.
        
        Args:
            query: User query text
            top_n: Number of top tags to return
            
        Returns:
            List of dictionaries containing tag information
        """
        if self.tags_df.empty or 'Embeddings' not in self.tags_df.columns:
            print("Tags database not properly loaded or doesn't contain embeddings")
            return []
        
        # Generate embedding for the query
        query_embedding = gpt_generate_embedding(query)
        
        # Calculate similarity scores for each tag
        similarities = []
        for idx, row in self.tags_df.iterrows():
            embedding_reference = row.get('Embeddings')
            if embedding_reference:
                # Load the embedding vector
                tag_embedding = self.get_embedding_vector(embedding_reference, is_article=False)
                
                if tag_embedding is not None:
                    sim_score = self.cosine_similarity(query_embedding, tag_embedding)
                    similarities.append({
                        'idx': idx,
                        'score': sim_score,
                        'primary_category': row['Primary Category'],
                        'secondary_category': row['Secondary Category'],
                        'tertiary_category': row['Tertiary Category'],
                        'explanation': row['Tags Explanation']
                    })
        
        # Sort by similarity score and get top N
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_n]
    
    def find_relevant_articles_by_tags(self, tags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Find articles that match the given tags using tag matching.
        
        Args:
            tags: List of tag dictionaries
            
        Returns:
            List of dictionaries containing relevant articles
        """
        if self.articles_df.empty:
            print("Articles database not properly loaded")
            return []
        
        # Extract tertiary categories from tags
        tertiary_categories = [tag['tertiary_category'] for tag in tags]
        
        # Find articles with matching tags
        relevant_articles = []
        
        for idx, article in self.articles_df.iterrows():
            article_tags = article.get('tags', '[]')
            
            # Parse the tags if they're stored as strings
            if isinstance(article_tags, str):
                try:
                    article_tags = ast.literal_eval(article_tags)
                except (ValueError, SyntaxError):
                    article_tags = []
            
            # Check if any of the tertiary categories match with article tags
            if any(tag in article_tags for tag in tertiary_categories):
                article_dict = {
                    'idx': idx,
                    'title': article.get('title', ''),
                    'text': article.get('text', ''),
                    'url': article.get('url', ''),
                    'tags': article_tags,
                    'score': 1.0  # Default score for tag-matched articles
                }
                
                # Add summary if available
                if 'summary' in article and article.get('summary'):
                    article_dict['summary'] = article.get('summary')
                
                relevant_articles.append(article_dict)
        
        return relevant_articles
    
    def find_relevant_articles_by_embedding(self, query: str, top_n: int = 5) -> List[Dict[str, Any]]:
        """
        Find articles that match the query based on embedding similarity.
        Only works if using the enhanced articles database with embeddings.
        
        Args:
            query: User query text
            top_n: Number of top articles to return
            
        Returns:
            List of dictionaries containing relevant articles with similarity scores
        """
        if not self.using_enhanced_articles or 'embedding_reference' not in self.articles_df.columns:
            # If not using enhanced database, return empty list
            return []
        
        # Generate embedding for the query
        query_embedding = gpt_generate_embedding(query)
        
        # Calculate similarity scores for each article
        similarities = []
        for idx, article in self.articles_df.iterrows():
            embedding_reference = article.get('embedding_reference')
            if embedding_reference:
                # Load the article embedding vector
                article_embedding = self.get_embedding_vector(embedding_reference, is_article=True)
                
                if article_embedding is not None:
                    sim_score = self.cosine_similarity(query_embedding, article_embedding)
                    
                    # Only consider articles with sufficient similarity
                    if sim_score > 0.5:  # Threshold can be adjusted
                        article_dict = {
                            'idx': idx,
                            'score': sim_score,
                            'title': article.get('title', ''),
                            'text': article.get('text', ''),
                            'url': article.get('url', ''),
                            'tags': article.get('tags', '[]')
                        }
                        
                        # Add summary if available
                        if 'summary' in article and article.get('summary'):
                            article_dict['summary'] = article.get('summary')
                            
                        similarities.append(article_dict)
        
        # Sort by similarity score and get top N
        similarities.sort(key=lambda x: x['score'], reverse=True)
        return similarities[:top_n]
    
    def find_relevant_articles(self, query: str, tags: List[Dict[str, Any]], 
                               max_articles: int = 5) -> List[Dict[str, Any]]:
        """
        Find relevant articles using both tag matching and embedding similarity.
        
        Args:
            query: User query text
            tags: List of tag dictionaries from find_top_tags
            max_articles: Maximum number of articles to return
            
        Returns:
            List of dictionaries containing relevant articles
        """
        if self.articles_df.empty:
            print("Articles database not properly loaded")
            return []
        
        # Get articles by tag matching
        tag_matched_articles = self.find_relevant_articles_by_tags(tags)
        
        # Get articles by embedding similarity (if available)
        embedding_matched_articles = []
        if self.using_enhanced_articles:
            embedding_matched_articles = self.find_relevant_articles_by_embedding(query)
        
        # Combine the two lists, removing duplicates
        all_articles = []
        seen_indices = set()
        
        # Add tag-matched articles first
        for article in tag_matched_articles:
            if article['idx'] not in seen_indices:
                seen_indices.add(article['idx'])
                all_articles.append(article)
        
        # Add embedding-matched articles that weren't already added
        for article in embedding_matched_articles:
            if article['idx'] not in seen_indices:
                seen_indices.add(article['idx'])
                all_articles.append(article)
        
        # Sort by score and limit to max_articles
        all_articles.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return all_articles[:max_articles]
    
    def process_query(self, query: str) -> Tuple[List[Dict[str, Any]], str]:
        """
        Process a user query and return relevant tags and context.
        
        Args:
            query: User query text
            
        Returns:
            Tuple containing (list of relevant tags, context string for the chatbot)
        """
        # Find top matching tags
        top_tags = self.find_top_tags(query)
        
        if not top_tags:
            return [], "No relevant information found."
        
        # Find articles related to these tags and the query
        relevant_articles = self.find_relevant_articles(query, top_tags)
        
        # Create context from relevant articles
        context = self._create_context_from_articles(relevant_articles, top_tags)
        
        return top_tags, context
    
    def _create_context_from_articles(self, articles: List[Dict[str, Any]], 
                                     tags: List[Dict[str, Any]]) -> str:
        """
        Create a context string from relevant articles and tags.
        
        Args:
            articles: List of relevant articles
            tags: List of matched tags
            
        Returns:
            Formatted context string for the chatbot
        """
        if not articles:
            return "No relevant articles found."
        
        # Create a context string with tag categories and article content
        context = "Here is information relevant to your query:\n\n"
        
        # Add matched tax categories
        context += "Relevant tax categories:\n"
        for i, tag in enumerate(tags, 1):
            context += f"{i}. {tag['primary_category']} > {tag['secondary_category']} > {tag['tertiary_category']}\n"
        
        context += "\nRelevant information:\n"
        
        # Add content from each article
        for i, article in enumerate(articles, 1):
            context += f"Article {i}: {article['title']}\n"
            
            # Use summary if available, otherwise parse text
            if 'summary' in article and article['summary']:
                context += article['summary']
            else:
                # Parse and extract article text
                text = article['text']
                if isinstance(text, str):
                    try:
                        # Try to parse as dictionary if it's stored that way
                        text_dict = ast.literal_eval(text)
                        if isinstance(text_dict, dict) and 'content' in text_dict:
                            context += text_dict['content']
                        else:
                            context += text
                    except (ValueError, SyntaxError):
                        context += text
            
            # Add similarity score if available (for debugging)
            if 'score' in article:
                context += f"\nRelevance score: {article['score']:.4f}"
                
            context += f"\nSource: {article['url']}\n\n"
        
        return context

# Function to update ChatBotGUI to use the RAG system
def fetch_relevant_documents(user_query: str) -> str:
    """
    Fetch relevant documents for a given user query using the RAG system.
    
    Args:
        user_query: The user's query text
        
    Returns:
        Context string containing relevant information
    """
    rag_system = RAGSystem()
    top_tags, context = rag_system.process_query(user_query)
    
    # Print matched tags for debugging
    if top_tags:
        print("\nMatched tags:")
        for i, tag in enumerate(top_tags, 1):
            print(f"{i}. {tag['tertiary_category']} (Score: {tag['score']:.4f})")

        print(context)
    
    return context