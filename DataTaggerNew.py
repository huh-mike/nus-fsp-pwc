from traceback import print_tb

import pandas as pd
import ast

from GPTServices import gpt_generate_embedding
from sklearn.metrics.pairwise import cosine_similarity

def create_embeddings_and_get_relevant_tags(raw_dataset):
    df = pd.read_csv("tag_data_with_embeddings.csv")
    valid_tags = set(df["tag"])

    def generate_best_tags(top_n, article):
        # top_n is the top number of tags
        article_embedding = gpt_generate_embedding(article)
        similarities = {
            tag: cosine_similarity([article_embedding], [tag_embedding])[0][0]
            for tag, tag_embedding in tag_embeddings.items()
        }
        sorted_tags = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        best_tags = [(tag, score) for tag, score in sorted_tags[:top_n] if tag in valid_tags]
        return best_tags

    if "embedding" not in df.columns or df["embedding"].isna().any():

        df["embedding"] = df["tag"].apply(gpt_generate_embedding)
        df["embedding"] = df["embedding"].apply(lambda x: str(x))

        df.to_csv("tag_data_with_embeddings.csv", index=False)
        print("embeddings saved to csv!")

    df["embedding"] = df["embedding"].apply(ast.literal_eval)

    tag_embeddings = {row["tag"]: list(map(float, row["embedding"])) for _, row in df.iterrows()}

    tagged_dataset = raw_dataset

    for item in tagged_dataset:
        text_content = item["text"]
        new_tags = [tag for tag, _ in generate_best_tags(5,text_content)]
        item["tags"] = new_tags

    return tagged_dataset

def main():
    print("something")


if __name__ == "__main__":
    main()



