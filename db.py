from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
import json
import ssl

def upload_to_mongo(data, db_name, collection_name):
    MONGO_URI = "mongodb://nehabhakat:kMxFQYPZ5Z5rqHOEVJx2BPToWVyHpmmrpnl9rniu6AokTNMO3TsAhHXeFvQaEv3lNqeTNXctc2EQACDbthoOWw==@nehabhakat.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@nehabhakat@"
    client = MongoClient(MONGO_URI, tls=True, tlsAllowInvalidCertificates=True)
    db = client[db_name]
    collection = db[collection_name]

    try:
        result = collection.insert_many(data)
        print(f"Uploaded {len(result.inserted_ids)} documents to {collection_name}")
    except Exception as e:
        print(f"An error occurred while uploading to MongoDB: {e}")



if __name__ == "__main__":
    #
    tagged_data = [
        {"id": 1, "text": "This is a sample sentence.", "tags": ["sample", "sentence"]},
        {"id": 2, "text": "Another example with tags.", "tags": ["example", "tags"]}
    ]

    upload_to_mongo(tagged_data, "TaggedDatabase", "TaggedCollection")

def fetch_from_mongo(db_name, collection_name, query={}, mongo_uri="mongodb://nehabhakat:kMxFQYPZ5Z5rqHOEVJx2BPToWVyHpmmrpnl9rniu6AokTNMO3TsAhHXeFvQaEv3lNqeTNXctc2EQACDbthoOWw==@nehabhakat.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@nehabhakat@"):

    try:
        client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db = client[db_name]
        collection = db[collection_name]

        results = collection.find(query)
        return list(results)  # Convert cursor to list of dictionaries

    except Exception as e:
        print(f"Error: {e}")
        return []
    finally:
        client.close()

if __name__ == "__main__":
    data = fetch_from_mongo("TaggedDatabase", "TaggedCollection")
    print("Fetched Data:", data)
