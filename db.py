from pymongo import MongoClient
from dotenv import load_dotenv
import os
import logging
import json
import ssl


def upload_to_mongo(data, db_name, collection_name,
                    mongo_uri="mongodb://nehabhakat:kMxFQYPZ5Z5rqHOEVJx2BPToWVyHpmmrpnl9rniu6AokTNMO3TsAhHXeFvQaEv3lNqeTNXctc2EQACDbthoOWw==@nehabhakat.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@nehabhakat@"):
    """
    Uploads tagged data to MongoDB on Azure.
    :param data: List of dictionaries containing tagged data.
    :param db_name: Name of the database.
    :param collection_name: Name of the collection.
    :param mongo_uri: MongoDB connection URI.
    """
    try:
        client = MongoClient(mongo_uri, tls=True, tlsAllowInvalidCertificates=True)
        db = client[db_name]
        collection = db[collection_name]

        if isinstance(data, list):
            collection.insert_many(data)
        else:
            collection.insert_one(data)

        print("Data successfully uploaded to MongoDB on Azure.")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    # Example tagged data
    tagged_data = [
        {"id": 1, "text": "This is a sample sentence.", "tags": ["sample", "sentence"]},
        {"id": 2, "text": "Another example with tags.", "tags": ["example", "tags"]}
    ]

    # Upload to MongoDB on Azure
    upload_to_mongo(tagged_data, "TaggedDatabase", "TaggedCollection")

def fetch_from_mongo(db_name, collection_name, query={}, mongo_uri="mongodb://nehabhakat:kMxFQYPZ5Z5rqHOEVJx2BPToWVyHpmmrpnl9rniu6AokTNMO3TsAhHXeFvQaEv3lNqeTNXctc2EQACDbthoOWw==@nehabhakat.mongo.cosmos.azure.com:10255/?ssl=true&replicaSet=globaldb&retrywrites=false&maxIdleTimeMS=120000&appName=@nehabhakat@"):
    """
    Fetches data from MongoDB based on a query.

    :param db_name: Name of the database.
    :param collection_name: Name of the collection.
    :param query: Dictionary query for filtering results.
    :param mongo_uri: MongoDB connection URI.
    """
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
