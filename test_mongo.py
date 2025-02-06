from pymongo import MongoClient
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get MongoDB credentials
mongo_uri = os.getenv("MONGO_URI")
db_name = os.getenv("MONGO_DB_NAME")

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[db_name]

# Print connection status
print("Connected to MongoDB:", db.name)


