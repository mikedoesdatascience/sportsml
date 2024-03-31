import os

from pymongo import MongoClient

MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_USERNAME = os.getenv("MONGODB_USERNAME")
MONGODB_PASSWORD = os.getenv("MONGODB_PASSWORD")

client = MongoClient(MONGODB_URI, username=MONGODB_USERNAME, password=MONGODB_PASSWORD)
