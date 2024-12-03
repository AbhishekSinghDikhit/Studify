from pymongo import MongoClient

def get_database():
    client = MongoClient("mongodb+srv://<username>:<password>@cluster.mongodb.net/?retryWrites=true&w=majority")
    return client['eduai-database']
