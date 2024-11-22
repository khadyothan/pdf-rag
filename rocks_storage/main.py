import json
from rocksdict import Rdict

# Load JSON file
file_path = "C:/Khadyu/ASU/Fall 2024/DPS/Group Project/pdf-rag/rocks_storage/sample.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Open RocksDB instance
db_path = "C:/Khadyu/ASU/Fall 2024/DPS/Group Project/pdf-rag/rocks_storage/db"
db = Rdict(db_path)

# Insert data into RocksDB
for paper_id, paper_data in data.items():
    db[paper_id] = json.dumps(paper_data)

# Verify and Retrieve data
retrieved_data = db["paper_1"]
print(json.loads(retrieved_data))

# Close RocksDB
db.close()
