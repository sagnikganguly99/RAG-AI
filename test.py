import json

data = json.load(open("embeddings_1024.json"))
first = next(iter(data.values()))
print("Vector dimension:", len(first))
