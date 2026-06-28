import os, warnings
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)
warnings.filterwarnings("ignore")

import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path=os.path.join(BASE_DIR, "knowledge_base", "chroma_store"))

for col_name in ["laser_cladding", "material_ml"]:
    col = client.get_collection(col_name)
    print(f"\nCollection: {col_name} ({col.count()} docs)")
    
    queries = ["JG-1 hardness", "laser power", "XGBoost"]
    for q in queries:
        res = col.query(query_texts=[q], n_results=3)
        if res["distances"] and res["distances"][0]:
            for doc, dist in zip(res["documents"][0][:2], res["distances"][0][:2]):
                sim = 1 - dist
                print(f"  Q='{q}' sim={sim:.3f} | {doc[:80]}...")
        else:
            print(f"  Q='{q}' - no results")
