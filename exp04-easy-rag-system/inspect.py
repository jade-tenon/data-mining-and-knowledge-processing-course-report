
from pymilvus import MilvusClient

DB_PATH = "./milvus_lite_data.db"
COLLECTION_NAME = "medical_rag_lite"

def main():
    print(f"Trying to open DB file: {DB_PATH}")
    client = MilvusClient(uri=DB_PATH)
    print("OK: Milvus Lite client created.")

    cols = client.list_collections()
    print("Collections:", cols)

    if COLLECTION_NAME not in cols:
        print(f"Collection {COLLECTION_NAME!r} does NOT exist.")
        return

    try:
        if hasattr(client, "num_entities"):
            count = client.num_entities(COLLECTION_NAME)
        else:
            stats = client.get_collection_stats(COLLECTION_NAME)
            count = int(stats.get("row_count", stats.get("rowCount", 0)))
        print(f"Collection {COLLECTION_NAME!r} row count:", count)
    except Exception as e:
        print("Error getting collection stats:", e)

if __name__ == "__main__":
    main()
