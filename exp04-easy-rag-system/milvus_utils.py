# File: `exp04-easy-rag-system/milvus_utils.py`
# language: python
# 主要修改：支持 server 模式并改进 id 映射/分配逻辑

import streamlit as st
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
import time
import os

from config import (
    MILVUS_LITE_DATA_PATH, COLLECTION_NAME, EMBEDDING_DIM,
    MAX_ARTICLES_TO_INDEX, INDEX_METRIC_TYPE, INDEX_TYPE, INDEX_PARAMS,
    SEARCH_PARAMS, TOP_K, id_to_doc_map, MILVUS_MODE, MILVUS_URI, PREVIEW_MAX_LENGTH
)

@st.cache_resource
def get_milvus_client():
    """Initializes and returns a MilvusClient instance for server or lite."""
    try:
        if MILVUS_MODE == "server" and MILVUS_URI:
            st.write(f"Connecting to Milvus server at: {MILVUS_URI}")
            client = MilvusClient(uri=MILVUS_URI)
        else:
            st.write(f"Initializing Milvus Lite client with data path: {MILVUS_LITE_DATA_PATH}")
            os.makedirs(os.path.dirname(MILVUS_LITE_DATA_PATH) or ".", exist_ok=True)
            client = MilvusClient(uri=MILVUS_LITE_DATA_PATH)
        st.success("Milvus client initialized!")
        return client
    except Exception as e:
        st.error(f"Failed to initialize Milvus client: {e}")
        return None

@st.cache_resource
def setup_milvus_collection(_client):
    """Ensures the specified collection exists, is created if needed, and loaded for search."""
    if not _client:
        st.error("Milvus client not available.")
        return False
    try:
        collection_name = COLLECTION_NAME
        dim = EMBEDDING_DIM

        has_collection = collection_name in _client.list_collections()

        if not has_collection:
            st.write(f"Collection '{collection_name}' not found. Creating...")
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
                FieldSchema(name="content_preview", dtype=DataType.VARCHAR, max_length=PREVIEW_MAX_LENGTH),
            ]
            schema = CollectionSchema(fields, f"PubMed RAG (dim={dim})")

            _client.create_collection(
                collection_name=collection_name,
                schema=schema
            )
            st.write(f"Collection '{collection_name}' created.")

            st.write(f"Creating index ({INDEX_TYPE})...")
            index_params = _client.prepare_index_params()
            index_params.add_index(
                field_name="embedding",
                index_type=INDEX_TYPE,
                metric_type=INDEX_METRIC_TYPE,
                params=INDEX_PARAMS
            )
            _client.create_index(collection_name, index_params)
            st.success(f"Index created for collection '{collection_name}'.")
        else:
            st.write(f"Found existing collection: '{collection_name}'.")

        # 尝试加载 collection，确保后续 search 可用
        try:
            if hasattr(_client, "load_collection"):
                _client.load_collection(collection_name)
                st.write(f"Collection '{collection_name}' loaded into memory.")
        except Exception as e_load:
            st.write(f"Warning: failed to load collection '{collection_name}': {e_load} (search may fail until loaded)")

        try:
            if hasattr(_client, 'num_entities'):
                current_count = _client.num_entities(collection_name)
            else:
                stats = _client.get_collection_stats(collection_name)
                current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
            st.write(f"Collection '{collection_name}' ready. Current entity count: {current_count}")
        except Exception:
            st.write(f"Collection '{collection_name}' ready.")

        return True

    except Exception as e:
        st.error(f"Error setting up Milvus collection '{COLLECTION_NAME}': {e}")
        return False


def index_data_if_needed(client, data, embedding_model):
    """Checks if data needs indexing and performs it. Uses/produces int primary keys and updates id_to_doc_map."""
    global id_to_doc_map

    if not client:
        st.error("Milvus client not available for indexing.")
        return False

    collection_name = COLLECTION_NAME
    # Retrieve current entity count with fallback
    try:
        if hasattr(client, 'num_entities'):
            current_count = client.num_entities(collection_name)
        else:
            stats = client.get_collection_stats(collection_name)
            current_count = int(stats.get("row_count", stats.get("rowCount", 0)))
    except Exception:
        st.write(f"Could not retrieve entity count, attempting to (re)setup collection.")
        if not setup_milvus_collection(client):
            return False
        current_count = 0

    st.write(f"Entities currently in Milvus collection '{collection_name}': {current_count}")

    data_to_index = data[:MAX_ARTICLES_TO_INDEX]
    docs_for_embedding = []
    data_to_insert = []
    temp_id_map = {}

    # ensure a sane max length
    max_len = int(PREVIEW_MAX_LENGTH or 500)

    # assign integer ids: prefer doc['id'] if int, else allocate sequential ints starting from current_count
    next_int_id = int(current_count)
    assigned = 0

    with st.spinner("Preparing data for indexing..."):
        for doc in data_to_index:
            title = doc.get('title', '') or ""
            abstract = doc.get('abstract', '') or ""
            content = f"Title: {title}\nAbstract: {abstract}".strip()
            if not content:
                continue

            raw_id = doc.get('id', None)
            if isinstance(raw_id, int):
                int_id = raw_id
            else:
                int_id = next_int_id + assigned
                assigned += 1

            # store full content in map (for generation), but ensure preview is truncated
            temp_id_map[int_id] = {
                'title': title, 'abstract': abstract, 'content': content, 'raw_id': raw_id
            }
            docs_for_embedding.append(content)

            # truncate preview using config value (NOT hardcoded 500)
            preview = content[:max_len]
            data_to_insert.append({
                "id": int_id,
                "embedding": None,
                "content_preview": preview
            })

    needed_count = len(data_to_insert)

    if current_count < needed_count and docs_for_embedding:
        st.warning(f"Indexing required ({current_count}/{needed_count} documents found). This may take a while...")

        st.write(f"Embedding {len(docs_for_embedding)} documents...")
        with st.spinner("Generating embeddings..."):
            start_embed = time.time()
            embeddings = embedding_model.encode(docs_for_embedding, show_progress_bar=True)
            end_embed = time.time()
            st.write(f"Embedding took {end_embed - start_embed:.2f} seconds.")

        for i, emb in enumerate(embeddings):
            data_to_insert[i]["embedding"] = emb

        # --- 额外校验：插入前再次确保所有 preview 不超过 max_len ---
        too_long = []
        for item in data_to_insert:
            cp = item.get("content_preview", "")
            if len(cp) > max_len:
                too_long.append((item["id"], len(cp)))
                item["content_preview"] = cp[:max_len]  # 强制截断

        if too_long:
            st.write(f"Detected and truncated {len(too_long)} previews exceeding {max_len} chars. Examples: {too_long[:5]}")

        st.write("Inserting data into Milvus...")
        with st.spinner("Inserting..."):
            try:
                start_insert = time.time()
                res = client.insert(collection_name=collection_name, data=data_to_insert)
                end_insert = time.time()
                inserted_count = len(data_to_insert)
                st.success(f"Successfully attempted to index {inserted_count} documents. Insert took {end_insert - start_insert:.2f} seconds.")
                # Update mapping so keys are the int primary keys used in Milvus
                id_to_doc_map.update(temp_id_map)
                return True
            except Exception as e:
                st.error(f"Error inserting data into Milvus: {e}")
                return False
    elif current_count >= needed_count:
        st.write("Data count suggests indexing is complete.")
        if not id_to_doc_map:
            # Populate map from data if possible (assign ints similarly)
            temp_map = {}
            base = 0
            for i, doc in enumerate(data_to_index):
                raw_id = doc.get('id', None)
                int_id = base + i
                temp_map[int_id] = {
                    'title': doc.get('title', ''), 'abstract': doc.get('abstract', ''), 'content': f"Title: {doc.get('title','')}\nAbstract: {doc.get('abstract','')}".strip()
                }
            id_to_doc_map.update(temp_map)
        return True
    else:
         st.error("No valid text content found in the data to index.")
         return False


def search_similar_documents(client, query, embedding_model):
    """Searches Milvus for documents similar to the query. If collection not loaded, try to load and retry once."""
    if not client or not embedding_model:
        st.error("Milvus client or embedding model not available for search.")
        return [], []

    collection_name = COLLECTION_NAME
    try:
        query_embedding = embedding_model.encode([query])[0]

        search_params = {
            "collection_name": collection_name,
            "data": [query_embedding],
            "anns_field": "embedding",
            "limit": TOP_K,
            "output_fields": ["id"]
        }

        def _do_search(params):
            if hasattr(client, 'search_with_params'):
                return client.search_with_params(**params, search_params=SEARCH_PARAMS)
            else:
                try:
                    return client.search(**params)
                except Exception as e1:
                    # 兼容不同 client 实现的调用签名
                    try:
                        return client.search(**params, **SEARCH_PARAMS)
                    except Exception:
                        final_params = params.copy()
                        final_params["nprobe"] = SEARCH_PARAMS.get("nprobe", 16)
                        return client.search(**final_params)

        # 尝试搜索，若出错（例如 collection not loaded）则尝试加载后重试一次
        try:
            res = _do_search(search_params)
        except Exception as e:
            st.warning(f"搜索失败: {e}，尝试加载 collection 并重试...")
            try:
                if hasattr(client, "load_collection"):
                    client.load_collection(collection_name)
                    st.write(f"已加载 collection '{collection_name}'，重试搜索...")
                else:
                    client.load_collection(collection_name)
            except Exception as e_load:
                st.error(f"无法加载 collection: {e_load}")
                return [], []
            try:
                res = _do_search(search_params)
            except Exception as e2:
                st.error(f"搜索重试失败: {e2}")
                return [], []

        if not res or not res[0]:
            return [], []

        hits = res[0]
        hit_ids = []
        distances = []
        for hit in hits:
            if isinstance(hit, dict):
                hit_id = hit.get('id')
                dist = hit.get('distance')
            else:
                hit_id = getattr(hit, 'id', None)
                dist = getattr(hit, 'distance', None)
            hit_ids.append(hit_id)
            distances.append(dist)
        return hit_ids, distances
    except Exception as e:
        st.error(f"Error during search: {e}")
        return [], []
