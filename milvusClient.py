import os
from pymilvus import connections, Collection, DataType, CollectionSchema, FieldSchema, utility
from sentence_transformers import SentenceTransformer
import time

class MilvusClient:
    def __init__(self, host="127.0.0.1", port="19530", retries=3):
        self.host = host
        self.port = port
        self.connect_with_retry(retries)
        collection_name = "html_chunks"
        self.setup_collection(collection_name)

    def connect_with_retry(self, retries):
        """Attempt to connect to Milvus with retries"""
        for attempt in range(retries):
            try:
                # Check if already connected
                if connections.has_connection("default"):
                    print("Already connected to Milvus")
                    return

                print(f"Attempting to connect to Milvus at {self.host}:{self.port}")
                connections.connect("default", host=self.host, port=self.port)
                print("Successfully connected to Milvus")
                return
            except Exception as e:
                print(f"Connection attempt {attempt + 1}/{retries} failed: {str(e)}")
                if attempt < retries - 1:
                    time.sleep(2)
                else:
                    raise Exception("Failed to connect to Milvus after multiple attempts")

    def setup_collection(self, collection_name):
        """Set up the collection with error handling"""
        try:
            
            # Check if collection exists
            if utility.has_collection(collection_name):
                print(f"Collection {collection_name} already exists")
                self.collection = Collection(collection_name)
            else:
                print(f"Creating new collection: {collection_name}")
                # Collection schema definition
                id_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False)
                text_field = FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
                vector_field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
                
                schema = CollectionSchema([id_field, text_field, vector_field], "HTML content chunks")
                self.collection = Collection(collection_name, schema=schema)
                
                # Create index if it doesn't exist
                if not self.collection.has_index():
                    index_params = {
                        "metric_type": "L2",
                        "index_type": "IVF_FLAT",
                        "params": {"nlist": 128}
                    }
                    self.collection.create_index("embedding", index_params)
            
            self.collection.load()
            print("Collection setup completed successfully")
            
        except Exception as e:
            raise Exception(f"Failed to setup collection: {str(e)}")

    def drop_collection(self, collection_name):
        """Drop the collection based on the 'drop collection' flag."""
        try:
            # Check if the collection exists
            if utility.has_collection(collection_name):
                # Drop the collection
                utility.drop_collection(collection_name)
                print(f"Collection '{collection_name}' dropped successfully")
            else:
                print(f"Collection '{collection_name}' does not exist.")
        except Exception as e:
            print(f"Failed to drop collection: {str(e)}")


    def index_data(self, chunks, embeddings):
        try:
            entities = [
                [i for i in range(len(chunks))],
                chunks,
                embeddings
            ]
            self.collection.insert(entities)
            # Add flush to ensure data is persisted
            self.collection.flush()
            print(f"Successfully indexed {len(chunks)} chunks")
            
            # Verify insertion
            print(f"Collection now has {self.collection.num_entities} entities")
        except Exception as e:
            raise Exception(f"Failed to index data: {str(e)}")
        
    def verify_data(self):
        try:
            print(f"Number of entities: {self.collection.num_entities}")
            # Query first few records
            results = self.collection.query(
                expr="id < 2",  # Get first 3 records
                output_fields=["id", "text", "embedding"]
            )
            print("Sample records:", results)
            return results
        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return None

    def search(self, query_text):
        try:
            print(f"Collection stats: {self.collection.num_entities} entities")
            
            embedding = SentenceTransformer("all-MiniLM-L6-v2").encode([query_text])
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            
            results = self.collection.search(
                data=embedding,
                anns_field="embedding",
                param=search_params,
                limit=10,
                output_fields=["text"],
                expr=None
            )
            
            # Convert distance to similarity score and format results
            search_results = []
            for result in results[0]:
                # L2 distance to similarity score (0-1 range)
                similarity = 1 / (1 + result.distance)
                search_results.append({
                    "chunk": result.entity.get("text"),
                    "relevance_score": round(similarity, 3),
                    "distance": round(result.distance, 3)
                })
            
            # Sort by relevance score
            search_results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return search_results
        except Exception as e:
            raise Exception(f"Search failed: {str(e)}")

