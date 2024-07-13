import uuid
from qdrant_client.http import models
from qdrant_client import QdrantClient
import os 
import dotenv

dotenv.load_dotenv()


class QdrantU:
    def __init__(self, collection_name):
        self.client = QdrantClient(
            url=os.getenv("qdrant_url"),
            api_key=os.getenv("qdrant_api_key"),
        )
        self.collection_name = collection_name

    def _upload_documents_to_Qdrant(self, data, source):
        points = []
        for title, content, publishdate, embedding in zip(data["title"], data["content"], data["publishdate"], data["embedding"]):
            new_id = str(uuid.uuid4())  # Generate a new UUID for each document
            point = models.PointStruct(
                id=new_id,
                vector=embedding,
                payload={
                    "title": title,
                    "content": content,
                    "publishdate": publishdate,
                    "source" : source
                }
            )
            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print("Uploaded:", len(data["embedding"]), "documents to the Qdrant database")


    def upload_to_Qdrant(self, data, batch_size=35, source=''):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            self._upload_documents_to_Qdrant(batch , source)
            print(f"Uploaded {i + len(batch)} documents")


    def get_number_of_vectors(self):
        collection_info = self.client.get_collection(self.collection_name)
        num_vectors = collection_info.points_count
        return num_vectors
    
    def close_connection(self):
        self.client.close()

    def search(self, query, text_embedder, limit):
        query_vector = text_embedder.embed_query(query_text=query)
        query_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector[0].tolist(),  # Convert tensor to list
            limit=limit,
            with_payload=True
        )
        return query_result