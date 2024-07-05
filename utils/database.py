import cv2
import torch
import numpy as np
import pickle
from PIL import Image
from pymilvus import (
    connections,
    utility,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)
from pymongo import MongoClient
from bson import ObjectId
import torch



class Database:

    def __init__(
        self,
        milvus_host,
        milvus_port,
        mongo_uri,
        db_name,
        milvus_collection_name,
        expected_dim,
    ):
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.milvus_collection_name = milvus_collection_name
        self.expected_dim = expected_dim
        self.milvus_collection = self.init_milvus()
        self.mongo_collection = self.init_mongodb()

    def init_milvus(self):
        print("Initializing Milvus...")
        connections.connect(
            alias="default", host=self.milvus_host, port=self.milvus_port
        )

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(
                name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.expected_dim
            ),
            FieldSchema(name="mongo_id", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=50),
        ]
        schema = CollectionSchema(fields, description="Image depthmap feature vectors")

        existing_collections = utility.list_collections()
        print(existing_collections)

        if self.milvus_collection_name not in existing_collections:
            collection = Collection(name=self.milvus_collection_name, schema=schema)
            print(f"Milvus collection '{self.milvus_collection_name}' created.")
        else:
            collection = Collection(name=self.milvus_collection_name)
            print(f"Milvus collection '{self.milvus_collection_name}' already exists.")

        index_params = {
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
            "metric_type": "L2",
        }
        collection.create_index(field_name="vector", index_params=index_params)
        print(f"Index created for collection '{self.milvus_collection_name}'.")

        collection.load()
        print(f"Milvus collection '{self.milvus_collection_name}' loaded into memory.")

        return collection

    def init_mongodb(self):
        print("Initializing MongoDB...")
        client = MongoClient(self.mongo_uri)
        db = client[self.db_name]
        collection = db["depthmaps2"]
        print("Connected to MongoDB...")
        return collection

    @staticmethod
    def adjust_vector_length(vector, dimension):
        if len(vector) > dimension:
            return vector[:dimension]
        elif len(vector) < dimension:
            return vector + [0.0] * (dimension - len(vector))
        return vector

    def extract_features(self, image_path):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return np.zeros(self.expected_dim, dtype=np.float32)

        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(image, None)

        if descriptors is None:
            print("No descriptors found, returning zero array.")
            return np.zeros(self.expected_dim, dtype=np.float32)


        feature_vector = descriptors.flatten().tolist()
        feature_vector = self.adjust_vector_length(feature_vector, self.expected_dim)
        feature_vector = np.array(feature_vector, dtype=np.float32)
        feature_vector /= np.linalg.norm(feature_vector) 

        return feature_vector

    def store_image_mongodb(self, depth):
         # Convert the tensor to a byte array
        depth_byte_array = pickle.dumps(depth.cpu().numpy())
        tensor_doc = {"tensor_data": depth_byte_array}
        result = self.mongo_collection.insert_one(tensor_doc)
        return str(result.inserted_id)


    def store_image_features(self, image_path, depth_path):
        print(f"Storing features for image: {image_path}")
        features = self.extract_features(image_path).tolist()
        mongo_id = self.store_image_mongodb(depth_path)
        data = [
            {"vector": features, "mongo_id": mongo_id, "name": image_path},
        ]
        self.milvus_collection.insert(data)
        print(f"Inserted features into Database for image: {image_path}")

    def retrieve_similar_images(self, image_path, top_k=2):
        print(f"Retrieving similar images for query image: {image_path}")
        query_features = self.extract_features(image_path)

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }

        results = self.milvus_collection.search(
            data=[query_features.tolist()],
            anns_field="vector",
            param=search_params,
            limit=top_k,
            expr=None,
            output_fields=["mongo_id", "name"],
        )

        matches = results[0]

        retrieved_images = []
        for match in matches:
            if match.distance < 1:
                mongo_id = match.entity.get("mongo_id")
                tensor_doc = self.mongo_collection.find_one({"_id": ObjectId(mongo_id)})
                if tensor_doc:
                    tensor_data = tensor_doc["tensor_data"]
                    low_dep = torch.from_numpy(pickle.loads(tensor_data))
                    retrieved_images.append((low_dep))
        print(f"Retrieved {len(retrieved_images)} results")
        return matches, retrieved_images
