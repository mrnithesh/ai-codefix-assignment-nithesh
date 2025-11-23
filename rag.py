import os
import logging
from typing import List, Dict
from sentence_transformers import SentenceTransformer
import faiss

# Configure logging
logger = logging.getLogger(__name__)

class RAGComponent:
    def __init__(self, recipes_dir: str = "recipes", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.recipes_dir = recipes_dir
        self.model_name = model_name
        self.documents: List[str] = []
        self.filenames: List[str] = []
        self.index = None
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.load_recipes()
            self.build_index()
        except Exception as e:
            logger.error(f"Failed to initialize RAGComponent: {e}")
            raise

    def load_recipes(self):
        #Load all text files from the recipes directory.
        if not os.path.exists(self.recipes_dir):
            logger.warning(f"Recipes directory not found: {self.recipes_dir}")
            return

        for filename in os.listdir(self.recipes_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.recipes_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read().strip()
                        if content:
                            # Prepend filename to content to improve retrieval context
                            clean_filename = filename.replace(".txt", "").replace("-", " ")
                            enriched_content = f"{clean_filename}\n{content}"
                            self.documents.append(enriched_content)
                            self.filenames.append(filename)
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")
        
        logger.info(f"Loaded {len(self.documents)} recipes from {self.recipes_dir}")

    def build_index(self):
        #Generate embeddings and build the FAISS index.
        if not self.documents:
            logger.warning("No documents to index.")
            return

        logger.info("Generating embeddings...")
        embeddings = self.model.encode(self.documents, convert_to_numpy=True)
        
        # Initialize FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        logger.info(f"Indexed {self.index.ntotal} documents.")

    def retrieve(self, query: str, k: int = 1) -> List[str]:
        #Retrieve top-k relevant recipes for a given query.
        if not self.index or self.index.ntotal == 0:
            return []

        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.documents):
                logger.info(f"Retrieved: {self.filenames[idx]} (distance: {distances[0][i]:.4f})")
                results.append(self.documents[idx])
        
        return results
