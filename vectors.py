import os
import warnings
import pandas as pd
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.schema import Document
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.llms import HuggingFaceHub
from qdrant_client.http import models

warnings.filterwarnings("ignore")
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HUGGINGFACEHUB_API_TOKEN")

class EmbeddingsManager:
    def __init__(
            self,
            model_name: str = "BAAI/bge-small-en",
            device: str = "cpu",
            encode_kwargs: dict = {"normalize_embeddings": True},
            qdrant_url: str = "http://localhost:6333",
            collection_name: str = "vector_db",
    ):
        """
        Initializes the EmbeddingsManager with the specified model and Qdrant settings.
        """
        self.model_name = model_name
        self.device = device
        self.encode_kwargs = encode_kwargs
        self.qdrant_url = qdrant_url
        self.collection_name = collection_name

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.device},
            encode_kwargs=self.encode_kwargs,
        )

    def create_embeddings(self, file_path: str):
        """
        Processes the PDF or CSV file, creates embeddings, and stores them in Qdrant.

        Args:
            file_path (str): The file path to the document (PDF or CSV).

        Returns:
            str: Success message upon completion.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")



        file_extension = file_path.split(".")[-1]
        if file_extension == "pdf":
            loader = UnstructuredPDFLoader(file_path)
            documents = loader.load()
        elif file_extension == "csv":
            # Handle CSV with multi-line fields properly
            documents = self._load_csv_with_multiline_fields(file_path)
        else:
            raise ValueError("Unsupported file type")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Create and store embeddings in Qdrant
        try:
            qdrant = Qdrant.from_documents(
                texts,
                self.embeddings,
                url=self.qdrant_url,
                force_recreate=True,
                prefer_grpc=False,
                collection_name=self.collection_name,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Qdrant: {e}")

        return "✅ Vector DB Successfully Created and Stored in Qdrant!"
    
    def _load_csv_with_multiline_fields(self, file_path: str):
        """
        Load CSV file with proper handling of multi-line fields.
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            list: List of Document objects
        """
        try:
            # Try different encodings to handle various CSV formats
            encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    print(f"Error reading CSV with {encoding} encoding: {e}")
                    continue
            
            if df is None:
                raise ValueError("Could not read the CSV file with any supported encoding")
            
            # Convert DataFrame to documents
            documents = []
            for index, row in df.iterrows():
                # Create a text representation of each row
                text_parts = []
                for column, value in row.items():
                    if pd.notna(value):  # Only include non-null values
                        text_parts.append(f"{column}: {str(value)}")
                
                # Join all parts with newlines for better readability
                text_content = "\n".join(text_parts)
                
                # Create metadata
                metadata = {
                    "source": file_path,
                    "row_index": index,
                    "file_type": "csv"
                }
                
                # Create Document object
                doc = Document(page_content=text_content, metadata=metadata)
                documents.append(doc)
            
            return documents
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {e}")
