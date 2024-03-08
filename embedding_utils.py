from sentence_transformers import SentenceTransformer
import PyPDF2
import json
import boto3
from typing import List


class TextProcessor:
    """Utility class for processing text data."""

    def __init__(self, max_chunk_length: int, overlap: int = 0):
        """
        Initialize TextProcessor.

        Args:
            max_chunk_length (int): Maximum length of each text chunk.
            overlap (int, optional): Number of characters to overlap between chunks. Defaults to 0.
        """
        self.max_chunk_length = max_chunk_length
        self.overlap = overlap

    def split_text(self, text: str) -> list:
        """
        Split text into chunks of specified length, removing NUL characters.

        Args:
            text (str): Input text.

        Returns:
            list: List of text chunks.
        """
        # Remove NUL characters from the text
        cleaned_text = ''.join(char for char in text if char != '\x00')

        if len(cleaned_text) <= self.max_chunk_length:
            return [cleaned_text]
        else:
            split_index = cleaned_text.rfind(' ', self.max_chunk_length - self.overlap, self.max_chunk_length)
            if split_index == -1:
                split_index = self.max_chunk_length
            return [cleaned_text[:split_index]] + self.split_text(cleaned_text[split_index - self.overlap:])

    def chunk_text(self, text: str) -> list:
        """
        Alias for split_text method.

        Args:
            text (str): Input text.

        Returns:
            list: List of text chunks.
        """
        return self.split_text(text)

    def chunk_pdf_file(self, pdf_path: str) -> list:
        """
        Chunk text from a PDF file.

        Args:
            pdf_path (str): Path to the PDF file.

        Returns:
            list: List of text chunks.
        """
        try:
            with open(pdf_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text_chunks = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    page_chunks = self.split_text(page_text)
                    text_chunks.extend(page_chunks)
        except FileNotFoundError as e:
            print(f"Error: File '{pdf_path}' not found. {e}")
            return []
        except Exception as e:
            print(f"Error processing PDF file '{pdf_path}'. {e}")
            return []
        return text_chunks

    def chunk_json_file(self, json_path: str) -> list:
        """
        Chunk data from a JSON file.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            list: List of JSON string chunks.
        """
        try:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
        except FileNotFoundError as e:
            print(f"Error: File '{json_path}' not found. {e}")
            return []
        except Exception as e:
            print(f"Error loading JSON file '{json_path}'. {e}")
            return []

        chunks = [json.dumps(item) for item in data]
        return chunks


class TitanEmbedder:
    def __init__(self, region: str = None):
        """
        Initialize the TitanEmbedder.

        Args:
            region (str): AWS region for the client.
        """
        if region:
            self.client = boto3.client('bedrock-runtime', region_name=region)
        else:
            self.client = boto3.client('bedrock-runtime')

    def embed_text(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed the given chunks of text using Titan.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List of embeddings for each chunk.
        """
        results = []
        for text in chunks:
            response = self._invoke_titan(text)
            if isinstance(response, list):
                results.append(response)
            else:
                raise ValueError("Invalid response format.")
        return results

    def generate_request_body(self, text: str) -> dict:
        """
        Generate request body for embedding.

        Args:
            text (str): Text to generate request body for.

        Returns:
            Request body in dictionary format.
        """
        return {"inputText": text}

    def _invoke_titan(self, text: str) -> List[float]:
        """
        Invoke Titan model for embedding.

        Args:
            text (str): Text to embed.

        Returns:
            List[float]: Embedding for the text.
        """
        body = json.dumps(self.generate_request_body(text))
        kwargs = {
            "modelId":  "amazon.titan-embed-text-v1",
            "contentType": "application/json",
            "accept": "*/*",
            "body": body
        }
        response = self.client.invoke_model(**kwargs)
        body = json.loads(response['body'].read())
        return body['embedding']

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, revision: str = None):
        """
        Initialize the SentenceTransformerEmbedder.

        Args:
            model_name (str): Name of the Sentence Transformer model.
            revision (str): Revision of the model.
        """
        self.model_embeddings = SentenceTransformer(model_name, revision=revision)

    def embed_text(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed the given chunks of text using Sentence Transformer.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List of embeddings for each chunk.
        """
        return self.model_embeddings.encode(chunks, convert_to_tensor=False).tolist()

class TextEmbedder:
    def __init__(self, model_name: str = None, revision: str = None, region: str = None):
        """
        Initialize the TextEmbedderController.

        Args:
            model_name (str): Name of the model.
            revision (str): Revision of the model.
            region (str): AWS region for the client.
        """
        if model_name == "amazon.titan-embed-text-v1":
            self.embedder = TitanEmbedder(region)
        else:
            self.embedder = SentenceTransformerEmbedder(model_name, revision)

    def embed_text(self, chunks: List[str]) -> List[List[float]]:
        """
        Embed the given chunks of text using the appropriate embedder.

        Args:
            chunks (List[str]): List of text chunks.

        Returns:
            List of embeddings for each chunk.
        """
        return self.embedder.embed_text(chunks)
