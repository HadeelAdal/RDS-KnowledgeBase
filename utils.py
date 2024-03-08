import os
import json
import boto3
from botocore.exceptions import ClientError

from embedding_utils import TextProcessor, TextEmbedder
from knowledgeBase2 import DocumentRetriever, DatabaseManager
from bot import format_prompt, invoke_bedrock, format_response

def load_embeddings(model_name):
    """
    Load the text embedder model.

    Args:
        model_name (str): The name of the model to load.

    Returns:
        TextEmbedder: The loaded text embedder model.
    """
    global text_embedder
    text_embedder = TextEmbedder(model_name=model_name)
    return text_embedder

def create_connection(auth):
    """
    Create a connection to the database.

    Args:
        auth (dict): A dictionary containing authentication information for the database.

    Returns:
        DatabaseManager: The database manager object.
    """
    global db_manager
    try:
        host = auth["host"]
        port = auth["port"]
        dbname = auth["database"]
        username = auth["user"]
        password = auth["password"]
        db_manager = DatabaseManager(host, port, dbname, username, password)
        db_manager.connect()
        return db_manager
    except Exception as e:
        # Handle exceptions
        print(f"Error creating database connection: {e}")
        return None

def close_connection(conn):
    """
    Close the connection to the database.

    Args:
        conn: The database connection object.
    """
    try:
        conn.close()
    except Exception as e:
        # Handle exceptions
        print(f"Error closing database connection: {e}")
        
def embed_docs(path, chunk_length=1000, chunk_overlap=100):
    """
    Embed the text data from a file.

    Args:
        text_embedder (TextEmbedder): The text embedder object.
        path (str): The path to the file.
        chunk_length (int): The maximum length of each text chunk.
        chunk_overlap (int): The overlap between text chunks.

    Returns:
        tuple: A tuple containing the list of text chunks and their embeddings.
    """
    try:
        text_processor = TextProcessor(max_chunk_length=chunk_length, overlap=chunk_overlap)
        _, file_extension = os.path.splitext(path)
        if file_extension == '.pdf':
            chunks = text_processor.chunk_pdf_file(path)
        elif file_extension == '.json':
            chunks = text_processor.chunk_json_file(path)
        else:
            raise ValueError("Unsupported file format")
        return chunks, text_embedder.embed_text(chunks)
    except Exception as e:
        # Handle exceptions
        print(f"Error embedding document: {e}")
        return [], []

def insert_data_into_database(chunks, embeddings, metadata=None, filter1=None, filter2=None):
    """
    Insert data into the database.

    Args:
        db_manager (DatabaseManager): The database manager object.
        chunks (list): A list of text chunks.
        embeddings (list): A list of embeddings corresponding to the text chunks.
        metadata (dict): Metadata associated with the data.
        filter1: Additional filter criteria.
        filter2: Additional filter criteria.

    Returns:
        str: The response from the database.
    """
    try:
        return db_manager.insert_data(chunks, embeddings, json.dumps(metadata), filter1, filter2)
    except Exception as e:
        # Handle exceptions
        print(f"Error inserting data into database: {e}")
        return None
        
def retrieve_documents(query, filter1=None, filter2=None):
    """
    Retrieve documents from the database based on a query.

    Args:
        query (str): The query string.
        filter1: Additional filter criteria.
        filter2: Additional filter criteria.

    Returns:
        list: A list of retrieved documents.
    """
    try:
        document_retriever = DocumentRetriever(text_embedder, db_manager)
        return document_retriever.retrieve_docs(query, filter1)
    except Exception as e:
        # Handle exceptions
        print(f"Error retrieving documents: {e}")
        return []

def chatbot(query, filter1=None, filter2=None):
    """
    Interact with the chatbot.

    Args:
        query (str): The query string.
        filter1: Additional filter criteria.
        filter2: Additional filter criteria.

    Returns:
        str: The response from the chatbot.
    """
    try:
        docs = retrieve_documents(query, filter1, filter2)
        prompt = format_prompt(query, docs)
        response = invoke_bedrock(prompt)
        return {"Chat": format_response(response), "Documents": docs}
    except Exception as e:
        # Handle exceptions
        print(f"Error interacting with chatbot: {e}")
        return "An error occurred while processing your request."

def get_secret(secret_name, region_name):
    """
    Get a secret from AWS Secrets Manager.

    Args:
        secret_name (str): The name of the secret.
        region_name (str): The AWS region name.

    Returns:
        str: The secret value.
    """
    try:
        session = boto3.session.Session()
        client = session.client(service_name='secretsmanager', region_name=region_name)
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response['SecretString']
        return json.loads(secret)
    except ClientError as e:
        # Handle exceptions
        print(f"Error retrieving secret: {e}")
        return None
    except Exception as e:
        # Handle other exceptions
        print(f"An error occurred: {e}")
        return None

def get_file_extension(file_path):
    """
    Get the file extension from a file path.

    Args:
        file_path (str): The file path.

    Returns:
        str: The file extension.
    """
    try:
        _, extension = os.path.splitext(file_path)
        return extension
    except Exception as e:
        # Handle exceptions
        print(f"Error getting file extension: {e}")
        return ""
