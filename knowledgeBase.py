import psycopg2
import uuid
import json
from sentence_transformers import SentenceTransformer
from embedding_utils import TitanEmbedder

class DatabaseManager:
    def __init__(self, host, port, dbname, username, password):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.username = username
        self.password = password

    def connect(self):
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                dbname=self.dbname,
                user=self.username,
                password=self.password
            )
            self.connection = conn
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")

    def disconnect(self):
        try:
            self.connection.close()
        except psycopg2.Error as e:
            print(f"Error connecting to the database: {e}")
            return None

    def insert_data(self, chunks, embeddings, metadata=None, filter1=None, filter2=None):
        try:
            cursor = self.connection.cursor()
            for i in range(len(chunks)):
                id = str(uuid.uuid4())
                sql = """
                    INSERT INTO bedrock_integration.bedrock_kb (id, embedding, chunks"""
                values = (id, embeddings[i], chunks[i])

                if metadata is not None:
                    sql += ", metadata"
                    values += (metadata,)

                if filter1 is not None:
                    sql += ", filter1"
                    values += (filter1,)

                if filter2 is not None:
                    sql += ", filter2"
                    values += (filter2,)

                sql += ") VALUES (" + "%s, " * len(values)
                sql = sql[:-2] + ")"

                cursor.execute(sql, values)
            self.connection.commit()
            cursor.close()
            print("Data inserted successfully.")
        except psycopg2.Error as e:
            print(f"Error inserting data: {e}")

class DocumentRetriever:
    def __init__(self, embedding_model, db_manager):
        self.model_embeddings = embedding_model
        self.db_manager = db_manager
        self.conn = self.db_manager.connection

    def retrieve_docs(self, query, filter1=None, filter2=None):
        # Convert the query to embedding JSON format
        query_embeds = self.model_embeddings.embed_text([query])
        embedding_json = json.dumps(query_embeds[0])

        if self.conn is None:
            return None

        # Execute the SQL query
        return self.execute_query(embedding_json, filter1, filter2)
        
    def execute_query(self, embedding_json, filter1=None, filter2=None):
        try:
            cursor = self.conn.cursor()

            # Define your SQL query
            sql = """
                SELECT chunks FROM bedrock_integration.bedrock_kb
            """
            # Add filters if provided
            if filter1 is not None:
                sql += f"WHERE filter1 = '{filter1}'"
                if filter2 is not None:
                    sql += f" AND filter2 = '{filter2}'"

            sql += """
                ORDER BY embedding <-> %s
                LIMIT 5;
            """

            # Execute the SQL query
            cursor.execute(sql, (embedding_json,))

            # Fetch all the results
            rows = cursor.fetchall()

            # Close the cursor and connection
            cursor.close()

            return rows
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None




