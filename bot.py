import boto3
import json

# Constants
SERVICE_NAME = "bedrock-runtime"
REGION_NAME = "us-east-1"
MODEL_ID = "anthropic.claude-v2:1"
CONTENT_TYPE = "application/json"
ACCEPT = "*/*"
ANTHROPIC_VERSION = "bedrock-2023-05-31"

def generate_request_body(prompt: str) -> dict:
    """
    Generate and return the body for the request.

    Args:
        prompt (str): The user prompt.

    Returns:
        dict: A dictionary containing the request body.
    """
    return {
        "prompt": f"Human: {prompt}\nAssistant:",
        "max_tokens_to_sample": 100,
        "temperature": 1,
        "top_k": 250,
        "top_p": 0.9,
        "stop_sequences": ["\n\nHuman:"],
        "anthropic_version": ANTHROPIC_VERSION
    }

def format_prompt(query, info):
    """
    Format the user prompt.

    Args:
        item: item information

    Returns:
        str: The formatted user prompt.
    """

    prompt = f"""
\n\nHuman: 

System: 
Please read the following information closely and then answer the question at the end based on your understanding, keep the answer short and concise:
\n{info[0]}
\n{info[1]}
\n{info[2]}
\n{info[3]}
\n{info[4]}

Question: {query}

\n\nAssistant: Based on the provided information, the correct answer is:
"""
    return prompt

def invoke_bedrock(prompt):
    bedrock = boto3.client(service_name=SERVICE_NAME, region_name=REGION_NAME)

    body = json.dumps(generate_request_body(prompt))

    kwargs = {
        "modelId": MODEL_ID,
        "contentType": CONTENT_TYPE,
        "accept": ACCEPT,
        "body": body
    }
    response = bedrock.invoke_model(**kwargs)
    return response

def format_response(response):
    reply = json.loads(response['body'].read())
    return reply['completion']