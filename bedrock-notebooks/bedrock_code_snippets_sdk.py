import sys

# Check major and minor version
if sys.version_info.major == 3 and sys.version_info.minor < 11:
    print("This script requires Python 3.11 or higher!")
    sys.exit(1)

import boto3
import json
import logging
from botocore.client import Config
from botocore.exceptions import ClientError 


## -- Bedrock Config
bedrock_config = Config(connect_timeout=120, read_timeout=120, retries={'max_attempts': 0})
bedrock_client = boto3.client('bedrock-runtime')
bedrock_agent_client = boto3.client("bedrock-agent-runtime", config=bedrock_config)
region = 'us-east-1'


# Queries a knowledge base and retrieves information from it.
def retrieve(query, kbId, numberOfResults=5):
    return bedrock_agent_client.retrieve(
        retrievalQuery= {
            'text': query
        },
        knowledgeBaseId=kbId,
        retrievalConfiguration= {
            'vectorSearchConfiguration': {
                'numberOfResults': numberOfResults,
                'overrideSearchType': "HYBRID", # optional
            }
        }
    )


# Queries a knowledge base and generates responses based on the retrieved results. 
# The response cites up to five sources but only selects the ones that are relevant to the query.
def retrieveAndGenerate(input, kbId, model_id):
    model_arn = f'arn:aws:bedrock:us-east-1::foundation-model/{model_id}'
    return bedrock_agent_client.retrieve_and_generate(
        input={
            'text': input
        },
        retrieveAndGenerateConfiguration={
            'type': 'KNOWLEDGE_BASE',
            'knowledgeBaseConfiguration': {
                'knowledgeBaseId': kbId,
                'modelArn': model_arn,
                'retrievalConfiguration': {
                    'vectorSearchConfiguration': {
                        'overrideSearchType': 'HYBRID'
                    }
                }
            }
        }

    )
    

# Get response from bedrock via Converse API
# Support for converse API - https://docs.aws.amazon.com/bedrock/latest/userguide/conversation-inference.html
def generate_conversation(bedrock_client,
                          model_id,
                          system_prompts,
                          messages):
    """
    Sends messages to a model.
    Args:
        bedrock_client: The Boto3 Bedrock runtime client.
        model_id (str): The model ID to use.
        system_prompts (JSON) : The system prompts for the model to use.
        messages (JSON) : The messages to send to the model.

    Returns:
        response (JSON): The conversation that the model generated.

    """

    logging.info("Generating message with model %s", model_id)

    # Inference parameters to use.
    temperature = 0.5
    top_p = 1

    # Base inference parameters to use.
    inference_config = {"temperature": temperature}

    # Additional inference parameters to use. For additional information - https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-jamba.html#api-inference-examples-a2i-jamba
    additional_model_fields = {"top_p": top_p}

    # Send the message.
    response = bedrock_client.converse(
        modelId=model_id,
        messages=messages,
        system=system_prompts,
        inferenceConfig=inference_config,
        additionalModelRequestFields=additional_model_fields
    )

    # Log token usage.
    token_usage = response['usage']
    logging.info("Input tokens: %s", token_usage['inputTokens'])
    logging.info("Output tokens: %s", token_usage['outputTokens'])
    logging.info("Total tokens: %s", token_usage['totalTokens'])
    logging.info("Stop reason: %s", response['stopReason'])

    return response


# List All Foundational Models available in your AWS account
def get_available_bedrock_models():
    endpoint_url = 'https://bedrock.us-east-1.amazonaws.com/'
    bedrock = boto3.client(service_name='bedrock',
                       region_name=region,
                       endpoint_url=endpoint_url)
    return bedrock.list_foundation_models()


def main():
    """
    Entrypoint for foundational model example.
    """

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s: %(message)s")

    # For all model IDs, please visit https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html
    model_id = "ai21.jamba-instruct-v1:0"

    # Setup the system prompts and messages to send to the model.
    system_prompts = [{"text": "You are an app that creates playlists for a radio station that plays rock and pop music."
                       "Only return song names and the artist."}]
    message_1 = {
        "role": "user",
        "content": [{"text": "Create a list of 3 pop songs."}]
    }
    message_2 = {
        "role": "user",
        "content": [{"text": "Make sure the songs are by artists from the United Kingdom."}]
    }
    messages = []

    try:

        bedrock_client = boto3.client(service_name='bedrock-runtime')

        # Start the conversation with the 1st message.
        messages.append(message_1)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages)

        # Add the response message to the conversation.
        output_message = response['output']['message']
        messages.append(output_message)

        # Continue the conversation with the 2nd message.
        messages.append(message_2)
        response = generate_conversation(
            bedrock_client, model_id, system_prompts, messages)

        output_message = response['output']['message']
        messages.append(output_message)

        # Show the complete conversation.
        for message in messages:
            print(f"Role: {message['role']}")
            for content in message['content']:
                print(f"Text: {content['text']}")
            print()

    except ClientError as err:
        message = err.response['Error']['Message']
        logging.error("A client error occurred: %s", message)
        print(f"A client error occured: {message}")

    else:
        print(
            f"Finished generating text with model {model_id}.")


if __name__ == "__main__":
    main()