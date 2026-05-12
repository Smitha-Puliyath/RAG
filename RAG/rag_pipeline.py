import os
import uuid
import pandas as pd

from dotenv import load_dotenv

# ADLS
from azure.storage.filedatalake import DataLakeServiceClient

# Azure AI Search
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery

from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    AzureOpenAIVectorizer,
    AzureOpenAIVectorizerParameters
)

# Azure OpenAI
from openai import AzureOpenAI

load_dotenv()

############################################
# CONFIG
############################################

STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
FILE_SYSTEM_NAME = os.getenv("FILE_SYSTEM_NAME")
FILE_PATH = os.getenv("FILE_PATH")

SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
INDEX_NAME = os.getenv("INDEX_NAME")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
EMBEDDING_DEPLOYMENT = os.getenv("OPENAI_DEPLOYMENT")
CHAT_DEPLOYMENT = os.getenv("CHAT_DEPLOYMENT")

############################################
# OPENAI CLIENT
############################################

openai_client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version="2024-02-01",
    azure_endpoint=OPENAI_ENDPOINT
)

############################################
# STEP 1: READ CSV FROM ADLS
############################################

def read_csv_from_adls():

    service_client = DataLakeServiceClient.from_connection_string(
        STORAGE_CONNECTION_STRING
    )

    file_system_client = service_client.get_file_system_client(
        file_system=FILE_SYSTEM_NAME
    )

    file_client = file_system_client.get_file_client(FILE_PATH)

    download = file_client.download_file()

    downloaded_bytes = download.readall()

    with open("temp.csv", "wb") as f:
        f.write(downloaded_bytes)

    df = pd.read_csv("temp.csv")

    return df

############################################
# STEP 2: CREATE SEARCH INDEX
############################################

def create_search_index():

    index_client = SearchIndexClient(
        endpoint=SEARCH_ENDPOINT,
        credential=AzureKeyCredential(SEARCH_KEY)
    )

    fields = [
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True
        ),

        SearchableField(
            name="content",
            type=SearchFieldDataType.String
        ),

        SearchField(
            name="contentVector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,
            vector_search_profile_name="my-vector-profile"
        )
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="my-hnsw"
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="my-vector-profile",
                algorithm_configuration_name="my-hnsw"
            )
        ]
    )

    index = SearchIndex(
        name=INDEX_NAME,
        fields=fields,
        vector_search=vector_search
    )

    try:
        index_client.create_index(index)
        print("Index created")
    except Exception as e:
        print(e)

############################################
# STEP 3: GENERATE EMBEDDING
############################################

def generate_embedding(text):

    response = openai_client.embeddings.create(
        input=text,
        model=EMBEDDING_DEPLOYMENT
    )

    return response.data[0].embedding

############################################
# STEP 4: UPLOAD DATA TO AI SEARCH
############################################

def upload_documents(df):

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY)
    )

    documents = []

    for _, row in df.iterrows():

        content = (
            f"Medicine: {row['medicine_name']}, "
            f"Sales: {row['sales']}, "
            f"Week: {row['week']}"
        )

        embedding = generate_embedding(content)

        documents.append({
            "id": str(uuid.uuid4()),
            "content": content,
            "contentVector": embedding
        })

    result = search_client.upload_documents(documents)

    print("Documents uploaded")

############################################
# STEP 5: QUERY RAG SYSTEM
############################################

def retrieve(query):

    search_client = SearchClient(
        endpoint=SEARCH_ENDPOINT,
        index_name=INDEX_NAME,
        credential=AzureKeyCredential(SEARCH_KEY)
    )

    query_vector = generate_embedding(query)

    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=3,
        fields="contentVector"
    )

    results = search_client.search(
        search_text=query,
        vector_queries=[vector_query],
        top=3
    )

    docs = []

    for result in results:
        docs.append(result["content"])

    return docs

############################################
# STEP 6: GENERATE FINAL RESPONSE
############################################

def ask_rag(query):

    retrieved_docs = retrieve(query)

    context = "\n".join(retrieved_docs)

    prompt = f"""
    Answer the question using only the context.

    Context:
    {context}

    Question:
    {query}
    """

    response = openai_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content

############################################
# MAIN
############################################

if __name__ == "__main__":

    df = read_csv_from_adls()

    create_search_index()

    upload_documents(df)

    answer = ask_rag("What are the sales in week 1?")

    print("\nFINAL ANSWER:\n")
    print(answer)