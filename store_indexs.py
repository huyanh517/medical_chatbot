from src.helper import load_file, text_split, get_hugging_face_embeddings
from langchain_pinecone import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data=load_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = get_hugging_face_embeddings()

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name="medical",
    embedding=embeddings,
)
