import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone
from langchain.schema import Document

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader(
        "./mediumblogs/mediumblog1.txt"
    )
    document = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(document)

    embeddings = OpenAIEmbeddings()

    # docsearch = Pinecone.from_documents(
    #     docs, embeddings, index_name="markiyan-test-index"
    # )

    # if you already have an index, you can load it like this
    docsearch = Pinecone.from_existing_index("markiyan-test-index", embeddings)

    qa = VectorDBQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    )
    query = "What is a vector DB? Give me a 15 word answer for a begginner"
    result = qa({"query": query})
    print(result)
