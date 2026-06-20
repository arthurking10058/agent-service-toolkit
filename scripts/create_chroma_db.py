import os
import shutil

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from the .env file
load_dotenv()

DEFAULT_COMPATIBLE_EMBEDDING_MODEL = "text-embedding-v3-small"


def create_embedding_model() -> OpenAIEmbeddings:
    """优先使用 OpenAI-compatible embeddings，其次才回退到 OpenAI embeddings。"""
    compatible_api_key = os.getenv("COMPATIBLE_API_KEY")
    compatible_base_url = os.getenv("COMPATIBLE_BASE_URL")
    compatible_embedding_model = os.getenv("COMPATIBLE_EMBEDDING_MODEL")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if compatible_api_key and compatible_base_url:
        return OpenAIEmbeddings(
            model=compatible_embedding_model or DEFAULT_COMPATIBLE_EMBEDDING_MODEL,
            openai_api_base=compatible_base_url,
            openai_api_key=compatible_api_key,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )

    if openai_api_key:
        return OpenAIEmbeddings(
            api_key=openai_api_key,
            check_embedding_ctx_length=False,
        )

    raise RuntimeError(
        "无法初始化 embedding 模型。请优先配置 COMPATIBLE_API_KEY、"
        "COMPATIBLE_BASE_URL 和可选的 COMPATIBLE_EMBEDDING_MODEL；"
        "如果你确实要走 OpenAI，再配置 OPENAI_API_KEY。"
    )


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db_small_chunks",
    delete_chroma_db: bool = True,
    chunk_size: int = 500,
    overlap: int = 100,
):
    embeddings = create_embedding_model()

    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=f"./{db_name}",
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Load document based on file extension
        # Add more loaders if required, i.e. JSONLoader, TxtLoader, etc.
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue  # Skip unsupported file types

        # Load and split document into chunks
        document = loader.load()
        chunks = text_splitter.split_documents(document)

        # Add chunks to Chroma vector store
        for chunk in chunks:
            chunk_id = chroma.add_documents([chunk])
            if chunk_id:
                print(f"Chunk added with ID: {chunk_id}")
            else:
                print("Failed to add chunk")

        print(f"Document {filename} added to database.")

    print(f"Vector database created and saved in {db_name}.")
    return chroma


if __name__ == "__main__":
    # Path to the folder containing the documents
    folder_path = "./data"

    # Create the Chroma database
    chroma = create_chroma_db(folder_path=folder_path)

    # Create retriever from the Chroma database
    retriever = chroma.as_retriever(search_kwargs={"k": 3})

    # Perform a similarity search
    query = "What's my company's mission and values"
    similar_docs = retriever.invoke(query)

    # Display results
    for i, doc in enumerate(similar_docs, start=1):
        print(f"\n🔹 Result {i}:\n{doc.page_content}\nTags: {doc.metadata.get('source', [])}")
