import os
import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader

# This sets up a logger so we can print helpful messages
# instead of just crashing when something goes wrong
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_documents(docs_dir: str = "docs") -> list:
    """
    Reads all PDF, TXT, and MD files from the docs/ folder.
    Returns a list of document objects, each containing
    the text content + info about where it came from.
    """

    # Convert the folder path into a Path object
    # Path() makes it work on both Windows and Mac/Linux
    docs_path = Path(docs_dir)

    # Check the folder actually exists before doing anything
    if not docs_path.exists():
        logger.error(f"Docs folder not found: {docs_dir}")
        return []

    all_documents = []

    # Loop through every file in the docs/ folder
    for file_path in docs_path.iterdir():

        # Skip anything that isn't a file (e.g. subfolders)
        if not file_path.is_file():
            continue

        # Get the file extension in lowercase (.pdf, .txt, .md)
        extension = file_path.suffix.lower()

        try:
            # Pick the right loader based on file type
            if extension == ".pdf":
                loader = PyPDFLoader(str(file_path))

            elif extension == ".txt":
                # encoding="utf-8" handles special characters
                loader = TextLoader(str(file_path), encoding="utf-8")

            elif extension == ".md":
                loader = UnstructuredMarkdownLoader(str(file_path))

            else:
                # Skip files we don't support (.docx, .jpg, etc.)
                logger.warning(f"Skipping unsupported file type: {file_path.name}")
                continue

            # Actually read the file and get the documents
            documents = loader.load()

            # Skip empty files
            if not documents:
                logger.warning(f"No content found in: {file_path.name}")
                continue

            logger.info(f"Loaded {len(documents)} page(s) from: {file_path.name}")
            all_documents.extend(documents)

        except Exception as e:
            # If one file fails, log it and keep going
            # Never crash the whole system for one bad file
            logger.error(f"Failed to load {file_path.name}: {e}")
            continue

    logger.info(f"Total documents loaded: {len(all_documents)}")
    return all_documents


# This block only runs when you execute this file directly
# It's a quick test to make sure the loader works
if __name__ == "__main__":
    docs = load_documents()
    print(f"\nLoaded {len(docs)} document(s)")
    if docs:
        print(f"First document preview:\n{docs[0].page_content[:200]}")