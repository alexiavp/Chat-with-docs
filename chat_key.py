import openai
import os
import io
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
# from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from templates.prompt import qa_template
import streamlit as st

# Load enviroment and initialize chat history
history = []


def download_files():
    scope = ['https://www.googleapis.com/auth/drive']
    service_account_json_key = 'credential-key.json'
    credentials = service_account.Credentials.from_service_account_file(
                              filename=service_account_json_key,
                              scopes=scope)
    service = build('drive', 'v3', credentials=credentials)

    # Call the Drive v3 API
    results = service.files().list(pageSize=1000, fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)", q='').execute()
    # get the results
    items = results.get('files', [])

    data = []
    for row in items:
        if row["mimeType"] != "application/vnd.google-apps.folder":
            row_data = []
            try:
                row_data.append(round(int(row["size"])/1000000, 2))
            except KeyError:
                row_data.append(0.00)
            row_data.append(row["id"])
            row_data.append(row["name"])
            row_data.append(row["modifiedTime"])
            row_data.append(row["mimeType"])
            data.append(row_data)
            try:
                request_file = service.files().get_media(fileId=row["id"])
                file = io.BytesIO()
                downloader = MediaIoBaseDownload(file, request_file)
                done = False
                while done is False:
                    done = downloader.next_chunk()
            except HttpError as error:
                print(F'An error occurred: {error}')

        file_retrieved: str = file.getvalue()
        with open(f"docs/{row['name']}", 'wb') as f:
            f.write(file_retrieved)

        cleared_df = pd.DataFrame(data, columns=['size', 'id', 'name',
                                                 'last_modification',
                                                 'type_of_file'])

    return cleared_df


# Function that loads the files in the directory given,
# and returns dataset. The files are loaded depending on
# the type of file.
def load_doc(name_dir, dataset_path, embeddings, token):

    docs = []

    for dirpath, dirnames, filenames in os.walk(name_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for file in filenames:
            file_path = os.path.join(dirpath, file)

            # Skip dotfiles
            if file.startswith("."):
                continue

            match (os.path.splitext(file)[1]):
                case ".pdf":
                    # Load file using PyPDFLoader
                    loader = PyPDFLoader(file_path, extract_images=True)
                    docs.extend(loader.load())
                case ".txt":
                    # Load file using TextLoader
                    loader = TextLoader(file_path, encoding="utf-8")
                    docs.extend(loader.load())
                case ".csv":
                    # Load file using CSVLoader
                    loader = CSVLoader(file_path, csv_args={
                        'delimiter': ',',
                        'quotechar': '"'}
                        )
                    docs.extend(loader.load())
                case ".xls":
                    # Load file using UnstructuredExcelLoader
                    loader = UnstructuredExcelLoader(file_path,
                                                     mode="elements")
                    docs.extend(loader.load())
                case ".xlsx":
                    # Load file using UnstructuredExcelLoader
                    loader = UnstructuredExcelLoader(file_path,
                                                     mode="elements")
                    docs.extend(loader.load())
                case ".docx":
                    # Load file using Doc2txtLoader
                    loader = Docx2txtLoader(file_path)
                    docs.extend(loader.load())

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    result = text_splitter.split_documents(docs)
    db = DeepLake.from_documents(result, dataset_path=dataset_path,
                                 token=token, embedding=embeddings,
                                 overwrite=True
                                 )
    return db


# TODO: create a function that compares if a new file is added in the folder
# list llistat de pandas id no canvies ni quan es modifica el nom

# Function to get answer using openai chatbot without context
def get_answer(history, query):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    prompt = qa_template.replace(
        "{conversation history}", history).replace(
            "{question}", query)

    response = openai.Completion.create(
        engine="text-davinci-003", prompt=prompt, max_tokens=2048, n=1,
        stop=None, temperature=0.5
    )
    response_text = response["choices"][0]["text"]
    chat_response = response_text.strip()

    return chat_response


def main():

    gb_msg = "Thank you for using our service! Have a great day!"

    # Define title, caption and initialize the chat of the API
    st.title("💬 Chat")
    st.caption("🚀 A chat to interact with your documents!")

    with st.sidebar:
        st.write("Please enter all the parameters to continue:")
        openai.api_key = st.text_input("OpenAI API Key",
                                       key="langchain_search_api_key_openai",
                                       type="password")
        DeepLake.username = st.text_input("Activeloop Username",
                                          key="activeloop_username")
        DeepLake.token = st.text_input("Activeloop Token",
                                       key="activeloop_token",
                                       type="password")
        "[Source Code](https://https://github.com/alexiavp/Chat-with-docs)"

    if not openai.api_key:
        st.info("Please add your OpenAI API key to continue.")
    if not DeepLake.token or not DeepLake.username:
        st.info("Please add your Activeloop credentials to continue.")

    if openai.api_key and DeepLake.token and DeepLake.username:

        if "loaded" not in st.session_state:
            # Create the varibles with the info loaded in the file .env
            # username = os.environ.get("ACTIVELOOP_USERNAME")
            # token = os.environ.get("ACTIVELOOP_TOKEN")
            dataset_path = f"hub://{DeepLake.username}/docs7"

            # Create the embeddings used in the vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

            # Treat the documents in the directory docs and load them
            # in the new dataset that it's created in the function.
            with st.spinner("Indexing documents... this might take a while⏳"):
                list_docs = download_files()
                # load_doc("docs", dataset_path, embeddings, DeepLake.token)
                # Load a dataset that already exists
                db = DeepLake(dataset_path=dataset_path,
                              token=DeepLake.token,
                              embedding=embeddings, read_only=True
                              )

            # Create the memory where the chat history is loaded
            memory = ConversationBufferMemory(memory_key="chat_history",
                                              return_messages=True)
            # To get answers from the paper loaded in the vector database
            qa = ConversationalRetrievalChain.from_llm(
                ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2,
                           streaming=True, openai_api_key=openai.api_key),
                retriever=db.as_retriever(qa_template=qa_template),
                memory=memory
                )
            st.session_state.loaded = True
            st.session_state.qa = qa
            st.session_state.list = list_docs
        else:
            qa = st.session_state.qa

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if query := st.chat_input("Write your question and press Enter..."):
            if query.lower() == "exit":
                st.session_state.messages.append({"role": "user",
                                                  "content": query})
                st.chat_message("user").markdown(query)
                st.session_state.messages.append({"role": "assistant",
                                                  "content": gb_msg})
                st.chat_message("assistant").markdown(gb_msg)
                for key in st.session_state.keys():
                    del st.session_state[key]
                st.stop()
                exit()
            st.session_state.messages.append({"role": "user",
                                              "content": query})
            st.chat_message("user").markdown(query)
            with st.spinner('Answering your question...'):
                response = qa({"question": query, "chat_history": history})
            st.session_state.messages.append({"role": "assistant",
                                              "content": response["answer"]})
            st.chat_message("assistant").markdown(response["answer"])
            history.append((query, response["answer"]))


main()
