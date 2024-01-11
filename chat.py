import openai
import deeplake
import os
import io
import pandas as pd
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
from openai.error import AuthenticationError
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyMuPDFLoader
from langchain.document_loaders import UnstructuredExcelLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from templates.prompt import qa_template
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


# Load enviroment, initialize chat history and connect with the
# service account of Google Cloud with the credentials file
history = []
scope = ['https://www.googleapis.com/auth/drive']
service_account_json_key = 'credential-key.json'
credentials = service_account.Credentials.from_service_account_file(
                            filename=service_account_json_key,
                            scopes=scope)
service = build('drive', 'v3', credentials=credentials)


# Function downloads all the files found in the Google Drive folder and saves
# them in the local docs folder. It returns a dataframe with all the files
def download_files():

    results = service.files().list(pageSize=1000, fields="nextPageToken, files(id, name, mimeType)", q='').execute()
    items = results.get('files', [])

    data = []
    for row in items:
        if row["mimeType"] != "application/vnd.google-apps.folder":
            row_data = []
            row_data.append(row["id"])
            row_data.append(row["name"])
            row_data.append(row["mimeType"])
            row_data.append(False)
            data.append(row_data)

            try:
                request_file = service.files().get_media(fileId=row["id"])
                file = io.BytesIO()
                downloader = MediaIoBaseDownload(file, request_file)
                done = False
                while done is False:
                    done = downloader.next_chunk()
            except HttpError as error:
                st.error(F'An error occurred: {error}', icon="🚨")

            file_retrieved: str = file.getvalue()
            with open(f"docs/{row['name']}", 'wb') as f:
                f.write(file_retrieved)

    df = pd.DataFrame(data, columns=['id', 'name', 'type_of_file', 'loaded'])
    return df


# Function that loads the files in the directory given,
# and returns dataset. The files are loaded depending on
# the type of file.
def load_doc(name_dir, dataset_path, embeddings, token, files):

    docs = []

    for dirpath, dirnames, filenames in os.walk(name_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for file in filenames:
            file_path = os.path.join(dirpath, file)

            # Skip dotfiles
            if file.startswith("."):
                continue

            result = files.query(f'name == "{file}" and loaded == True')

            if not result.empty:
                continue

            match (os.path.splitext(file)[1]):
                case ".pdf":
                    # Load file using PyPDFLoader
                    loader = PyMuPDFLoader(file_path, extract_images=True)
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
                case ".pptx":
                    # Load file using UnstructuredPowerPointLoader
                    loader = UnstructuredPowerPointLoader(file_path,
                                                          mode="elements")
                    docs.extend(loader.load())

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=24)
    result = text_splitter.split_documents(docs)
    db = DeepLake(dataset_path=dataset_path, embedding=embeddings, token=token)
    db.add_documents(result)
    return db


# Function that checks if new files are loaded in the folder. If it finds new
# files it downloads and adds to the older list of files.
def check_new_files(files):
    new = False
    results = service.files().list(pageSize=1000, fields="nextPageToken, files(id, name, mimeType)", q='').execute()
    items = results.get('files', [])

    data = []
    for row in items:
        if row["mimeType"] != "application/vnd.google-apps.folder":
            if row["id"] not in files["id"].values:
                new = True
                row_data = []
                row_data.append(row["id"])
                row_data.append(row["name"])
                row_data.append(row["mimeType"])
                row_data.append(False)
                data.append(row_data)

                try:
                    request_file = service.files().get_media(fileId=row["id"])
                    file = io.BytesIO()
                    downloader = MediaIoBaseDownload(file, request_file)
                    done = False
                    while done is False:
                        done = downloader.next_chunk()
                except HttpError as error:
                    st.error(F'An error occurred: {error}', icon="🚨")

                file_retrieved: str = file.getvalue()
                with open(f"docs/{row['name']}", 'wb') as f:
                    f.write(file_retrieved)
    if new:
        df = pd.DataFrame(data, columns=['id', 'name', 'type_of_file',
                                         'loaded'])
        merged = pd.merge(files, df, on=['id', 'name', 'type_of_file',
                                         'loaded'],
                          how='outer')
        files = merged
    return new, files


# It creates an emty dataset in which al the chuncks will be loaded later.
def create_empty_dataset(dataset_path, token):
    ds = deeplake.empty(dataset_path,  token=token, overwrite=True)
    ds.create_tensor("ids")
    ds.create_tensor("metadata")
    ds.create_tensor("embedding")
    ds.create_tensor("text")


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


# Main function for the chat functioning
def main():

    # Messages to show in the chat if needed
    gb_msg = "Thank you for using our service! Have a great day!"
    new_msg = "New files founded and loaded!"

    # Define title, caption and initialize the chat of the API
    st.title("💬 Chat")
    st.caption("🚀 A chat to interact with your documents!")

    # Define the sidebar with text to introduce the credentials
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
        st.write("Make sure they are correct 👀")

        add_vertical_space(4)

        # Definition of the button to look for new files in the folder,
        # downloading and creates again the qa variable
        if "loaded" in st.session_state:
            st.write("Last time the folder was checked ⏰:")
            st.write(f"{st.session_state.hour_check}")
            if st.button('🔍 Look for new files!', type='primary'):
                date = datetime.now()
                date_formated = date.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.hour_check = date_formated
                new, st.session_state.files = check_new_files(
                    st.session_state.files)
                if new:
                    with st.spinner("New files founded! Indexing them ⏳"):
                        db = load_doc("docs", st.session_state.path,
                                      st.session_state.embeddings,
                                      DeepLake.token,
                                      st.session_state.files)
                        qa = ConversationalRetrievalChain.from_llm(
                            ChatOpenAI(model="gpt-3.5-turbo-16k",
                                       temperature=0.2,
                                       streaming=True,
                                       openai_api_key=openai.api_key),
                            retriever=db.as_retriever(qa_template=qa_template),
                            memory=st.session_state.memory
                            )
                        st.session_state.qa = qa
                        st.session_state.messages.append({"role": "system",
                                                          "content": new_msg})
        "[Source Code](https://https://github.com/alexiavp/Chat-with-docs)"

    # If the user doesn't introduce the OpenAI Key
    if not openai.api_key:
        st.info("Please add your OpenAI API key to continue.")
    # If the user doesn't introduce the Activeloop credentials
    if not DeepLake.token or not DeepLake.username:
        st.info("Please add your Activeloop credentials to continue.")

    # When all the credentials are introduced
    if openai.api_key and DeepLake.token and DeepLake.username:

        # An if to only create the dataset and load the files once
        if "loaded" not in st.session_state:
            # Define the dataset_path where the files are loaded
            dataset_path = f"hub://{DeepLake.username}/PDF"
            st.session_state.path = dataset_path

            # Create the embeddings used in the vector store
            embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

            # Treat the documents in the Google Drive, downloads them, creates
            # empty dataset treats the files in docs and loads them and saves
            # the date and hour
            with st.spinner("Indexing documents... this might take a while⏳"):
                files = download_files()
                # create_empty_dataset(dataset_path, DeepLake.token)
                # db=load_doc("docs", dataset_path, embeddings, DeepLake.token,
                #              files)
                db = DeepLake(dataset_path=dataset_path, embedding=embeddings,
                              token=DeepLake.token, read_only=True)
                date = datetime.now()
                date_formated = date.strftime("%Y-%m-%d %H:%M:%S")
                st.session_state.hour_check = date_formated

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
            # Save useful variables in session_state
            st.session_state.qa = qa
            st.session_state.files = files
            st.session_state.memory = memory
            st.session_state.embeddings = embeddings
            st.session_state.loaded = True
        else:
            # Load qa variable from session_state
            qa = st.session_state.qa

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Writes all the messages in the chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Reads what the user introduces to the chat input and search for the
        # correct answer and writes the answer in the chat
        if query := st.chat_input("Write your question and press Enter..."):

            # If the user writes exit then
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

            # Writes the query in the chat
            st.session_state.messages.append({"role": "user",
                                              "content": query})
            st.chat_message("user").markdown(query)

            # While the we're searching for the answer it shows a spinner
            with st.spinner('Answering your question...'):
                try:
                    response = qa({"question": query, "chat_history": history})
                    # Adds the answer to the chat
                    st.session_state.messages.append({"role": "assistant",
                                                      "content":
                                                      response["answer"]})
                    st.chat_message("assistant").markdown(response["answer"])
                    history.append((query, response["answer"]))
                except AuthenticationError as error:
                    st.error(F'An error occurred: {error}', icon="🚨")
                    response = "OpenAI key incorrect!"
                    st.session_state.messages.append({"role": "system",
                                                      "content":
                                                      response})
                    st.chat_message("system").markdown(response)


main()
