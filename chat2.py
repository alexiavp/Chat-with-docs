import openai
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from templates.prompt import qa_template


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

    # Split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    result = text_splitter.split_documents(docs)
    db = DeepLake.from_documents(result, dataset_path=dataset_path,
                                 token=token, embedding=embeddings, overwrite=True
                                 )
    return db


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
    chatbot_response = response_text.strip()

    return chatbot_response


def main():
    # Load enviroment and initialize the chat history
    load_dotenv()
    history = []

    # Define title, caption and initialize the chat of the API
    st.title("💬 Chat")
    st.caption("🚀 A chat to interact with your documents!")

    # Create the varibles with the info loaded in the file .env
    username = os.environ.get("ACTIVELOOP_USERNAME")
    token = os.environ.get("ACTIVELOOP_TOKEN")
    dataset_path = f"hub://{username}/PDF3"

    # Create the embeddings used in the vector store
    embeddings = OpenAIEmbeddings()

    # Treat the documents in docs and load them in the new dataset that creates
    load_doc("docs", dataset_path, embeddings, token)

    # Load a dataset that already exists
    db = DeepLake(dataset_path=dataset_path, token=token,
                  embedding=embeddings, read_only=True
                  )

    # To get answers from the paper loaded in the vector database
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.2),
        retriever=db.as_retriever(qa_template=qa_template)
    )

    # Endless bucle to execute the chat with the bot
    while True:
        query = input("> ")
        if query.lower() == "exit":
            break
        response = qa({"question": query, "chat_history": history})
        print(f'{response["answer"]}')
        history.append((query, response["answer"]))


main()
