import openai
import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from templates.prompt import qa_template


# Function to load a file given the name
def load_doc(name):
    # Load file using PyPDFLoader
    loader = PyPDFLoader(name, extract_images=True)
    pages = loader.load()
    # Split the documents into small chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(pages)
    return docs


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
    # Load enviroment
    load_dotenv()

    history = ""

    docs = load_doc("white_paper.pdf")

    embeddings = OpenAIEmbeddings()

    # Upload the documents in the vector database using the variables in the enviroment
    username = os.environ.get("ACTIVELOOP_USERNAME")
    token = os.environ.get("ACTIVELOOP_TOKEN")
    dataset_path = f"hub://{username}/PDFs"

    db = DeepLake.from_documents(docs, dataset_path=dataset_path,
                                 token=token, embedding=embeddings,
                                 overwrite=True,
                                 )

    # To get answers from the paper loaded in the vector database
    qa = RetrievalQA.from_llm(
        ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.5),
        retriever=db.as_retriever()
    )

    # Endless bucle to execute the chat with the bot
    while True:
        query = input("> ")
        if query == "exit":
            break
        print("Answer from paper:\n")
        print(f"{qa.run(query)}\n")

        print("Answer without paper background:\n")
        response = get_answer(history, query)
        print(f"{response}\n")

        history += f"User: {query}\nBot: {response}\n"


main()
