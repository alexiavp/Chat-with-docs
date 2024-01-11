# Chat with Docs

This repository contains my final degree project, it enables users to interact with their documents using a chat interface. 
It leverages OpenAI's language model and Google Drive integration to provide a conversational experience.

## Files

- `chat.py`: This script creates a Streamlit web application that lets the user interact with the documents
loaded in the DeepLake instance, generating chatbot responses using OpenAI GPT-3.5-turbo-16k.
- `templates/prompt.py`: Prompt created with the instructions of how the LLM has to act.

## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

- Python (version 3.6 or higher)

Create your own service account in Google Cloud, you can follow the instructions [here](https://medium.com/@matheodaly.md/create-a-google-cloud-platform-service-account-in-3-steps-7e92d8298800) and save the JSON file with the keys in the working repository as `credential-key.json`. Also, you have to share the Google Drive folder with the service account mail.

### Installation

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/alexiavp/Chat-with-docs.git
   cd Chat-with-docs
   ```

2. Install all the required dependencies with `pip`:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys:

   - Obtain an [OpenAI API key](https://openai.com/).
   - Obtain your [Activeloop credentials](https://www.activeloop.ai/) (username and token).

## Usage

1. Run the main script:

   ```bash
   streamlit run chat.py
   ```

2. Enter your OpenAI API key and Activeloop credentials in the sidebar, making sure are correctly writed.

3. Use the chat interface to interact with your documents.

## Features

- **Document Loading:** Automatically downloads documents from Google Drive.
- **Chat Interface:** Interact with documents using a conversational interface.
- **Question-Answering:** Use OpenAI's language model to answer user queries.


## Acknowledgments

- [OpenAI](https://www.openai.com/) for providing the powerful language model.
- [Activeloop](https://www.activeloop.ai/) for the data versioning and collaboration tools.