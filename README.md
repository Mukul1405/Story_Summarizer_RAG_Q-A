# Story_Summarizer_RAG_Q-A

## Firstly Create a Environment
1. Go to repo path and type "conda create -p venv python==3.12 -y"
2. Then you will notice a command on terminal with "conda activate <your/path>". Copy that and paste that command
3. Then run -> "pip install -r requirements.txt"

## Creating an Environment file to load the tokens
1. Create a file named exactly ".env"
2. Then paste this content in key value format

- AZURE_DEVOPS_ORG=dpwhotfsonline  
- AZURE_DEVOPS_PROJECT=DTLP  #enter your project
- AZURE_DEVOPS_PAT="your Personal Acess Token"

- model=openai/gpt-oss-120b  #you can also use another model 
- GROQ_API_KEY="create your groq key"
- HF_TOKEN="create your huggingface key"

3. Now save the file

## Run the code
streamlit run multiple.py

