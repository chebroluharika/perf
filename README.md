# agentic-AI-Perf-bot
Agentic AI implementation for Ceph Performance

# Pre-requisite
1. Ensure ollama is installed and run LLM from ollama ( Install ollama first)
  ```
  # for MAC-OS
  # brew install ollama
  # ollama run llama3
  ```

# To run the UI Bot (frontend) code
1. Enable virtualenv
    ```
    # python3 -m venv venv
    # source venv/bin/activate
    ```
2. git clone git@github.ibm.com:Sunil-Kumar-N2/agentic-ai-perf-bot.git
3. cd agentic-ai-perf-bot/
4. pip install -r backend/requirements.txt
5. pip install -r frontend/requirements.txt
6. cd frontend
7. streamlit run app.py
