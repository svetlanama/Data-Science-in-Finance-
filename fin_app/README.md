pip install -r requirements.txt



cd src
BE: uvicorn  credit_scoring_api:app --reload --port 8502
FE: streamlit run credit_scoring_client.py