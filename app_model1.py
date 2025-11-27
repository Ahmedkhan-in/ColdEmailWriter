# Updated Imports: Remove Groq, add ChatHuggingFace
from langchain_huggingface import ChatHuggingFace
from langchain_huggingface import HuggingFaceEndpoint # Required for the ChatHuggingFace backend
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.output_parsers import JsonOutputParser
from langchain_huggingface import HuggingFaceEmbeddings # Keep for vector search
import pandas as pd
import chromadb
import uuid 
import streamlit as st
import os
load_dotenv()

# --- 1. Hugging Face LLM Setup ---

# Select a suitable Hugging Face LLM model ID. 
# CRITICAL: This model must have a deployed endpoint available on the Hugging Face Hub.
# Example: 'google/gemma-7b-it' or 'mistralai/Mistral-7B-Instruct-v0.2'
LLM_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"

# 1a. Initialize the HuggingFaceEndpoint
# This tool handles the API communication with the Hugging Face Inference API.
# It requires the HUGGINGFACEHUB_API_TOKEN environment variable.
hf_endpoint = HuggingFaceEndpoint(
    repo_id=LLM_MODEL_ID,
    task="text-generation",
    # Set a high max_new_tokens for generation tasks like email/JSON extraction
    max_new_tokens=2048, 
    # Use the token from your .env file
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN") 
)

# 1b. Wrap the endpoint in the LangChain Chat model format
llm = ChatHuggingFace(llm=hf_endpoint)

# --- Hugging Face Embedding Setup (for ChromaDB, unchanged) ---
embedding_model_name = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

# This LLM is used to generate cold emails. This tools helps Software and AI services companies send cold emails to their potential clients.
st.title("Cold Email Generator for Software and AI Services Companies")
st.write("Enter the URL of job description to generate a personalized cold email:")
job_url = st.text_input("Job Description URL")

if st.button("Generate Cold Email"):
    try:
        loader = WebBaseLoader(job_url)
        pages = loader.load()
        if not pages:
            st.error("Failed to load page content from the URL.")
        else:
            page_data = pages.pop().page_content

            prompt = load_prompt("JobExtractorPrompt.json")
            # The extraction chain now uses the Hugging Face LLM
            chains = prompt | llm 
            # Note: For JSON extraction, you may need to adjust your prompt or use a specialized
            # model to ensure it produces valid JSON output.
            response = chains.invoke({"job_posting_text": page_data})

            # Check if response.content is directly available or needs extraction from response object
            # HuggingFace chat models typically return a message with content
            content_to_parse = response.content if hasattr(response, 'content') else str(response)

            json_parser = JsonOutputParser()
            parsed_response = json_parser.parse(content_to_parse)

            # ... (Rest of the code for skill extraction and ChromaDB query remains the same) ...

            # ensure parsed_response is a dict and extract skills safely
            if not isinstance(parsed_response, dict):
                st.error("Unexpected extractor output format.")
                raise RuntimeError("Extractor did not return a dict")

            # build a safe query_texts list for chroma
            skills = parsed_response.get("skills") or parsed_response.get("skillset") or ""
            if isinstance(skills, list):
                query_texts = [s for s in skills if isinstance(s, str) and s.strip()]
            elif isinstance(skills, str) and skills.strip():
                query_texts = [skills.strip()]
            else:
                # fallback to using a short snippet of the job posting
                snippet = (parsed_response.get("title") or "") or page_data[:1000]
                query_texts = [snippet] if snippet else []
            
            if not query_texts:
                st.error("No skills or job text found to query the portfolio.")
                raise RuntimeError("Empty query_texts for vector search")

            df = pd.read_csv("my_portfolio.csv")

            client = chromadb.PersistentClient('vectorstore')
            # Pass the Hugging Face Embeddings to ChromaDB
            collection = client.get_or_create_collection(
                name="portfolio", 
                embedding_function=embeddings
            )
            
            if not collection.count():
                for _, row in df.iterrows():
                    tech = row.get('Techstack') if isinstance(row, dict) else row['Techstack'] if 'Techstack' in row else None
                    links = row.get('Links') if isinstance(row, dict) else row['Links'] if 'Links' in row else None
                    if pd.isna(tech) or not str(tech).strip():
                        continue
                    collection.add(
                        documents=[str(tech)],
                        metadatas=[{"links": str(links) if links is not None else ""}],
                        ids=[str(uuid.uuid4())]
                    )

            # Query with a validated list of texts
            try:
                result = collection.query(query_texts=query_texts, n_results=2)
            except Exception as qerr:
                st.error(f"Vector query failed: {qerr}")
                raise

            metadatas = result.get('metadatas', [])

            # normalize metadatas to a flat list of link strings
            link_values = []
            for item in metadatas:
                if isinstance(item, list):
                    for m in item:
                        if isinstance(m, dict) and 'links' in m and m['links']:
                            link_values.append(str(m['links']))
                elif isinstance(item, dict) and 'links' in item and item['links']:
                    link_values.append(str(item['links']))
            link_list_str = ", ".join(link_values) if link_values else ""

            prompt_email = load_prompt("email_prompt.json")
            # The email generation chain now uses the Hugging Face LLM
            chain_email = prompt_email | llm
            res = chain_email.invoke({"job_description": str(parsed_response), "link_list": link_list_str})

            st.subheader("Generated Cold Email:")
            st.write(res.content)
            print(res.content)

    except Exception as e:
        st.error(f"Error: {e}")