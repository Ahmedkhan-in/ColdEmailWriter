# # ...existing code...
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv
# from langchain_community.document_loaders import WebBaseLoader
# from langchain_core.prompts import PromptTemplate, load_prompt
# from langchain_core.output_parsers import JsonOutputParser
# import pandas as pd
# import chromadb
# import uuid 
# import streamlit as st
# load_dotenv()

# llm = ChatGroq(model="llama-3.3-70b-versatile")

# # This LLM is used to generate cold emails. This tools helps Software and AI services companies send cold emails to their potential clients.
# st.title("Cold Email Generator for Software and AI Services Companies")
# st.write("Enter the URL of job description to generate a personalized cold email:")
# job_url = st.text_input("Job Description URL")

# if st.button("Generate Cold Email"):
#     try:
#         loader = WebBaseLoader(job_url)
#         page = loader.load()
#         if not page:
#             st.error("Failed to load page content from the URL.")
#         else:
#             page_data = page.pop().page_content

#             prompt = load_prompt("JobExtractorPrompt2.json")
#             chains = prompt | llm
#             response = chains.invoke({"job_posting_text": page_data})

#             json_parser = JsonOutputParser()
#             parsed_response = json_parser.parse(response.content)

#             df = pd.read_csv("my_portfolio.csv")

#             client = chromadb.PersistentClient('vectorstore')
#             collection = client.get_or_create_collection(name="portfolio")
#             if not collection.count():
#                 for _, row in df.iterrows():
#                     collection.add(
#                         documents=[row['Techstack']],
#                         metadatas=[{"links": row['Links']}],
#                         ids=[str(uuid.uuid4())]
#                     )

#             job = parsed_response

#             result = collection.query(query_texts=job.get('skills', ""), n_results=2)
#             metadatas = result.get('metadatas', [])

#             # normalize metadatas to a flat list of link strings
#             link_values = []
#             for item in metadatas:
#                 if isinstance(item, list):
#                     for m in item:
#                         if isinstance(m, dict) and 'links' in m:
#                             link_values.append(m['links'])
#                 elif isinstance(item, dict) and 'links' in item:
#                     link_values.append(item['links'])
#             link_list_str = ", ".join(link_values)

#             prompt_email = load_prompt("email_prompt(2).json")
#             chain_email = prompt_email | llm
#             res = chain_email.invoke({"job_description": str(job), "link_list": link_list_str})

#             st.subheader("Generated Cold Email:")
#             st.write(res.content)
#             print(res.content)

#     except Exception as e:
#         st.error(f"Error: {e}")
# # ...existing code...


# ...existing code...
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.output_parsers import JsonOutputParser
import pandas as pd
import chromadb
import uuid 
import streamlit as st
load_dotenv()

llm = ChatGroq(model="llama-3.3-70b-versatile")

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
            chains = prompt | llm
            response = chains.invoke({"job_posting_text": page_data})

            json_parser = JsonOutputParser()
            parsed_response = json_parser.parse(response.content)

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
            collection = client.get_or_create_collection(name="portfolio")
            if not collection.count():
                for _, row in df.iterrows():
                    tech = row.get('Techstack') if isinstance(row, dict) else row['Techstack'] if 'Techstack' in row else None
                    links = row.get('Links') if isinstance(row, dict) else row['Links'] if 'Links' in row else None
                    if pd.isna(tech) or not str(tech).strip():
                        continue
                    # ensure documents and metadatas are lists
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
            chain_email = prompt_email | llm
            res = chain_email.invoke({"job_description": str(parsed_response), "link_list": link_list_str})

            st.subheader("Generated Cold Email:")
            st.write(res.content)
            print(res.content)

    except Exception as e:
        st.error(f"Error: {e}")
# ...existing code...
