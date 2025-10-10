import os
import re
import json
import requests
from typing import List, Tuple


import streamlit as st
from dotenv import load_dotenv


from io import BytesIO
from tempfile import NamedTemporaryFile


# Optionally available libs for docx/pdf/html parsing
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None


try:
    import PyPDF2
except Exception:
    PyPDF2 = None


try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None


# LangChain pieces (standard names)
try:
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory
    from langchain.docstore.document import Document
except Exception as e:
    st.error("Required langchain packages not found. Please install langchain and related packages.")
    raise


# Try to import Groq provider (optional)
USE_GROQ = False
ChatGroq = None
try:
    from langchain_groq import ChatGroq  # type: ignore
    USE_GROQ = True
except Exception:
    USE_GROQ = False
    ChatGroq = None


# -------------------------
# Utility functions
# -------------------------


def load_env_to_session():
    load_dotenv()
    if "org" not in st.session_state:
        st.session_state["org"] = os.getenv("AZURE_DEVOPS_ORG", "")
    if "project" not in st.session_state:
        st.session_state["project"] = os.getenv("AZURE_DEVOPS_PROJECT", "")
    if "pat" not in st.session_state:
        st.session_state["pat"] = os.getenv("AZURE_DEVOPS_PAT", "")
    if "groq_key" not in st.session_state:
        st.session_state["groq_key"] = os.getenv("GROQ_API_KEY", "")
    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None
    if "retrieval_chain" not in st.session_state:
        st.session_state["retrieval_chain"] = None
    if "docs_sources" not in st.session_state:
        st.session_state["docs_sources"] = []
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "qa_history" not in st.session_state:
        st.session_state["qa_history"] = []
    if "flow_diagram" not in st.session_state:
        st.session_state["flow_diagram"] = None
    if "story_data" not in st.session_state:
        st.session_state["story_data"] = []


def clean_org_for_api(org: str) -> str:
    if not org:
        return ""
    org = org.strip()
    m = re.search(r"dev\.azure\.com/([^/]+)", org, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    m = re.search(r"https?://([^./]+)\.visualstudio\.com", org, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    if org.startswith("http"):
        parts = [p for p in org.split("/") if p]
        if parts:
            return parts[-1]
    return org


def strip_html_to_text(html: str) -> str:
    if not html:
        return ""
    if BeautifulSoup:
        try:
            return BeautifulSoup(html, "html.parser").get_text(separator="\n").strip()
        except Exception:
            pass
    text = re.sub(r"<[^>]+>", "", html)
    return text.strip()


def azure_fetch_work_item(org: str, project: str, work_item_id: str, pat: str) -> Tuple[str, str]:
    if not all([org, project, work_item_id, pat]):
        raise ValueError("Azure DevOps org, project, work_item_id and PAT are required.")
    org_clean = clean_org_for_api(org)
    base_url = f"https://dev.azure.com/{org_clean}"
    url = f"{base_url}/{project}/_apis/wit/workitems/{work_item_id}?api-version=7.1"
    resp = requests.get(url, auth=("", pat))
    if resp.status_code != 200:
        ct = resp.headers.get("content-type", "")
        if resp.status_code in (401, 403) or ("text/html" in ct and "Sign In" in resp.text[:200]):
            raise RuntimeError(
                f"Authentication failed when fetching work item {work_item_id}. "
                "Check your PAT (permissions: Work Items read) and that org/project are correct."
            )
        raise RuntimeError(f"Failed to fetch work item {work_item_id}: {resp.status_code} - {resp.text[:300]}")
    data = resp.json()
    fields = data.get("fields", {})
    description = fields.get("System.Description") or fields.get("Microsoft.VSTS.TCM.ReproSteps") or ""
    acceptance = fields.get("Microsoft.VSTS.Common.AcceptanceCriteria") or fields.get("Custom.AcceptanceCriteria") or ""
    return strip_html_to_text(description), strip_html_to_text(acceptance)


def azure_run_wiql(org: str, project: str, wiql_query: str, pat: str) -> List[str]:
    if not all([org, project, wiql_query, pat]):
        raise ValueError("WIQL: org, project, query and PAT are required.")
    org_clean = clean_org_for_api(org)
    base_url = f"https://dev.azure.com/{org_clean}"
    url = f"{base_url}/{project}/_apis/wit/wiql?api-version=7.1"
    headers = {"Content-Type": "application/json"}
    body = {"query": wiql_query}
    resp = requests.post(url, auth=("", pat), headers=headers, data=json.dumps(body))
    if resp.status_code != 200:
        raise RuntimeError(f"WIQL query failed: {resp.status_code} - {resp.text[:500]}")
    data = resp.json()
    items = data.get("workItems", [])
    ids = [str(i.get("id")) for i in items if i.get("id")]
    return ids


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    content = uploaded_file.read()
    if name.endswith((".txt", ".md")):
        try:
            return content.decode("utf-8", errors="ignore")
        except Exception:
            return str(content)
    if name.endswith(".docx"):
        if DocxDocument is None:
            raise RuntimeError("python-docx not installed. Install: pip install python-docx")
        tmp = NamedTemporaryFile(delete=False, suffix=".docx")
        tmp.write(content)
        tmp.flush()
        tmp.close()
        doc = DocxDocument(tmp.name)
        text = "\n".join([p.text for p in doc.paragraphs])
        try:
            os.unlink(tmp.name)
        except Exception:
            pass
        return text
    if name.endswith(".pdf"):
        if PyPDF2 is None:
            raise RuntimeError("PyPDF2 not installed. Install: pip install PyPDF2")
        tmp = BytesIO(content)
        reader = PyPDF2.PdfReader(tmp)
        pages = []
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            pages.append(t)
        return "\n".join(pages)
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception:
        return str(content)


def build_docs_from_texts(title_text_pairs: List[Tuple[str, str]]) -> List[Document]:
    docs = []
    for title, text in title_text_pairs:
        if not text or not text.strip():
            continue
        content = f"Source: {title}\n\n{text.strip()}"
        docs.append(Document(page_content=content, metadata={"source": title}))
    return docs


def extract_llm_text(llm_result) -> str:
    if isinstance(llm_result, str):
        return llm_result
    
    # Try to get content attribute directly (for ChatGroq responses)
    try:
        if hasattr(llm_result, 'content'):
            return llm_result.content
    except Exception:
        pass
    
    try:
        gens = getattr(llm_result, "generations", None)
        if gens:
            g0 = gens[0][0]
            txt = getattr(g0, "text", None) or getattr(g0, "generation", None)
            if txt:
                return txt
    except Exception:
        pass
    if isinstance(llm_result, dict):
        for k in ("answer", "text", "content", "output_text"):
            if k in llm_result and llm_result[k]:
                return llm_result[k] if isinstance(llm_result[k], str) else str(llm_result[k])
        if "choices" in llm_result and isinstance(llm_result["choices"], (list, tuple)) and len(llm_result["choices"]) > 0:
            c = llm_result["choices"][0]
            if isinstance(c, dict):
                m = c.get("message") or c.get("text")
                if isinstance(m, dict):
                    return m.get("content") or str(m)
                if isinstance(m, str):
                    return m
    return str(llm_result)


def get_llm_instance():
    groq_key = st.session_state.get("groq_key") or os.getenv("GROQ_API_KEY", "")
    if groq_key:
        try:
            try:
                llm = ChatGroq(groq_api_key=groq_key, model_name=os.getenv("model"), temperature=0.0)
            except TypeError:
                llm = ChatGroq(api_key=groq_key, model_name=os.getenv("model"), temperature=0.0)
            return llm
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatGroq: {e}")
    raise RuntimeError(
        "Groq LLM not configured or langchain_groq not installed. "
        "Set GROQ_API_KEY in .env and install langchain-groq, or modify get_llm_instance() to return a HuggingFace LLM."
    )


def strict_split(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if len(chunk) > chunk_size:
            chunk = chunk[:chunk_size]
        chunks.append(chunk)
        start += chunk_size - chunk_overlap
    return chunks


def generate_flow_diagram(title_text_pairs: List[Tuple[str, str]]) -> str:
    """
    Generate a Mermaid flow diagram using LLM to analyze story relationships
    """
    try:
        llm = get_llm_instance()
        
        # Prepare content for LLM analysis - limit content length
        stories_text = "\n\n".join([
            f"Story {idx+1}: {title}\nContent: {text[:1000]}"
            for idx, (title, text) in enumerate(title_text_pairs[:10])  # Limit to 10 stories
        ])
        
        prompt = f"""Analyze the following user stories and create a Mermaid flowchart diagram showing their relationships and workflow.

CRITICAL RULES:
1. Use ONLY these node types:
   - Rectangle: A[Text Here]
   - Rounded: A([Text Here])
   - Diamond: A{{{{Text Here}}}}  (use DOUBLE curly braces)
2. Keep node text SHORT (max 6 words)
3. Use SIMPLE arrow syntax: A --> B
4. Do NOT use special characters: ? : ; # & " in node text
5. Start with: graph LR
6. Output ONLY the Mermaid code, no explanations

STORIES:
{stories_text}

Generate Mermaid flowchart (graph LR format):"""

        result = llm.invoke(prompt)
        mermaid_code = extract_llm_text(result)
        
        # Clean up the response
        mermaid_code = mermaid_code.strip()
        mermaid_code = re.sub(r'^```mermaid\s*', '', mermaid_code)
        mermaid_code = re.sub(r'^```\s*', '', mermaid_code)
        mermaid_code = re.sub(r'```\s*$', '', mermaid_code)
        mermaid_code = mermaid_code.strip()
        
        # Ensure it starts with graph
        if not mermaid_code.startswith('graph'):
            mermaid_code = 'graph LR\n' + mermaid_code
        
        return mermaid_code
    
    except Exception as e:
        # Fallback: Create a simple diagram if LLM fails
        st.warning(f"Using fallback diagram (LLM error: {str(e)[:100]})")
        
        mermaid_code = "graph LR\n"
        mermaid_code += "    Start([Start])\n"
        
        # Limit to 8 stories
        for idx, (title, _) in enumerate(title_text_pairs[:8]):
            clean_title = title.replace('-', ' ').replace('_', ' ')[:40]
            clean_title = re.sub(r'[^\w\s]', '', clean_title)
            clean_title = ' '.join(clean_title.split())
            
            node_id = f"S{idx+1}"
            mermaid_code += f"    {node_id}[\"{clean_title}\"]\n"
            
            if idx == 0:
                mermaid_code += f"    Start --> {node_id}\n"
            else:
                mermaid_code += f"    S{idx} --> {node_id}\n"
        
        last_node = f"S{min(len(title_text_pairs), 8)}"
        mermaid_code += f"    End([Complete])\n"
        mermaid_code += f"    {last_node} --> End\n"
        
        return mermaid_code
# -------------------------
# App UI & flow
# -------------------------

st.set_page_config(page_title="Story Summarizer (Multi) + QnA", layout="wide")
load_env_to_session()

st.title("Azure DevOps Story Summarizer — Multi-story / WIQL / Multi-PDF + Q&A")

st.sidebar.header("Configuration / Credentials")
org_input = st.sidebar.text_input("Azure DevOps Organization (org)", key="org", help="Example: dpwhotfsonline or https://dev.azure.com/dpwhotfsonline")
project_input = st.sidebar.text_input("Azure DevOps Project", key="project")
pat_input = st.sidebar.text_input("Azure DevOps PAT (personal access token)", key="pat", type="password")
st.sidebar.caption("Make sure PAT has 'Work Items (Read)' permissions.")
st.sidebar.write("GROQ_API_KEY present: " + ("Yes" if st.session_state.get("groq_key") else "No"))

mode = st.radio("Input mode", ["Story IDs (comma-separated)", "WIQL Query", "Upload files (PDF/DOCX/TXT)"])

col1, col2 = st.columns([3,1])
with col1:
    if mode == "Story IDs (comma-separated)":
        story_ids = st.text_input("Enter story/work item IDs (comma-separated)", placeholder="e.g., 123, 456, 789")
    elif mode == "WIQL Query":
        wiql_query = st.text_area("Enter WIQL query", placeholder="SELECT [System.Id] FROM WorkItems WHERE ...")
    else:
        uploaded_files = st.file_uploader("Upload files (PDF, DOCX, TXT). You can upload multiple.", accept_multiple_files=True, type=["pdf","docx","txt","md"])

with col2:
    st.markdown("### Options")
    append_mode = st.checkbox("Append to existing embedded corpus (don't overwrite)", value=False)
    generate_diagram = st.checkbox("Generate Flow Diagram", value=True, help="Create visual workflow diagram")
    process_btn = st.button("Fetch / Embed & Summarize")

status = st.empty()
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

if process_btn:
    try:
        status.info("Collecting inputs...")
        title_text_pairs = []
        if mode == "Story IDs (comma-separated)":
            if not story_ids or not story_ids.strip():
                st.warning("Please enter at least one story ID.")
            else:
                ids = [s.strip() for s in re.split(r"[,\s]+", story_ids.strip()) if s.strip()]
                status.info(f"Fetching {len(ids)} work items from Azure DevOps...")
                for sid in ids:
                    try:
                        desc, ac = azure_fetch_work_item(org_input, project_input, sid, pat_input)
                        combined = "\n\n".join([("Description:\n" + desc) if desc else "", ("Acceptance Criteria:\n" + ac) if ac else ""])
                        title_text_pairs.append((f"WorkItem-{sid}", combined.strip()))
                    except Exception as e:
                        st.warning(f"Failed to fetch {sid}: {e}")
        elif mode == "WIQL Query":
            if not wiql_query or not wiql_query.strip():
                st.warning("Please provide a WIQL query.")
            else:
                status.info("Running WIQL...")
                ids = azure_run_wiql(org_input, project_input, wiql_query, pat_input)
                st.info(f"WIQL returned {len(ids)} work items.")
                for sid in ids:
                    try:
                        desc, ac = azure_fetch_work_item(org_input, project_input, sid, pat_input)
                        combined = "\n\n".join([("Description:\n" + desc) if desc else "", ("Acceptance Criteria:\n" + ac) if ac else ""])
                        title_text_pairs.append((f"WorkItem-{sid}", combined.strip()))
                    except Exception as e:
                        st.warning(f"Failed to fetch {sid}: {e}")
        else:
            if not uploaded_files:
                st.warning("Please upload at least one file.")
            else:
                for f in uploaded_files:
                    try:
                        text = extract_text_from_file(f)
                        title_text_pairs.append((f.name, text))
                    except Exception as e:
                        st.warning(f"Failed to read {f.name}: {e}")

        if not title_text_pairs:
            status.error("No content collected to embed. Fix errors above and try again.")
        else:
            st.session_state["story_data"] = title_text_pairs
            
            docs = build_docs_from_texts(title_text_pairs)
            if not docs:
                status.error("No valid text extracted from inputs.")
            else:
                status.info("Splitting & embedding documents into FAISS...")
                split_docs = []
                for d in docs:
                    chunks = strict_split(d.page_content, chunk_size=1000, chunk_overlap=200)
                    for i, c in enumerate(chunks):
                        split_docs.append(Document(page_content=c, metadata={"source": d.metadata.get("source"), "chunk": i}))
                embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
                new_vs = FAISS.from_documents(split_docs, embeddings)
                
                if append_mode and st.session_state.get("vectorstore") is not None:
                    st.session_state["vectorstore"] = new_vs
                else:
                    st.session_state["vectorstore"] = new_vs
                    
                st.session_state["docs_sources"] = [t for t, _ in title_text_pairs]
                
                try:
                    llm = get_llm_instance()
                    if st.session_state.get("retrieval_chain") is not None:
                        memory = st.session_state["retrieval_chain"].memory
                    else:
                        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                    
                    retriever = st.session_state["vectorstore"].as_retriever(search_kwargs={"k": 4})
                    conv_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
                    st.session_state["retrieval_chain"] = conv_chain
                    status.success("Embeddings created and LLM chain initialized.")
                except Exception as e:
                    st.session_state["retrieval_chain"] = None
                    status.warning(f"Embeddings created but LLM not initialized: {e}")
                
                if generate_diagram and len(title_text_pairs) > 0:
                    status.info("Generating flow diagram...")
                    try:
                        flow_diagram = generate_flow_diagram(title_text_pairs)
                        st.session_state["flow_diagram"] = flow_diagram
                    except Exception as e:
                        st.warning(f"Could not generate flow diagram: {e}")
                
                status.info("Generating summary...")
                combined_text = "\n\n".join([f"{title}\n\n{text[:5000]}" for title, text in title_text_pairs])
                summary_prompt = f"""Summarize the following user stories in markdown format.
                
CONTENT START
{combined_text}
CONTENT END"""

                try:
                    llm_temp = get_llm_instance()
                    raw_out = llm_temp.invoke(summary_prompt)
                    layman_summary = extract_llm_text(raw_out)
                    st.session_state["layman_summary"] = layman_summary
                except Exception as e:
                    st.session_state["layman_summary"] = f"(Summary not generated: {e})"
                    
                status.success("Done!")
    except Exception as e:
        status.error(f"Failed: {e}")

# Flow Diagram Section
# Flow Diagram Section
if st.session_state.get("flow_diagram"):
    st.markdown("---")
    st.header("📊 Visual Flow Diagram")
    
    if st.button("🔄 Regenerate", key="regen_btn"):
        if st.session_state.get("story_data"):
            try:
                flow_diagram = generate_flow_diagram(st.session_state["story_data"])
                st.session_state["flow_diagram"] = flow_diagram
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")
    
    diagram_code = st.session_state["flow_diagram"]
    
    # Clean diagram code
    if "content='" in diagram_code:
        match = re.search(r'(graph\s+\w+.*?)(?=\'|$)', diagram_code, re.DOTALL)
        if match:
            diagram_code = match.group(1)
    
    diagram_code = diagram_code.replace('\\n', '\n')
    diagram_code = re.sub(r'```.*?```', '', diagram_code, flags=re.DOTALL)
    diagram_code = diagram_code.strip()
    
    # Create tabs
    tab1, tab2 = st.tabs(["🎯 Visual Diagram", "📝 Code"])
    
    with tab1:
        # Render using Mermaid.js
        mermaid_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
            <script>
                mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
            </script>
            <style>
                body {{ 
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background: white;
                }}
                .mermaid {{ text-align: center; }}
            </style>
        </head>
        <body>
            <div class="mermaid">
{diagram_code}
            </div>
        </body>
        </html>
        """
        
        import streamlit.components.v1 as components
        components.html(mermaid_html, height=500, scrolling=True)
    
    with tab2:
        st.code(diagram_code, language="mermaid")
        st.download_button(
            label="💾 Download Mermaid Code",
            data=diagram_code,
            file_name="flow_diagram.mmd",
            mime="text/plain",
            key="download_btn"
        )
# Layman Summary
st.markdown("---")
st.header("📄 Summary")

if st.session_state.get("layman_summary"):
    summary = st.session_state["layman_summary"]
    if isinstance(summary, str):
        summary = summary.replace("\\n", "\n")
    st.markdown(summary, unsafe_allow_html=True)
else:
    st.info("No summary yet.")

# Q&A Section
st.markdown("---")
st.header("💬 Ask Questions")

if st.session_state.get("vectorstore") is None:
    st.info("Embed content first.")
elif st.session_state.get("retrieval_chain") is None:
    st.warning("LLM not initialized.")
else:
    with st.form(key="qa_form", clear_on_submit=True):
        user_q = st.text_input("Ask a question:", key="q_input")
        submit_q = st.form_submit_button("Ask")

    if submit_q and user_q.strip():
        try:
            chain = st.session_state["retrieval_chain"]
            result = chain.invoke({"question": user_q})
            answer = extract_llm_text(result)
            st.session_state.qa_history.insert(0, (user_q, answer))
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    if st.session_state.qa_history:
        st.markdown(f"### 💬 History ({len(st.session_state.qa_history)} questions)")
        for idx, (q, a) in enumerate(st.session_state.qa_history):
            with st.expander(f"Q{len(st.session_state.qa_history) - idx}: {q}", expanded=(idx == 0)):
                st.markdown(f"**Answer:**\n\n{a}")

# Debug
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📄 Sources", key="src_btn"):
        st.json(st.session_state.get("docs_sources", []))
with col2:
    if st.button("🎨 Diagram", key="diag_btn"):
        if st.session_state.get("story_data"):
            flow_diagram = generate_flow_diagram(st.session_state["story_data"])
            st.session_state["flow_diagram"] = flow_diagram
            st.rerun()
with col3:
    if st.button("🗑️ Reset", key="reset_btn"):
        for key in ["vectorstore", "retrieval_chain", "docs_sources", "layman_summary", "chat_history", "qa_history", "flow_diagram", "story_data"]:
            st.session_state.pop(key, None)
        st.rerun()