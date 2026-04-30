import os
import re
import json
import requests
from typing import List, Tuple
import time
import base64

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

# Excel handling
try:
    import pandas as pd
    import openpyxl
except Exception:
    pd = None
    openpyxl = None

# LangChain pieces (standard names)
# Replace the old import block (lines 34-42) with this:

# LangChain pieces (updated imports for newer versions)
try:
    # Text splitter - try new location first
    try:
        from langchain_text_splitters import CharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import CharacterTextSplitter
    
    # Embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Vector store
    from langchain_community.vectorstores import FAISS
    
    # Chains - ConversationalRetrievalChain moved to langchain.chains.retrieval
    try:
        from langchain.chains import ConversationalRetrievalChain
    except (ImportError, ModuleNotFoundError):
        try:
            from langchain.chains.retrieval import ConversationalRetrievalChain
        except (ImportError, ModuleNotFoundError):
            # For langchain >= 1.0, we need to build it manually using LCEL
            from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    
    # Memory
    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        try:
            from langchain_community.chat_message_histories import ChatMessageHistory
            from langchain.memory import ConversationBufferMemory
        except ImportError:
            from langchain_core.chat_history import BaseChatMessageHistory
            from langchain.memory import ConversationBufferMemory
    
    # Document
    try:
        from langchain.docstore.document import Document
    except ImportError:
        try:
            from langchain_core.documents import Document
        except ImportError:
            from langchain_community.docstore.document import Document
            
except Exception as e:
    st.error(f"Required langchain packages not found. Please install langchain and related packages.\nError: {e}")
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

# Groq client for test case generation
try:
    from groq import Groq
    GROQ_CLIENT_AVAILABLE = True
except Exception:
    GROQ_CLIENT_AVAILABLE = False
    Groq = None

MODEL_CONFIG = {
    "llama-3.3-70b-versatile": {"max_tokens": 8000, "max_input_chars": 6000},
    "openai/gpt-oss-20b":      {"max_tokens": 4000, "max_input_chars": 4000},
    "groq/compound":            {"max_tokens": 4000, "max_input_chars": 4000},
    "qwen/qwen3-32b":           {"max_tokens": 4000, "max_input_chars": 3000},
}
DEFAULT_MODEL_CONFIG = {"max_tokens": 4000, "max_input_chars": 4000}

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
    if "test_cases_excel" not in st.session_state:
        st.session_state["test_cases_excel"] = None

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
            soup = BeautifulSoup(html, "html.parser")
            for s in soup(["script", "style"]):
                s.decompose()
            text = soup.get_text(separator="\n")
            text = re.sub(r"\n\s*\n+", "\n\n", text).strip()
            return text
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

def azure_fetch_work_item_comments(org: str, project: str, work_item_id: str, pat: str) -> List[dict]:
    """
    Fetch ALL comments for a work item with pagination and priority ranking.
    Returns list of dicts: [{'author': str, 'text': str, 'created': str, 'weight': int}]
    """
    try:
        org_clean = clean_org_for_api(org)
        base_url = f"https://dev.azure.com/{org_clean}"
        url = f"{base_url}/{project}/_apis/wit/workItems/{work_item_id}/comments?api-version=7.1-preview.3"

        all_comments = []
        continuation_token = None

        while True:
            headers = {}
            if continuation_token:
                headers["x-ms-continuationtoken"] = continuation_token

            resp = requests.get(url, auth=("", pat), headers=headers)
            if resp.status_code == 404:
                break  # No comments
            resp.raise_for_status()

            data = resp.json()
            comments = data.get("comments", [])
            for c in comments:
                author = c.get("createdBy", {}).get("displayName", "Unknown")
                text = strip_html_to_text(c.get("text", "")).strip()
                created = c.get("createdDate", "")
                if text:
                    weight = 1
                    name_lower = author.lower()
                    if any(k in name_lower for k in ["dev", "developer", "engineer"]):
                        weight = 3
                    elif any(k in name_lower for k in ["qa", "test"]):
                        weight = 2
                    elif any(k in name_lower for k in ["po", "manager", "lead"]):
                        weight = 1

                    all_comments.append({
                        "author": author,
                        "text": text,
                        "created": created,
                        "weight": weight
                    })

            continuation_token = resp.headers.get("x-ms-continuationtoken")
            if not continuation_token:
                break  # no more pages

        # Sort by weight (desc) then recency
        all_comments.sort(key=lambda x: (x["weight"], x["created"]), reverse=True)
        return all_comments

    except Exception as e:
        st.warning(f"Comment fetch failed for {work_item_id}: {e}")
        return []

import urllib.parse

def azure_fetch_linked_prs(org: str, project: str, work_item_id: str, pat: str) -> List[dict]:
    """
    Fetch Pull Requests linked to a given Azure DevOps work item.
    Supports 3-part vstfs PullRequestId format:
    vstfs:///Git/PullRequestId/<projectId>%2F<repoId>%2F<prId>
    Returns: list of dicts [{ 'projectId': str, 'repoId': str, 'prId': str }]
    """
    try:
        org_clean = clean_org_for_api(org)
        base_url = f"https://dev.azure.com/{org_clean}"
        url = f"{base_url}/{project}/_apis/wit/workItems/{work_item_id}?$expand=relations&api-version=7.1"

        resp = requests.get(url, auth=("", pat))
        resp.raise_for_status()
        data = resp.json()

        relations = data.get("relations", [])
        prs = []

        for rel in relations:
            url_val = rel.get("url", "")
            if "PullRequestId" in url_val:
                encoded = url_val.split("PullRequestId/")[-1]
                decoded = urllib.parse.unquote(encoded)  # decode %2F → /
                parts = decoded.split("/")
                if len(parts) == 3:  # ProjectID / RepoID / PRID
                    project_id, repo_id, pr_id = parts
                    prs.append({
                        "projectId": project_id,
                        "repoId": repo_id,
                        "prId": pr_id
                    })
        return prs
    except Exception as e:
        st.warning(f"PR fetch failed for {work_item_id}: {e}")
        return []

def azure_fetch_pr_file_diffs(org: str, project_id: str, repo_id: str, pr_id: str, pat: str) -> List[dict]:
    """
    Fetch actual file diffs from Azure DevOps PR using the correct API endpoints.
    Returns list of dicts with actual diff content.
    
    APPROACHES:
    1. Try iterations API with diff format
    2. Fall back to commit comparison
    3. Final fallback: file content only
    """
    try:
        org_clean = clean_org_for_api(org)
        base_url = f"https://dev.azure.com/{org_clean}"
        
        st.info("🔍 Attempting to fetch PR diffs...")
        
        # APPROACH 1: Get PR details first
        pr_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/pullRequests/{pr_id}?api-version=7.1"
        pr_resp = requests.get(pr_url, auth=("", pat))
        
        if pr_resp.status_code != 200:
            st.error(f"❌ Cannot fetch PR details: {pr_resp.status_code}")
            return []
        
        pr_data = pr_resp.json()
        source_commit = pr_data.get("lastMergeSourceCommit", {}).get("commitId")
        target_commit = pr_data.get("lastMergeTargetCommit", {}).get("commitId")
        
        st.info(f"✅ PR commits - Source: {source_commit[:8] if source_commit else 'N/A'}, Target: {target_commit[:8] if target_commit else 'N/A'}")
        
        result = []
        
        # APPROACH 2: Get changed files using iterations API
        iter_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/pullRequests/{pr_id}/iterations?api-version=7.1"
        iter_resp = requests.get(iter_url, auth=("", pat))
        
        if iter_resp.status_code != 200:
            st.warning(f"⚠️ Iterations API failed: {iter_resp.status_code}")
            return []
        
        iter_data = iter_resp.json()
        iterations = iter_data.get("value", [])
        
        if not iterations:
            st.warning("⚠️ No iterations found")
            return []
        
        latest_iter_id = iterations[-1].get("id")
        st.info(f"📋 Using iteration {latest_iter_id}")
        
        # APPROACH 3: Get changes with proper expansion
        changes_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/pullRequests/{pr_id}/iterations/{latest_iter_id}/changes?api-version=7.1"
        changes_resp = requests.get(changes_url, auth=("", pat))
        
        if changes_resp.status_code != 200:
            st.error(f"❌ Changes API failed: {changes_resp.status_code}")
            st.error(f"Response: {changes_resp.text[:500]}")
            return []
        
        changes_data = changes_resp.json()
        change_entries = changes_data.get("changeEntries", [])
        
        st.info(f"📊 Found {len(change_entries)} changed files")
        
        # Process each changed file
        for idx, entry in enumerate(change_entries[:15], 1):  # Limit to 15 files
            item = entry.get("item", {})
            change_type = entry.get("changeType", "edit")
            
            path = item.get("path", "")
            if not path:
                continue
            
            # Skip directories and non-code files
            if item.get("gitObjectType") == "tree":
                continue
            
            file_name = path.split("/")[-1]
            extension = path.split(".")[-1] if "." in path else "unknown"
            
            st.info(f"  📄 {idx}. Processing {file_name}...")
            
            # APPROACH 4: Try to get actual diff using commits
            diff_content = "[Diff not available]"
            
            if source_commit and target_commit and change_type.lower() in ["edit", "add"]:
                try:
                    # Method A: Try diff API
                    diff_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/diffs/commits?baseVersion={target_commit}&targetVersion={source_commit}&api-version=7.1"
                    diff_resp = requests.get(diff_url, auth=("", pat), timeout=10)
                    
                    if diff_resp.status_code == 200:
                        diff_data = diff_resp.json()
                        # Find our file in the diff
                        for change in diff_data.get("changes", []):
                            if change.get("item", {}).get("path") == path:
                                diff_content = f"Change detected in {path}\n"
                                diff_content += f"Lines added/modified in this file\n"
                                st.success(f"    ✓ Got diff for {file_name}")
                                break
                    
                    # Method B: Get file content from both commits and compare
                    if diff_content == "[Diff not available]":
                        source_blob = item.get("objectId")
                        if source_blob:
                            blob_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/blobs/{source_blob}?api-version=7.1"
                            blob_resp = requests.get(blob_url, auth=("", pat), timeout=5)
                            
                            if blob_resp.status_code == 200:
                                content = blob_resp.text[:2000]  # First 2000 chars
                                diff_content = f"=== Modified: {file_name} ===\n\n{content}\n\n[Content truncated...]"
                                st.success(f"    ✓ Got file content for {file_name}")
                
                except Exception as e:
                    st.warning(f"    ⚠️ Could not fetch diff for {file_name}: {str(e)[:100]}")
                    diff_content = f"[Error: {str(e)[:100]}]"
            
            result.append({
                'path': path,
                'fileName': file_name,
                'extension': extension,
                'changeType': change_type,
                'diff': diff_content,
                'summary': f"File: {file_name}\nPath: {path}\nChange: {change_type}"
            })
        
        st.success(f"✅ Successfully processed {len(result)} files")
        return result
        
    except Exception as e:
        st.error(f"❌ Exception in PR diff fetch: {e}")
        import traceback
        st.error(traceback.format_exc())
        return []


def azure_fetch_pr_changes(org: str, project_id: str, repo_id: str, pr_id: str, pat: str) -> List[dict]:
    """
    Wrapper function - tries multiple strategies to get PR changes
    """
    # Try the main diff fetching method
    results = azure_fetch_pr_file_diffs(org, project_id, repo_id, pr_id, pat)
    
    if results:
        return results
    
    # FALLBACK: Metadata-only approach (compliance-safe)
    st.warning("⚠️ Falling back to metadata-only analysis (compliance-safe mode)")
    return azure_fetch_pr_metadata_only(org, project_id, repo_id, pr_id, pat)


def azure_fetch_pr_metadata_only(org: str, project_id: str, repo_id: str, pr_id: str, pat: str) -> List[dict]:
    """
    FALLBACK: Get file metadata only (no diffs) - always works, compliance-safe
    """
    try:
        org_clean = clean_org_for_api(org)
        base_url = f"https://dev.azure.com/{org_clean}"
        
        # Get PR commits
        commits_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/pullRequests/{pr_id}/commits?api-version=7.1"
        commits_resp = requests.get(commits_url, auth=("", pat))
        
        if commits_resp.status_code != 200:
            return []
        
        commits_data = commits_resp.json()
        commits = commits_data.get("value", [])
        
        result = []
        file_extensions = set()
        
        # Get file list from latest commit
        if commits:
            latest_commit = commits[-1].get("commitId")
            changes_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/commits/{latest_commit}/changes?api-version=7.1"
            changes_resp = requests.get(changes_url, auth=("", pat))
            
            if changes_resp.status_code == 200:
                changes_data = changes_resp.json()
                changes = changes_data.get("changes", [])
                
                for change in changes[:20]:
                    item = change.get("item", {})
                    path = item.get("path", "")
                    
                    if not path or item.get("gitObjectType") == "tree":
                        continue
                    
                    file_name = path.split("/")[-1]
                    extension = path.split(".")[-1] if "." in path else "unknown"
                    change_type = change.get("changeType", "edit")
                    
                    file_extensions.add(extension)
                    
                    result.append({
                        'path': path,
                        'fileName': file_name,
                        'extension': extension,
                        'changeType': change_type,
                        'diff': f"[Metadata only - {change_type} operation on {extension} file]",
                        'summary': f"File: {file_name}\nPath: {path}\nType: {extension}\nChange: {change_type}"
                    })
        
        st.info(f"📊 Metadata-only mode: {len(result)} files, types: {', '.join(file_extensions)}")
        return result
        
    except Exception as e:
        st.error(f"Even metadata fetch failed: {e}")
        return []


def generate_pr_test_focus(org: str, project: str, story_id: str, pat: str, story_text: str) -> str:
    """
    Analyze linked PRs - tries to get diffs, falls back to metadata if needed
    """
    try:
        st.info(f"🔍 Step 1: Fetching linked PRs for story {story_id}...")
        prs = azure_fetch_linked_prs(org, project, story_id, pat)
        
        if not prs:
            return f"*(No PRs linked to story {story_id})*"
        
        st.info(f"✅ Found {len(prs)} linked PR(s)")
        
        all_changes = []
        has_actual_diffs = False
        
        for idx, pr in enumerate(prs, 1):
            st.info(f"📁 Analyzing PR {idx}/{len(prs)} (ID: {pr['prId']})...")
            pr_changes = azure_fetch_pr_changes(org, pr["projectId"], pr["repoId"], pr["prId"], pat)
            
            if pr_changes:
                # Check if we got actual diffs or just metadata
                sample_diff = pr_changes[0].get('diff', '')
                if '[Metadata only' not in sample_diff and '[Diff not available]' not in sample_diff:
                    has_actual_diffs = True
                
                st.success(f"  ✓ Retrieved {len(pr_changes)} file changes")
                all_changes.extend(pr_changes)
            else:
                st.warning(f"  ⚠ No changes found for PR {pr['prId']}")
        
        if not all_changes:
            return "*(No changed files detected for linked PRs)*"
        
        st.info(f"📊 Total files to review: {len(all_changes)}")
        
        # Build context based on what we have
        if has_actual_diffs:
            st.success("✅ Got actual code diffs - performing detailed code review")
            return generate_detailed_code_review(all_changes, story_text)
        else:
            st.warning("⚠️ Using metadata-only analysis (compliance-safe)")
            return generate_metadata_based_analysis(all_changes, story_text)
        
    except Exception as e:
        import traceback
        st.error(f"Error: {e}")
        st.error(traceback.format_exc())
        return f"❌ Error during PR analysis: {e}"


def generate_detailed_code_review(changes: List[dict], story_text: str) -> str:
    """Generate detailed code review when we have actual diffs"""
    context = []
    for idx, change in enumerate(changes[:10], 1):
        context.append(f"""
### File {idx}: {change.get('fileName')}
**Path:** {change.get('path')}
**Change:** {change.get('changeType')}
**Code:**
```
{change.get('diff', '')[:2000]}
```
""")
    
    prompt = f"""
You are a SENIOR CODE REVIEWER analyzing actual code changes.

**STORY CONTEXT:**
{story_text[:1000]}

**CODE CHANGES:**
{"".join(context)}

**Perform line-by-line analysis and identify:**

### 🐛 CONFIRMED BUGS
- List actual bugs you see in the code with line-level details

### ⚠️ POTENTIAL ISSUES
- Code smells, logic flaws, missing validations

### 🎯 CRITICAL TEST SCENARIOS
- Must-test cases based on code changes

Be specific with file names and code patterns.
"""
    
    llm = get_llm_instance()
    result = extract_llm_text(llm.invoke(prompt))
    return result + "\n\n*✅ Based on actual code review*"


def generate_metadata_based_analysis(changes: List[dict], story_text: str) -> str:
    """Generate risk analysis when we only have metadata"""
    
    # Group by file type
    by_type = {}
    for change in changes:
        ext = change.get('extension', 'unknown')
        if ext not in by_type:
            by_type[ext] = []
        by_type[ext].append(change)
    
    context = []
    for ext, files in by_type.items():
        context.append(f"\n**{ext.upper()} Files ({len(files)}):**")
        for f in files[:5]:
            context.append(f"- {f.get('fileName')} ({f.get('changeType')})")
    
    stats = f"""
**Files Changed:** {len(changes)}
**File Types:** {', '.join(by_type.keys())}
{"".join(context)}
"""
    
    prompt = f"""
You are a QA ARCHITECT analyzing PR metadata (file paths, types, changes).

**STORY CONTEXT:**
{story_text[:1000]}

{stats}

**Based on file types and locations, identify:**

### 🔴 HIGH RISK AREAS
- Backend files → business logic bugs
- Database files → data integrity issues
- Config files → environment bugs

### 🎯 TESTING PRIORITIES
- Must-test scenarios based on changed areas

### ⚠️ COMMON PITFALLS
- Typical bugs for these file types

Be specific about file names and risk reasoning.
"""
    
    llm = get_llm_instance()
    result = extract_llm_text(llm.invoke(prompt))
    return result + "\n\n*🔒 Compliance-safe: metadata-only analysis (no source code sent to AI)*"
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
                llm = ChatGroq(groq_api_key=groq_key, model_name=os.getenv("model", "llama-3.3-70b-versatile"), temperature=0.0)
            except TypeError:
                llm = ChatGroq(api_key=groq_key, model_name=os.getenv("model", "llama-3.3-70b-versatile"), temperature=0.0)
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
    Generate intelligent flow diagram for ANY type of story
    """
    try:
        llm = get_llm_instance()
        
        # Prepare comprehensive story content
        all_text = ""
        for idx, (title, text) in enumerate(title_text_pairs[:8], 1):
            clean_title = title.replace('WorkItem-', '').replace('_', ' ')
            all_text += f"\n\n=== Story {idx}: {clean_title} ===\n{text[:1200]}"
        
        # Universal intelligent prompt
        prompt = f"""You are an expert business analyst creating a technical flow diagram. Read 
            the user stories below and create a Mermaid flowchart that accurately represents 
            the ACTUAL LOGIC AND WORKFLOW described.

            **USER STORIES:**
            {all_text}

            **YOUR TASK:**
            Carefully analyze the content above and extract:
            1. What are the ACTUAL steps/processes mentioned?
            2. What are the DECISION points (if/else, conditions, checks)?
            3. What are the DIFFERENT SCENARIOS or paths?
            4. What gets CREATED, SAVED, or PROCESSED differently in each case?
            5. Are there CONFIG checks, FLAG settings, or VALIDATION rules?

            **CRITICAL INSTRUCTIONS:**

            1. **READ THE CONTENT CAREFULLY** - Don't assume generic flows
            - Extract EXACT scenarios mentioned (don't invent)
            - Use ACTUAL terminology from the stories
            - Show REAL decision points, not generic ones

            2. **Identify the Flow Type:**
            - If stories mention configs/flags/settings → Show config checks and conditional behavior
            - If stories mention user actions/screens → Show user workflow
            - If stories mention API/data/processing → Show data flow
            - If stories mention different modes/types → Show branching for each

            3. **Mermaid Syntax (STRICTLY FOLLOW):**
            - Start with: graph TD
            - Decisions: NodeName{{Question?}} (single braces!)
            - Steps: NodeName[Action Description]
            - Start/End: Start([Begin]) and End([Complete])
            - Conditional arrows: Node -->|Label| NextNode
            - Keep labels under 50 characters
            - NO special characters in node text: avoid : ; # & "

            4. **Extract from Stories:**
            - Look for "As Is" vs "To Be" → Show both flows or just "To Be"
            - Look for "if/when/based on" → These are decision points
            - Look for "depending on" → These are branches
            - Look for "should/will/must" → These are required steps
            - Look for specific scenarios (Mode A, Mode B, Type X, Type Y)

            5. **Quality Checks:**
            - Does your diagram reflect what's ACTUALLY in the stories?
            - Have you included ALL mentioned scenarios?
            - Are decision points based on REAL conditions from the story?
            - Do different paths show DIFFERENT outcomes mentioned?

            **EXAMPLE 1 - Config-Based Logic:**
            ```
            graph TD
                Start([Start])
                Start --> Check{{Mode Type?}}
                Check -->|Type A| ConfigA[Check Config A]
                ConfigA --> DefaultA[Set Default Values]
                DefaultA --> SaveA[Save Type A]
                Check -->|Type B| ConfigB[Check Config B]
                ConfigB --> SpecialB[Apply Special Logic]
                SpecialB --> SaveB[Save Type B Differently]
                SaveA --> End([Complete])
                SaveB --> End
            ```

            **EXAMPLE 2 - User Workflow:**
            ```
            graph TD
                Start([User Opens Screen])
                Start --> Input[Enter Data]
                Input --> Validate{{Valid?}}
                Validate -->|Yes| Process[Process Request]
                Validate -->|No| Error[Show Error]
                Error --> Input
                Process --> Confirm{{Confirmed?}}
                Confirm -->|Yes| Save[Save Data]
                Confirm -->|No| Cancel[Cancel]
                Save --> End([Complete])
                Cancel --> End
            ```

            **EXAMPLE 3 - Multi-Scenario:**
            ```
            graph TD
                Start([Begin])
                Start --> SelectType{{Select Type?}}
                SelectType -->|Scenario 1| Path1[Process Scenario 1]
                Path1 --> Action1[Specific Action 1]
                SelectType -->|Scenario 2| Path2[Process Scenario 2]
                Path2 --> Check2{{Condition?}}
                Check2 -->|Yes| Action2A[Action 2A]
                Check2 -->|No| Action2B[Action 2B]
                SelectType -->|Scenario 3| Path3[Process Scenario 3]
                Path3 --> Action3[Specific Action 3]
                Action1 --> End([Complete])
                Action2A --> End
                Action2B --> End
                Action3 --> End
            ```

            **OUTPUT REQUIREMENTS:**
            - Create a diagram that MATCHES the actual story content
            - Use REAL terminology from the stories
            - Show ALL scenarios/modes/types mentioned
            - Include ACTUAL decision points from the stories
            - Output ONLY the Mermaid code, no explanations
            - Start with graph TD
"""

        result = llm.invoke(prompt)
        mermaid_code = extract_llm_text(result)
        
        # Clean up response
        mermaid_code = mermaid_code.strip()
        mermaid_code = re.sub(r'^```mermaid\s*', '', mermaid_code)
        mermaid_code = re.sub(r'^```\s*', '', mermaid_code)
        mermaid_code = re.sub(r'```\s*$', '', mermaid_code)
        mermaid_code = mermaid_code.strip()
        
        # Ensure it starts with graph
        if not mermaid_code.startswith('graph'):
            mermaid_code = 'graph TD\n' + mermaid_code
        
        # Minimal safe cleanup
        mermaid_code = safe_clean_mermaid(mermaid_code)
        
        # Validate the diagram has meaningful content
        line_count = mermaid_code.count('\n')
        node_count = mermaid_code.count('[') + mermaid_code.count('{')
        
        if line_count < 5 or node_count < 3:
            st.warning("Generated diagram seems too simple, using content-based fallback")
            return generate_content_based_fallback(title_text_pairs)
        
        return mermaid_code
    except Exception as e:
        st.warning(f"LLM generation failed: {str(e)[:100]}")
        return generate_content_based_fallback(title_text_pairs)

def safe_clean_mermaid(code: str) -> str:
    """
    Minimal safe cleanup - only fix obvious issues
    """
    # Remove dangerous characters from labels only (inside brackets)
    def clean_bracket_content(match):
        content = match.group(1)
        # Remove only truly problematic chars
        content = content.replace('"', '').replace("'", '')
        content = content.replace(':', '').replace(';', '')
        content = content[:50]  # Limit length
        return f'[{content}]'
    
    # Clean rectangular brackets
    code = re.sub(r'\[([^\]]+)\]', clean_bracket_content, code)
    
    # Clean rounded brackets  
    def clean_paren_content(match):
        content = match.group(1)
        content = content.replace('"', '').replace("'", '')
        content = content[:30]
        return f'([{content}])'
    
    code = re.sub(r'\(\[([^\]]+)\]\)', clean_paren_content, code)
    
    return code


def generate_content_based_fallback(title_text_pairs: List[Tuple[str, str]]) -> str:
    """
    Intelligent fallback that analyzes actual content patterns
    """
    all_text = " ".join([text.lower() for _, text in title_text_pairs])
    
    # Extract decision points from content
    decision_indicators = re.findall(r'(?:if|when|based on|depending on)\s+([^,.\n]{10,60})', all_text)
    
    # Extract scenarios/modes/types
    scenario_patterns = [
        r'(air|sea|road|rail|fcl|lcl|ftl|ltl)',
        r'(type\s+\w+|mode\s+\w+|scenario\s+\w+)',
        r'(option\s+\w+|category\s+\w+)'
    ]
    
    scenarios = set()
    for pattern in scenario_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        scenarios.update([m.strip().title() for m in matches if m.strip()])
    
    # Extract actions (verbs)
    action_patterns = r'(?:should|will|must|to)\s+(create|save|update|delete|check|validate|process|calculate|generate|display|send|receive)\s+([^,.\n]{5,40})'
    actions = re.findall(action_patterns, all_text)
    
    # Build diagram based on extracted patterns
    code = "graph TD\n"
    code += "    Start([Begin Process])\n"
    
    if scenarios:
        # Has multiple scenarios - create branching
        code += "    Start --> Select{Choose Option}\n"
        
        for idx, scenario in enumerate(list(scenarios)[:5], 1):  # Max 5 scenarios
            scenario_clean = scenario.replace(' ', '')[:20]
            code += f"    Select -->|{scenario}| S{idx}Process\n"
            code += f"    S{idx}Process[Process {scenario}]\n"
            
            # Add actions if found
            if actions and idx <= len(actions):
                verb, obj = actions[idx-1]
                code += f"    S{idx}Process --> S{idx}Action[{verb.title()} {obj[:30]}]\n"
                code += f"    S{idx}Action --> End\n"
            else:
                code += f"    S{idx}Process --> End\n"
    
    elif decision_indicators:
        # Has decisions - create conditional flow
        prev = "Start"
        for idx, decision in enumerate(decision_indicators[:3], 1):
            decision_clean = decision.strip()[:40]
            code += f"    {prev} --> D{idx}{{{decision_clean}?}}\n"
            code += f"    D{idx} -->|Yes| A{idx}Yes[Process Yes Path]\n"
            code += f"    D{idx} -->|No| A{idx}No[Process No Path]\n"
            code += f"    A{idx}Yes --> End\n"
            code += f"    A{idx}No --> End\n"
            prev = f"D{idx}"
            break  # Just show first decision
    
    elif actions:
        # Has actions - create linear flow
        prev = "Start"
        for idx, (verb, obj) in enumerate(actions[:5], 1):
            code += f"    {prev} --> A{idx}[{verb.title()} {obj[:30]}]\n"
            prev = f"A{idx}"
        code += f"    {prev} --> End\n"
    
    else:
        # Generic fallback
        code += "    Start --> Process[Process Request]\n"
        code += "    Process --> Validate{{Valid?}}\n"
        code += "    Validate -->|Yes| Complete[Complete]\n"
        code += "    Validate -->|No| Error[Handle Error]\n"
        code += "    Error --> Process\n"
        code += "    Complete --> End\n"
    
    code += "    End([Complete])\n"
    
    return code

# -------------------------
# TEST CASE GENERATION FUNCTIONS
# -------------------------

def sanitize_groq_output(text: str) -> str:
    """
    Fix common invalid JSON issues from AI output:
    - Escape unescaped backslashes
    - Replace invalid single quotes in keys with double quotes
    - Remove control chars if needed
    """
    # Escape unescaped backslashes (excluding already escaped ones)
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    
    # Replace single quoted property names with double quotes
    text = re.sub(r"(?<=\{|\s)'([^']+)':", r'"\1":', text)

    # Remove control characters that can break JSON parsing
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)

    return text

def make_groq_test_prompt(title: str, description: str, 
                          acceptance_criteria: str, model: str = "") -> str:
    cfg = MODEL_CONFIG.get(model, DEFAULT_MODEL_CONFIG)
    max_chars = cfg["max_input_chars"]

    # Truncate inputs to stay within token budget
    desc_truncated = description[:max_chars]
    ac_truncated = acceptance_criteria[:max_chars // 2]

    prompt = f"""
You are a senior test-design engineer. For the user story below, produce a structured list of test cases.
Your output MUST be valid JSON (an array of objects). Each object should have:
  - test_case_id
  - priority (High/Medium/Low)
  - test_case (short title)
  - precondition
  - test_steps (array of strings)
  - expected_result

User Story Title: {title}

Description:
{desc_truncated}

Acceptance Criteria:
{ac_truncated}

RULES:
1. Generate 20-40 test cases covering positive, negative, edge, boundary, and integration scenarios.
2. Assign priority based on risk and business value.
3. Each expected_result must be 1-2 sentences.
4. Return ONLY the JSON array, no extra commentary.
"""
    return prompt.strip()

def call_groq_for_tests(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: int = 120):
    if not GROQ_CLIENT_AVAILABLE:
        raise RuntimeError("Groq client not available. Install: pip install groq")
    
    cfg = MODEL_CONFIG.get(model, DEFAULT_MODEL_CONFIG)
    safe_max_tokens = cfg["max_tokens"]
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=safe_max_tokens,
        timeout=timeout,
    )
    text = response.choices[0].message.content.strip()

    # Sanitize the output to fix JSON issues before loading
    sanitized_text = sanitize_groq_output(text)

    try:
        return json.loads(sanitized_text)
    except Exception:
        first, last = sanitized_text.find("["), sanitized_text.rfind("]")
        if first != -1 and last != -1:
            try:
                return json.loads(sanitized_text[first:last+1])
            except Exception as e:
                raise ValueError(f"Groq output not valid JSON after sanitizing:\n{sanitized_text[:1000]}\nError: {str(e)}")
        raise ValueError(f"Groq output not valid JSON:\n{sanitized_text[:1000]}")

def story_to_testcases(title: str, description_html: str, acceptance_html: str, groq_api_key: str, model: str):
    desc_text = strip_html_to_text(description_html)
    ac_text = strip_html_to_text(acceptance_html)
    prompt = make_groq_test_prompt(title, desc_text, ac_text, model)  # ← pass model
    ai_results = call_groq_for_tests(prompt, api_key=groq_api_key, model=model)

    rows = []
    for idx, item in enumerate(ai_results, start=1):
        tc_id = item.get("test_case_id") or f"TC_{idx}"
        priority = item.get("priority") or "Medium"
        tc_title = item.get("test_case") or ""
        pre = item.get("precondition") or ""
        steps = item.get("test_steps") or []
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.splitlines() if s.strip()]
        expected = item.get("expected_result") or ""

        rows.append({
            "Test Case ID": tc_id,
            "Priority": priority,
            "Test Case": tc_title,
            "Pre-condition / Test Data": pre,
            "Test Steps": "<br>".join([f"{i+1}. {s}" for i, s in enumerate(steps)]),
            "Expected Result": expected
        })

    return pd.DataFrame(rows, columns=[
        "Test Case ID", "Priority", "Test Case",
        "Pre-condition / Test Data", "Test Steps", "Expected Result"
    ])

def replace_br_with_newline_in_memory(wb):
    """
    Replace <br> tags with newlines in all cells of the workbook
    """
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and "<br>" in cell.value:
                    cell.value = cell.value.replace("<br>", "\n")
    
    return wb

def generate_test_cases_excel(story_ids: List[str], org: str, project: str, pat: str, groq_key: str, model: str) -> BytesIO:
    """
    Generate test cases for multiple stories and return Excel file as BytesIO
    """
    if pd is None or openpyxl is None:
        raise RuntimeError("pandas and openpyxl are required. Install: pip install pandas openpyxl")
    
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sid in story_ids:
            try:
                desc, ac = azure_fetch_work_item(org, project, sid, pat)
                
                # Fetch title for better sheet naming
                org_clean = clean_org_for_api(org)
                base_url = f"https://dev.azure.com/{org_clean}"
                url = f"{base_url}/{project}/_apis/wit/workitems/{sid}?api-version=7.1"
                resp = requests.get(url, auth=("", pat))
                resp.raise_for_status()
                data = resp.json()
                title = data.get("fields", {}).get("System.Title", f"WorkItem_{sid}")
                
                st.info(f"Generating test cases for {sid} - {title}")
                
                df = story_to_testcases(title, desc, ac, groq_key, model)
                
                safe_sheet = f"{sid}"[:31]  # Excel sheet name limit
                df.to_excel(writer, sheet_name=safe_sheet, index=False)
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                st.warning(f"Failed to generate test cases for {sid}: {e}")
                # Create fallback sheet
                df = pd.DataFrame([{
                    "Test Case ID": "NA",
                    "Priority": "NA",
                    "Test Case": f"Generation failed for {sid}",
                    "Pre-condition / Test Data": "",
                    "Test Steps": str(e)[:500],
                    "Expected Result": ""
                }])
                safe_sheet = f"{sid}"[:31]
                df.to_excel(writer, sheet_name=safe_sheet, index=False)
    
    # Load the workbook and replace <br> with newlines
    output.seek(0)
    wb = openpyxl.load_workbook(output)
    wb = replace_br_with_newline_in_memory(wb)
    
    # Save back to BytesIO
    final_output = BytesIO()
    wb.save(final_output)
    final_output.seek(0)
    
    return final_output

# -------------------------
# App UI & flow
# -------------------------

st.set_page_config(page_title="Story Summarizer (Multi) + QnA + Test Cases", layout="wide")
load_env_to_session()

st.title("Azure DevOps Story Summarizer — Multi-story / WIQL / Multi-PDF + Q&A + Test Cases")

st.sidebar.header("Configuration / Credentials")
org_input = st.sidebar.text_input("Azure DevOps Organization (org)", key="org", help="Example: dpwhotfsonline or https://dev.azure.com/dpwhotfsonline")
project_input = st.sidebar.text_input("Azure DevOps Project", key="project")
pat_input = st.sidebar.text_input("Azure DevOps PAT (personal access token)", key="pat", type="password")
st.sidebar.caption("Make sure PAT has 'Work Items (Read)' permissions.")
st.sidebar.write("GROQ_API_KEY present: " + ("Yes" if st.session_state.get("groq_key") else "No"))

# Create main tabs
main_tab1, main_tab2 = st.tabs(["📋 Story Summarizer & Q&A", "🧪 Test Case Generator"])

# -------------------------
# TAB 1: Story Summarizer & Q&A (Original functionality)
# -------------------------
with main_tab1:
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
                            comments = azure_fetch_work_item_comments(org_input, project_input, sid, pat_input)
                            if comments:
                                formatted_comments = "\n".join(
                                    [f"- [{c['author']}] ({c['created']}) ⭐{c['weight']}: {c['text']}" for c in comments]
                                )
                            else:
                                formatted_comments = "No comments found."
                            comments_text = formatted_comments

                            combined = "\n\n".join([
                                ("Description:\n" + desc) if desc else "",
                                ("Acceptance Criteria:\n" + ac) if ac else "",
                                ("Comments:\n" + comments_text) if comments_text else ""
                            ])
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
                    summary_prompt = f"""
You are an expert Agile analyst and QA architect.
Summarize the following Azure DevOps user stories clearly in Markdown format.

Make sure to:
1. Extract main objectives from Description and Acceptance Criteria.
2. Analyze developer and QA comments for decisions, deferrals, or concerns.
3. Highlight any risk areas, incomplete work, or postponed validations.
4. Give a short "Discussion Summary" section for each story.

Each story summary should follow this format:

### 🧩 {{Story Title}}

**Overview**
- One or two lines explaining what this story is about.

**Layman Summary**
- Summarize the whole story in layman terms + bullet points

**P0 Scenarios**
- Tell the most critical and P0 scenarios which are mandatory

**Acceptance & Completion Criteria**
- Summarize the acceptance or done conditions.

**Comment Insights**
- Mention relevant comments or discussion takeaways.
- Prioritize developer or QA comments (⭐3 or ⭐2).

---

CONTENT START
{combined_text}
CONTENT END
"""

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
    # Find this section in your code (around line 1050-1150) and REPLACE it:

# Flow Diagram Section
    if st.session_state.get("flow_diagram"):
        st.markdown("---")
        
        # Header with expand button
        col_header1, col_header2, col_header3 = st.columns([3, 1, 1])
        with col_header1:
            st.header("📊 Visual Flow Diagram")
        with col_header2:
            if st.button("🔄 Regenerate", key="regen_btn"):
                if st.session_state.get("story_data"):
                    try:
                        flow_diagram = generate_flow_diagram(st.session_state["story_data"])
                        st.session_state["flow_diagram"] = flow_diagram
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed: {e}")
        with col_header3:
            # Toggle fullscreen mode
            if "diagram_fullscreen" not in st.session_state:
                st.session_state["diagram_fullscreen"] = False
            
            if st.session_state["diagram_fullscreen"]:
                if st.button("🗙 Exit Fullscreen", key="exit_fullscreen_btn", type="primary"):
                    st.session_state["diagram_fullscreen"] = False
                    st.rerun()
            else:
                if st.button("⛶ Fullscreen", key="fullscreen_btn"):
                    st.session_state["diagram_fullscreen"] = True
                    st.rerun()
        
        diagram_code = st.session_state["flow_diagram"]
        
        # Clean diagram code
        if "content='" in diagram_code:
            match = re.search(r'(graph\s+\w+.*?)(?=\'|$)', diagram_code, re.DOTALL)
            if match:
                diagram_code = match.group(1)
        
        diagram_code = diagram_code.replace('\\n', '\n')
        diagram_code = re.sub(r'```.*?```', '', diagram_code, flags=re.DOTALL)
        diagram_code = diagram_code.strip()
        
        # Determine height based on fullscreen mode
        diagram_height = 1200 if st.session_state.get("diagram_fullscreen") else 500
        
        # Create tabs
        tab1, tab2 = st.tabs(["🎯 Visual Diagram", "📝 Code"])
        
        with tab1:
            if st.session_state.get("diagram_fullscreen"):
                st.info("💡 Fullscreen mode active - Scroll down to see the complete diagram. Click 'Exit Fullscreen' above to return.")
            
            # Render using Mermaid.js with dynamic height
            mermaid_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
                <script>
                    mermaid.initialize({{ 
                        startOnLoad: true, 
                        theme: 'default',
                        flowchart: {{
                            useMaxWidth: true,
                            htmlLabels: true,
                            curve: 'basis'
                        }}
                    }});
                </script>
                <style>
                    body {{ 
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        background: white;
                        overflow: auto;
                    }}
                    .mermaid {{ 
                        text-align: center;
                        width: 100%;
                        height: auto;
                        min-height: {diagram_height - 100}px;
                    }}
                    svg {{
                        max-width: 100%;
                        height: auto;
                    }}
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
            components.html(mermaid_html, height=diagram_height, scrolling=True)
        
        with tab2:
            st.code(diagram_code, language="mermaid")
            
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    label="💾 Download Mermaid Code",
                    data=diagram_code,
                    file_name="flow_diagram.mmd",
                    mime="text/plain",
                    key="download_btn"
                )
            with col_dl2:
                # Option to download as image (via link to mermaid.live)
                import urllib.parse
                encoded_diagram = urllib.parse.quote(diagram_code)
                mermaid_live_url = f"https://mermaid.live/edit#pako:{encoded_diagram}"
                st.markdown(f"[🖼️ Open in Mermaid Live Editor]({mermaid_live_url})")
    # if st.session_state.get("flow_diagram"):
    #     st.markdown("---")
    #     st.header("📊 Visual Flow Diagram")
        
    #     if st.button("🔄 Regenerate", key="regen_btn"):
    #         if st.session_state.get("story_data"):
    #             try:
    #                 flow_diagram = generate_flow_diagram(st.session_state["story_data"])
    #                 st.session_state["flow_diagram"] = flow_diagram
    #                 st.rerun()
    #             except Exception as e:
    #                 st.error(f"Failed: {e}")
        
    #     diagram_code = st.session_state["flow_diagram"]
        
    #     # Clean diagram code
    #     if "content='" in diagram_code:
    #         match = re.search(r'(graph\s+\w+.*?)(?=\'|$)', diagram_code, re.DOTALL)
    #         if match:
    #             diagram_code = match.group(1)
        
    #     diagram_code = diagram_code.replace('\\n', '\n')
    #     diagram_code = re.sub(r'```.*?```', '', diagram_code, flags=re.DOTALL)
    #     diagram_code = diagram_code.strip()
        
    #     # Create tabs
    #     tab1, tab2 = st.tabs(["🎯 Visual Diagram", "📝 Code"])
        
    #     with tab1:
    #         # Render using Mermaid.js
    #         mermaid_html = f"""
    #         <!DOCTYPE html>
    #         <html>
    #         <head>
    #             <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    #             <script>
    #                 mermaid.initialize({{ startOnLoad: true, theme: 'default' }});
    #             </script>
    #             <style>
    #                 body {{ 
    #                     font-family: Arial, sans-serif;
    #                     margin: 20px;
    #                     background: white;
    #                 }}
    #                 .mermaid {{ text-align: center; }}
    #             </style>
    #         </head>
    #         <body>
    #             <div class="mermaid">
    # {diagram_code}
    #             </div>
    #         </body>
    #         </html>
    #         """
            
    #         import streamlit.components.v1 as components
    #         components.html(mermaid_html, height=500, scrolling=True)
        
    #     with tab2:
    #         st.code(diagram_code, language="mermaid")
    #         st.download_button(
    #             label="💾 Download Mermaid Code",
    #             data=diagram_code,
    #             file_name="flow_diagram.mmd",
    #             mime="text/plain",
    #             key="download_btn"
    #         )

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

    # SMART TESTING FOCUS / RISK ANALYSIS SECTION
    st.markdown("---")
    st.header("🧪 Smart Testing Focus & Risk Zones")

    if st.session_state.get("story_data") and st.session_state.get("vectorstore"):
        selected_story = st.selectbox(
            "Select a story to analyze for linked PRs:",
            options=[t for t, _ in st.session_state["story_data"]],
            help="Choose a story whose PR impact you want to analyze"
        )
        
        if st.button("🔍 Analyze Linked PRs & Suggest Test Focus"):
            story_text = next((text for title, text in st.session_state["story_data"] if title == selected_story), "")
            story_id = re.sub(r"[^\d]", "", selected_story)  # Extract numeric ID
            with st.spinner("Analyzing PRs and generating smart testing insights..."):
                pr_analysis = generate_pr_test_focus(
                    org=st.session_state["org"],
                    project=st.session_state["project"],
                    story_id=story_id,
                    pat=st.session_state["pat"],
                    story_text=story_text
                )
                st.session_state["pr_analysis"] = pr_analysis
                st.success("✅ Analysis complete!")

    if st.session_state.get("pr_analysis"):
        st.markdown(st.session_state["pr_analysis"], unsafe_allow_html=True)
    else:
        st.info("Select a story and click 'Analyze Linked PRs & Suggest Test Focus' to generate insights.")

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
            for key in ["vectorstore", "retrieval_chain", "docs_sources", "layman_summary", "chat_history", "qa_history", "flow_diagram", "story_data", "pr_analysis"]:
                st.session_state.pop(key, None)
            st.rerun()

# -------------------------
# TAB 2: Test Case Generator (New functionality)
# -------------------------
with main_tab2:
    st.header("🧪 AI-Powered Test Case Generator")
    st.markdown("""
    Generate comprehensive test cases for Azure DevOps user stories using AI.
    The system will create 30-100 test cases covering all scenarios including edge cases, negative testing, and more.
    """)
    
    # Check dependencies
    if not GROQ_CLIENT_AVAILABLE:
        st.error("⚠️ Groq client not installed. Run: `pip install groq`")
    if pd is None or openpyxl is None:
        st.error("⚠️ Excel libraries not installed. Run: `pip install pandas openpyxl`")
    
    test_mode = st.radio("Select input mode for test case generation:", 
                         ["Story IDs (comma-separated)", "WIQL Query"], 
                         key="test_mode")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if test_mode == "Story IDs (comma-separated)":
            test_story_ids = st.text_input(
                "Enter story/work item IDs (comma-separated)", 
                placeholder="e.g., 123, 456, 789",
                key="test_story_ids"
            )
        else:
            test_wiql_query = st.text_area(
                "Enter WIQL query", 
                placeholder="SELECT [System.Id] FROM WorkItems WHERE ...",
                key="test_wiql"
            )
    
    with col2:
        st.markdown("### Settings")
        
        test_model = st.selectbox(
            "AI Model",
            ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "groq/compound","qwen/qwen3-32b"],
            key="test_model"
        )
        generate_test_btn = st.button("🚀 Generate Test Cases", type="primary")
    
    test_status = st.empty()
    
    if generate_test_btn:
        if not st.session_state.get("groq_key"):
            test_status.error("❌ GROQ_API_KEY not found. Please set it in your .env file or sidebar.")
        elif not GROQ_CLIENT_AVAILABLE or pd is None or openpyxl is None:
            test_status.error("❌ Required libraries not installed. Check error messages above.")
        else:
            try:
                test_status.info("🔍 Collecting story IDs...")
                
                # Collect story IDs
                story_ids_list = []
                if test_mode == "Story IDs (comma-separated)":
                    if not test_story_ids or not test_story_ids.strip():
                        test_status.warning("⚠️ Please enter at least one story ID.")
                    else:
                        story_ids_list = [s.strip() for s in re.split(r"[,\s]+", test_story_ids.strip()) if s.strip()]
                else:
                    if not test_wiql_query or not test_wiql_query.strip():
                        test_status.warning("⚠️ Please provide a WIQL query.")
                    else:
                        test_status.info("🔍 Running WIQL query...")
                        story_ids_list = azure_run_wiql(
                            st.session_state["org"],
                            st.session_state["project"],
                            test_wiql_query,
                            st.session_state["pat"]
                        )
                        test_status.info(f"✅ Found {len(story_ids_list)} work items")
                
                if not story_ids_list:
                    test_status.error("❌ No stories found to process.")
                else:
                    test_status.info(f"⚙️ Generating test cases for {len(story_ids_list)} stories...")
                    
                    # Generate test cases
                    excel_file = generate_test_cases_excel(
                        story_ids_list,
                        st.session_state["org"],
                        st.session_state["project"],
                        st.session_state["pat"],
                        st.session_state["groq_key"],
                        test_model
                    )
                    
                    st.session_state["test_cases_excel"] = excel_file
                    test_status.success(f"✅ Test cases generated successfully for {len(story_ids_list)} stories!")
                    
            except Exception as e:
                test_status.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Download section
    if st.session_state.get("test_cases_excel"):
        st.markdown("---")
        st.subheader("📥 Download Test Cases")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.download_button(
                label="💾 Download Excel File",
                data=st.session_state["test_cases_excel"],
                file_name=f"test_cases_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_test_cases"
            )
        
        with col2:
            if st.button("🗑️ Clear Results", key="clear_test_results"):
                st.session_state["test_cases_excel"] = None
                st.rerun()
        
        st.info("ℹ️ The Excel file contains formatted test cases with proper line breaks. Each story has its own sheet.")
    
    # Instructions
    st.markdown("---")
    with st.expander("ℹ️ How to use Test Case Generator"):
        st.markdown("""
        ### Instructions
        
        1. **Configure Azure DevOps credentials** in the sidebar (Organization, Project, PAT)
        2. **Ensure GROQ_API_KEY** is set in your environment or .env file
        3. **Choose input mode:**
           - **Story IDs**: Enter comma-separated work item IDs (e.g., 123, 456, 789)
           - **WIQL Query**: Write a custom WIQL query to fetch multiple stories
        4. **Select AI model** (llama-3.3-70b-versatile recommended)
        5. **Click "Generate Test Cases"** and wait for processing
        6. **Download the Excel file** with comprehensive test cases
        
        ### What you get
        - 30-100 test cases per story covering:
          - ✅ Positive scenarios
          - ❌ Negative scenarios  
          - 🔄 Edge cases and boundary values
          - 🔗 Integration scenarios
          - 🔐 Security test cases
          - ♿ Accessibility checks
          - 📱 Compatibility testing
        
        ### Excel Format
        Each sheet contains:
        - Test Case ID
        - Priority (High/Medium/Low)
        - Test Case title
        - Pre-conditions/Test Data
        - Test Steps (formatted with line breaks)
        - Expected Results
        """)