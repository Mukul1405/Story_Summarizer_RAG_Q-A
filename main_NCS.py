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

try:
    import pandas as pd
    import openpyxl
except Exception:
    pd = None
    openpyxl = None

try:
    try:
        from langchain_text_splitters import CharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import CharacterTextSplitter

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    try:
        from langchain.chains import ConversationalRetrievalChain
    except (ImportError, ModuleNotFoundError):
        try:
            from langchain.chains.retrieval import ConversationalRetrievalChain
        except (ImportError, ModuleNotFoundError):
            from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

    try:
        from langchain.memory import ConversationBufferMemory
    except ImportError:
        try:
            from langchain_community.chat_message_histories import ChatMessageHistory
            from langchain.memory import ConversationBufferMemory
        except ImportError:
            from langchain_core.chat_history import BaseChatMessageHistory
            from langchain.memory import ConversationBufferMemory

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

USE_GROQ = False
ChatGroq = None
try:
    from langchain_groq import ChatGroq  # type: ignore
    USE_GROQ = True
except Exception:
    USE_GROQ = False
    ChatGroq = None

try:
    from groq import Groq
    GROQ_CLIENT_AVAILABLE = True
except Exception:
    GROQ_CLIENT_AVAILABLE = False
    Groq = None

# -------------------------
# Session / Env
# -------------------------

SOURCE_TYPES = ["Azure DevOps", "Jira", "Confluence"]

def load_env_to_session():
    load_dotenv()
    if "source_type" not in st.session_state:
        st.session_state["source_type"] = "Azure DevOps"
    # Azure DevOps
    if "org" not in st.session_state:
        st.session_state["org"] = os.getenv("AZURE_DEVOPS_ORG", "")
    if "project" not in st.session_state:
        st.session_state["project"] = os.getenv("AZURE_DEVOPS_PROJECT", "")
    if "pat" not in st.session_state:
        st.session_state["pat"] = os.getenv("AZURE_DEVOPS_PAT", "")
    # Jira
    if "jira_base_url" not in st.session_state:
        st.session_state["jira_base_url"] = os.getenv("JIRA_BASE_URL", "")
    if "jira_email" not in st.session_state:
        st.session_state["jira_email"] = os.getenv("JIRA_EMAIL", "")
    if "jira_token" not in st.session_state:
        st.session_state["jira_token"] = os.getenv("JIRA_PAT", "")
    if "jira_project" not in st.session_state:
        st.session_state["jira_project"] = os.getenv("JIRA_PROJECT", "")
    if "jira_verify_ssl" not in st.session_state:
        st.session_state["jira_verify_ssl"] = os.getenv("JIRA_VERIFY", "true").lower() != "false"
    # Confluence
    if "conf_base_url" not in st.session_state:
        st.session_state["conf_base_url"] = os.getenv("CONFLUENCE_BASE_URL", os.getenv("JIRA_BASE_URL", ""))
    if "conf_email" not in st.session_state:
        st.session_state["conf_email"] = os.getenv("CONFLUENCE_EMAIL", os.getenv("JIRA_EMAIL", ""))
    if "conf_token" not in st.session_state:
        st.session_state["conf_token"] = os.getenv("CONFLUENCE_TOKEN", os.getenv("JIRA_PAT", ""))
    if "conf_space" not in st.session_state:
        st.session_state["conf_space"] = os.getenv("CONFLUENCE_SPACE", "")
    if "conf_verify_ssl" not in st.session_state:
        st.session_state["conf_verify_ssl"] = True
    # Shared
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

# -------------------------
# Shared utilities
# -------------------------

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

def adf_to_text(adf_doc) -> str:
    """Convert Atlassian Document Format (ADF) JSON to plain text."""
    if not adf_doc:
        return ""
    if isinstance(adf_doc, str):
        return adf_doc
    texts = []

    def _extract(node):
        if not isinstance(node, dict):
            return
        ntype = node.get("type", "")
        if ntype == "text":
            texts.append(node.get("text", ""))
        elif ntype == "hardBreak":
            texts.append("\n")
        elif ntype in ("paragraph", "heading", "listItem", "blockquote"):
            for child in node.get("content", []):
                _extract(child)
            texts.append("\n")
        else:
            for child in node.get("content", []):
                _extract(child)

    _extract(adf_doc)
    return "".join(texts).strip()

# -------------------------
# Azure DevOps functions
# -------------------------

import urllib.parse

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
                break
            resp.raise_for_status()
            data = resp.json()
            for c in data.get("comments", []):
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
                    all_comments.append({"author": author, "text": text, "created": created, "weight": weight})
            continuation_token = resp.headers.get("x-ms-continuationtoken")
            if not continuation_token:
                break

        all_comments.sort(key=lambda x: (x["weight"], x["created"]), reverse=True)
        return all_comments
    except Exception as e:
        st.warning(f"Comment fetch failed for {work_item_id}: {e}")
        return []

def azure_fetch_linked_prs(org: str, project: str, work_item_id: str, pat: str) -> List[dict]:
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
                decoded = urllib.parse.unquote(encoded)
                parts = decoded.split("/")
                if len(parts) == 3:
                    project_id, repo_id, pr_id = parts
                    prs.append({"projectId": project_id, "repoId": repo_id, "prId": pr_id})
        return prs
    except Exception as e:
        st.warning(f"PR fetch failed for {work_item_id}: {e}")
        return []

def azure_fetch_pr_file_diffs(org: str, project_id: str, repo_id: str, pr_id: str, pat: str) -> List[dict]:
    try:
        org_clean = clean_org_for_api(org)
        base_url = f"https://dev.azure.com/{org_clean}"
        st.info("🔍 Attempting to fetch PR diffs...")

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
        iter_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/pullRequests/{pr_id}/iterations?api-version=7.1"
        iter_resp = requests.get(iter_url, auth=("", pat))
        if iter_resp.status_code != 200:
            st.warning(f"⚠️ Iterations API failed: {iter_resp.status_code}")
            return []

        iterations = iter_resp.json().get("value", [])
        if not iterations:
            st.warning("⚠️ No iterations found")
            return []

        latest_iter_id = iterations[-1].get("id")
        st.info(f"📋 Using iteration {latest_iter_id}")

        changes_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/pullRequests/{pr_id}/iterations/{latest_iter_id}/changes?api-version=7.1"
        changes_resp = requests.get(changes_url, auth=("", pat))
        if changes_resp.status_code != 200:
            st.error(f"❌ Changes API failed: {changes_resp.status_code}")
            return []

        change_entries = changes_resp.json().get("changeEntries", [])
        st.info(f"📊 Found {len(change_entries)} changed files")

        for idx, entry in enumerate(change_entries[:15], 1):
            item = entry.get("item", {})
            change_type = entry.get("changeType", "edit")
            path = item.get("path", "")
            if not path or item.get("gitObjectType") == "tree":
                continue

            file_name = path.split("/")[-1]
            extension = path.split(".")[-1] if "." in path else "unknown"
            st.info(f"  📄 {idx}. Processing {file_name}...")

            diff_content = "[Diff not available]"
            if source_commit and target_commit and change_type.lower() in ["edit", "add"]:
                try:
                    source_blob = item.get("objectId")
                    if source_blob:
                        blob_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/blobs/{source_blob}?api-version=7.1"
                        blob_resp = requests.get(blob_url, auth=("", pat), timeout=5)
                        if blob_resp.status_code == 200:
                            content = blob_resp.text[:2000]
                            diff_content = f"=== Modified: {file_name} ===\n\n{content}\n\n[Content truncated...]"
                            st.success(f"    ✓ Got file content for {file_name}")
                except Exception as e:
                    diff_content = f"[Error: {str(e)[:100]}]"

            result.append({
                "path": path, "fileName": file_name, "extension": extension,
                "changeType": change_type, "diff": diff_content,
                "summary": f"File: {file_name}\nPath: {path}\nChange: {change_type}"
            })

        st.success(f"✅ Successfully processed {len(result)} files")
        return result
    except Exception as e:
        st.error(f"❌ Exception in PR diff fetch: {e}")
        return []

def azure_fetch_pr_metadata_only(org: str, project_id: str, repo_id: str, pr_id: str, pat: str) -> List[dict]:
    try:
        org_clean = clean_org_for_api(org)
        base_url = f"https://dev.azure.com/{org_clean}"
        commits_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/pullRequests/{pr_id}/commits?api-version=7.1"
        commits_resp = requests.get(commits_url, auth=("", pat))
        if commits_resp.status_code != 200:
            return []
        commits = commits_resp.json().get("value", [])
        result = []
        file_extensions = set()
        if commits:
            latest_commit = commits[-1].get("commitId")
            changes_url = f"{base_url}/{project_id}/_apis/git/repositories/{repo_id}/commits/{latest_commit}/changes?api-version=7.1"
            changes_resp = requests.get(changes_url, auth=("", pat))
            if changes_resp.status_code == 200:
                for change in changes_resp.json().get("changes", [])[:20]:
                    item = change.get("item", {})
                    path = item.get("path", "")
                    if not path or item.get("gitObjectType") == "tree":
                        continue
                    file_name = path.split("/")[-1]
                    extension = path.split(".")[-1] if "." in path else "unknown"
                    change_type = change.get("changeType", "edit")
                    file_extensions.add(extension)
                    result.append({
                        "path": path, "fileName": file_name, "extension": extension,
                        "changeType": change_type,
                        "diff": f"[Metadata only - {change_type} operation on {extension} file]",
                        "summary": f"File: {file_name}\nPath: {path}\nType: {extension}\nChange: {change_type}"
                    })
        st.info(f"📊 Metadata-only mode: {len(result)} files, types: {', '.join(file_extensions)}")
        return result
    except Exception as e:
        st.error(f"Even metadata fetch failed: {e}")
        return []

def azure_fetch_pr_changes(org: str, project_id: str, repo_id: str, pr_id: str, pat: str) -> List[dict]:
    results = azure_fetch_pr_file_diffs(org, project_id, repo_id, pr_id, pat)
    if results:
        return results
    st.warning("⚠️ Falling back to metadata-only analysis (compliance-safe mode)")
    return azure_fetch_pr_metadata_only(org, project_id, repo_id, pr_id, pat)

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
    return [str(i.get("id")) for i in data.get("workItems", []) if i.get("id")]

# -------------------------
# Jira functions
# -------------------------

def _jira_headers(email: str, token: str) -> dict:
    creds = base64.b64encode(f"{email}:{token}".encode()).decode()
    return {"Authorization": f"Basic {creds}", "Content-Type": "application/json", "Accept": "application/json"}

def jira_fetch_issue(base_url: str, email: str, token: str, issue_key: str, verify_ssl: bool = True) -> Tuple[str, str]:
    """Fetch a Jira issue and return (description, acceptance_criteria)."""
    base_url = base_url.rstrip("/")
    url = f"{base_url}/rest/api/3/issue/{issue_key}"
    resp = requests.get(url, headers=_jira_headers(email, token), verify=verify_ssl)
    if resp.status_code == 401:
        raise RuntimeError(f"Jira authentication failed for {issue_key}. Check email and API token.")
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch Jira issue {issue_key}: {resp.status_code} - {resp.text[:300]}")
    fields = resp.json().get("fields", {})

    desc_raw = fields.get("description") or ""
    description = adf_to_text(desc_raw) if isinstance(desc_raw, dict) else strip_html_to_text(str(desc_raw))

    ac = ""
    for key, val in fields.items():
        if "acceptance" in key.lower() and val:
            ac = adf_to_text(val) if isinstance(val, dict) else strip_html_to_text(str(val))
            break

    return description, ac

def jira_fetch_issue_title(base_url: str, email: str, token: str, issue_key: str, verify_ssl: bool = True) -> str:
    base_url = base_url.rstrip("/")
    url = f"{base_url}/rest/api/3/issue/{issue_key}?fields=summary"
    resp = requests.get(url, headers=_jira_headers(email, token), verify=verify_ssl)
    if resp.status_code == 200:
        return resp.json().get("fields", {}).get("summary", issue_key)
    return issue_key

def jira_fetch_issue_comments(base_url: str, email: str, token: str, issue_key: str, verify_ssl: bool = True) -> List[dict]:
    try:
        base_url = base_url.rstrip("/")
        all_comments = []
        url = f"{base_url}/rest/api/3/issue/{issue_key}/comment?maxResults=100&orderBy=created"
        while url:
            resp = requests.get(url, headers=_jira_headers(email, token), verify=verify_ssl)
            if resp.status_code != 200:
                break
            data = resp.json()
            for c in data.get("comments", []):
                author = c.get("author", {}).get("displayName", "Unknown")
                body_raw = c.get("body") or ""
                text = adf_to_text(body_raw) if isinstance(body_raw, dict) else strip_html_to_text(str(body_raw))
                created = c.get("created", "")
                if text.strip():
                    all_comments.append({"author": author, "text": text.strip(), "created": created, "weight": 1})
            url = data.get("nextPage")
        return all_comments
    except Exception as e:
        st.warning(f"Jira comment fetch failed for {issue_key}: {e}")
        return []

def jira_run_jql(base_url: str, email: str, token: str, jql: str, verify_ssl: bool = True) -> List[str]:
    if not all([base_url, email, token, jql]):
        raise ValueError("Jira base URL, email, token, and JQL are required.")
    base_url = base_url.rstrip("/")
    url = f"{base_url}/rest/api/3/search"
    body = {"jql": jql, "maxResults": 100, "fields": ["key", "summary"]}
    resp = requests.post(url, headers=_jira_headers(email, token), json=body, verify=verify_ssl)
    if resp.status_code != 200:
        raise RuntimeError(f"JQL query failed: {resp.status_code} - {resp.text[:500]}")
    return [issue.get("key") for issue in resp.json().get("issues", []) if issue.get("key")]

# -------------------------
# Confluence functions
# -------------------------

def confluence_fetch_page(base_url: str, email: str, token: str, page_id: str, verify_ssl: bool = True) -> Tuple[str, str]:
    """Fetch a Confluence page and return (title, content_text)."""
    base_url = base_url.rstrip("/")
    url = f"{base_url}/wiki/rest/api/content/{page_id}?expand=body.storage,title"
    resp = requests.get(url, headers=_jira_headers(email, token), verify=verify_ssl)
    if resp.status_code == 401:
        raise RuntimeError(f"Confluence authentication failed for page {page_id}. Check email and API token.")
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch Confluence page {page_id}: {resp.status_code} - {resp.text[:300]}")
    data = resp.json()
    title = data.get("title", f"Page-{page_id}")
    html_content = data.get("body", {}).get("storage", {}).get("value", "")
    return title, strip_html_to_text(html_content)

def confluence_search_cql(base_url: str, email: str, token: str, cql: str, verify_ssl: bool = True) -> List[str]:
    if not all([base_url, email, token, cql]):
        raise ValueError("Confluence base URL, email, token, and CQL are required.")
    base_url = base_url.rstrip("/")
    encoded_cql = requests.utils.quote(cql)
    url = f"{base_url}/wiki/rest/api/content/search?cql={encoded_cql}&limit=50"
    resp = requests.get(url, headers=_jira_headers(email, token), verify=verify_ssl)
    if resp.status_code != 200:
        raise RuntimeError(f"CQL query failed: {resp.status_code} - {resp.text[:500]}")
    return [result.get("id") for result in resp.json().get("results", []) if result.get("id")]

# -------------------------
# Document / file utilities
# -------------------------

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
        reader = PyPDF2.PdfReader(BytesIO(content))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                pages.append("")
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

# -------------------------
# LLM helpers
# -------------------------

def extract_llm_text(llm_result) -> str:
    if isinstance(llm_result, str):
        return llm_result
    try:
        if hasattr(llm_result, "content"):
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
        if "choices" in llm_result and llm_result["choices"]:
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
    raise RuntimeError("Groq LLM not configured. Set GROQ_API_KEY in .env and install langchain-groq.")

def strict_split(text, chunk_size=1000, chunk_overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

# -------------------------
# Flow diagram
# -------------------------

def generate_flow_diagram(title_text_pairs: List[Tuple[str, str]]) -> str:
    try:
        llm = get_llm_instance()
        stories_text = "\n\n".join([
            f"Item {idx+1}: {title}\nContent: {text[:1000]}"
            for idx, (title, text) in enumerate(title_text_pairs[:10])
        ])
        prompt = f"""Analyze the following items and create a Mermaid flowchart diagram showing their relationships and workflow.

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

ITEMS:
{stories_text}

Generate Mermaid flowchart (graph LR format):"""

        result = llm.invoke(prompt)
        mermaid_code = extract_llm_text(result).strip()
        mermaid_code = re.sub(r'^```mermaid\s*', '', mermaid_code)
        mermaid_code = re.sub(r'^```\s*', '', mermaid_code)
        mermaid_code = re.sub(r'```\s*$', '', mermaid_code).strip()
        if not mermaid_code.startswith("graph"):
            mermaid_code = "graph LR\n" + mermaid_code
        return mermaid_code
    except Exception as e:
        st.warning(f"Using fallback diagram (LLM error: {str(e)[:100]})")
        mermaid_code = "graph LR\n    Start([Start])\n"
        for idx, (title, _) in enumerate(title_text_pairs[:8]):
            clean_title = re.sub(r"[^\w\s]", "", title.replace("-", " ").replace("_", " "))[:40]
            clean_title = " ".join(clean_title.split())
            node_id = f"S{idx+1}"
            mermaid_code += f'    {node_id}["{clean_title}"]\n'
            if idx == 0:
                mermaid_code += f"    Start --> {node_id}\n"
            else:
                mermaid_code += f"    S{idx} --> {node_id}\n"
        last_node = f"S{min(len(title_text_pairs), 8)}"
        mermaid_code += f"    End([Complete])\n    {last_node} --> End\n"
        return mermaid_code

# -------------------------
# PR analysis (Azure DevOps only)
# -------------------------

def generate_detailed_code_review(changes: List[dict], story_text: str) -> str:
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
    prompt = f"""You are a SENIOR CODE REVIEWER analyzing actual code changes.

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
    return extract_llm_text(llm.invoke(prompt)) + "\n\n*✅ Based on actual code review*"

def generate_metadata_based_analysis(changes: List[dict], story_text: str) -> str:
    by_type = {}
    for change in changes:
        ext = change.get("extension", "unknown")
        by_type.setdefault(ext, []).append(change)
    context = []
    for ext, files in by_type.items():
        context.append(f"\n**{ext.upper()} Files ({len(files)}):**")
        for f in files[:5]:
            context.append(f"- {f.get('fileName')} ({f.get('changeType')})")
    stats = f"**Files Changed:** {len(changes)}\n**File Types:** {', '.join(by_type.keys())}\n{''.join(context)}"
    prompt = f"""You are a QA ARCHITECT analyzing PR metadata.

**STORY CONTEXT:**
{story_text[:1000]}

{stats}

### 🔴 HIGH RISK AREAS
- Backend files → business logic bugs
- Database files → data integrity issues
- Config files → environment bugs

### 🎯 TESTING PRIORITIES
- Must-test scenarios based on changed areas

### ⚠️ COMMON PITFALLS
- Typical bugs for these file types
"""
    llm = get_llm_instance()
    return extract_llm_text(llm.invoke(prompt)) + "\n\n*🔒 Compliance-safe: metadata-only analysis (no source code sent to AI)*"

def generate_pr_test_focus(org: str, project: str, story_id: str, pat: str, story_text: str) -> str:
    try:
        st.info(f"🔍 Fetching linked PRs for story {story_id}...")
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
                sample_diff = pr_changes[0].get("diff", "")
                if "[Metadata only" not in sample_diff and "[Diff not available]" not in sample_diff:
                    has_actual_diffs = True
                st.success(f"  ✓ Retrieved {len(pr_changes)} file changes")
                all_changes.extend(pr_changes)
            else:
                st.warning(f"  ⚠ No changes found for PR {pr['prId']}")
        if not all_changes:
            return "*(No changed files detected for linked PRs)*"
        st.info(f"📊 Total files to review: {len(all_changes)}")
        if has_actual_diffs:
            st.success("✅ Got actual code diffs - performing detailed code review")
            return generate_detailed_code_review(all_changes, story_text)
        else:
            st.warning("⚠️ Using metadata-only analysis (compliance-safe)")
            return generate_metadata_based_analysis(all_changes, story_text)
    except Exception as e:
        import traceback
        st.error(traceback.format_exc())
        return f"❌ Error during PR analysis: {e}"

# -------------------------
# Test case generation
# -------------------------

def sanitize_groq_output(text: str) -> str:
    text = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', text)
    text = re.sub(r"(?<=\{|\s)'([^']+)':", r'"\1":', text)
    text = re.sub(r"[\x00-\x1f\x7f]", "", text)
    return text

def make_groq_test_prompt(title: str, description: str, acceptance_criteria: str) -> str:
    return f"""
You are a senior test-design engineer. For the user story below, produce a highly comprehensive, structured list of test cases.
Your output MUST be valid JSON (an array of objects). Each object should have:
  - test_case_id
  - priority (High/Medium/Low)
  - test_case (short title)
  - precondition
  - test_steps (ordered list of steps as an array)
  - expected_result

User Story Title:
{title}

Description:
{description}

Acceptance Criteria:
{acceptance_criteria}

RULES:
1. Generate complete set of test cases ranging(30-100 cases), covering all realistic, edge, boundary, integration, negative, performance, usability, security, accessibility, and error handling scenarios.
2. Go beyond acceptance criteria: include positive, negative, boundary value, data validation, regression, integration, compatibility, performance, security, usability, and accessibility test cases.
3. Assign priority based on risk, business value, and acceptance criteria.
4. Each expected_result must be 1–2 sentences, concise yet clear.
5. Ensure deep coverage so that even if acceptance criteria are limited, cases extrapolate all flows.
6. Return only the JSON array, no extra commentary.
""".strip()

def call_groq_for_tests(prompt: str, api_key: str, model: str = "llama-3.3-70b-versatile", timeout: int = 120):
    if not GROQ_CLIENT_AVAILABLE:
        raise RuntimeError("Groq client not available. Install: pip install groq")
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=32768,
        timeout=timeout,
    )
    text = sanitize_groq_output(response.choices[0].message.content.strip())
    try:
        return json.loads(text)
    except Exception:
        first, last = text.find("["), text.rfind("]")
        if first != -1 and last != -1:
            try:
                return json.loads(text[first:last+1])
            except Exception as e:
                raise ValueError(f"Groq output not valid JSON after sanitizing:\n{text[:1000]}\nError: {str(e)}")
        raise ValueError(f"Groq output not valid JSON:\n{text[:1000]}")

def story_to_testcases(title: str, description_text: str, acceptance_text: str, groq_api_key: str, model: str):
    prompt = make_groq_test_prompt(title, description_text, acceptance_text)
    ai_results = call_groq_for_tests(prompt, api_key=groq_api_key, model=model)
    rows = []
    for idx, item in enumerate(ai_results, start=1):
        steps = item.get("test_steps") or []
        if isinstance(steps, str):
            steps = [s.strip() for s in steps.splitlines() if s.strip()]
        rows.append({
            "Test Case ID": item.get("test_case_id") or f"TC_{idx}",
            "Priority": item.get("priority") or "Medium",
            "Test Case": item.get("test_case") or "",
            "Pre-condition / Test Data": item.get("precondition") or "",
            "Test Steps": "<br>".join([f"{i+1}. {s}" for i, s in enumerate(steps)]),
            "Expected Result": item.get("expected_result") or "",
        })
    return pd.DataFrame(rows, columns=["Test Case ID", "Priority", "Test Case", "Pre-condition / Test Data", "Test Steps", "Expected Result"])

def replace_br_with_newline_in_memory(wb):
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        for row in ws.iter_rows():
            for cell in row:
                if isinstance(cell.value, str) and "<br>" in cell.value:
                    cell.value = cell.value.replace("<br>", "\n")
    return wb

def generate_test_cases_excel_azure(story_ids: List[str], org: str, project: str, pat: str, groq_key: str, model: str) -> BytesIO:
    if pd is None or openpyxl is None:
        raise RuntimeError("pandas and openpyxl are required.")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sid in story_ids:
            try:
                desc, ac = azure_fetch_work_item(org, project, sid, pat)
                org_clean = clean_org_for_api(org)
                url = f"https://dev.azure.com/{org_clean}/{project}/_apis/wit/workitems/{sid}?api-version=7.1"
                resp = requests.get(url, auth=("", pat))
                resp.raise_for_status()
                title = resp.json().get("fields", {}).get("System.Title", f"WorkItem_{sid}")
                st.info(f"Generating test cases for {sid} - {title}")
                df = story_to_testcases(title, desc, ac, groq_key, model)
                df.to_excel(writer, sheet_name=f"{sid}"[:31], index=False)
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"Failed for {sid}: {e}")
                pd.DataFrame([{"Test Case ID": "NA", "Priority": "NA", "Test Case": f"Generation failed for {sid}",
                                "Pre-condition / Test Data": "", "Test Steps": str(e)[:500], "Expected Result": ""}]).to_excel(
                    writer, sheet_name=f"{sid}"[:31], index=False)
    output.seek(0)
    wb = replace_br_with_newline_in_memory(openpyxl.load_workbook(output))
    final_output = BytesIO()
    wb.save(final_output)
    final_output.seek(0)
    return final_output

def generate_test_cases_excel_jira(issue_keys: List[str], base_url: str, email: str, token: str, groq_key: str, model: str, verify_ssl: bool = True) -> BytesIO:
    if pd is None or openpyxl is None:
        raise RuntimeError("pandas and openpyxl are required.")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for key in issue_keys:
            try:
                desc, ac = jira_fetch_issue(base_url, email, token, key, verify_ssl)
                title = jira_fetch_issue_title(base_url, email, token, key, verify_ssl)
                st.info(f"Generating test cases for {key} - {title}")
                df = story_to_testcases(title, desc, ac, groq_key, model)
                safe_sheet = re.sub(r"[\\/*?:\[\]]", "_", key)[:31]
                df.to_excel(writer, sheet_name=safe_sheet, index=False)
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"Failed for {key}: {e}")
                safe_sheet = re.sub(r"[\\/*?:\[\]]", "_", key)[:31]
                pd.DataFrame([{"Test Case ID": "NA", "Priority": "NA", "Test Case": f"Generation failed for {key}",
                                "Pre-condition / Test Data": "", "Test Steps": str(e)[:500], "Expected Result": ""}]).to_excel(
                    writer, sheet_name=safe_sheet, index=False)
    output.seek(0)
    wb = replace_br_with_newline_in_memory(openpyxl.load_workbook(output))
    final_output = BytesIO()
    wb.save(final_output)
    final_output.seek(0)
    return final_output

def generate_test_cases_excel_confluence(page_ids: List[str], base_url: str, email: str, token: str, groq_key: str, model: str, verify_ssl: bool = True) -> BytesIO:
    if pd is None or openpyxl is None:
        raise RuntimeError("pandas and openpyxl are required.")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for pid in page_ids:
            try:
                title, content = confluence_fetch_page(base_url, email, token, pid, verify_ssl)
                st.info(f"Generating test cases for page {pid} - {title}")
                df = story_to_testcases(title, content, "", groq_key, model)
                safe_sheet = re.sub(r"[\\/*?:\[\]]", "_", title[:20] or pid)[:31]
                df.to_excel(writer, sheet_name=safe_sheet, index=False)
                time.sleep(0.5)
            except Exception as e:
                st.warning(f"Failed for page {pid}: {e}")
                pd.DataFrame([{"Test Case ID": "NA", "Priority": "NA", "Test Case": f"Generation failed for page {pid}",
                                "Pre-condition / Test Data": "", "Test Steps": str(e)[:500], "Expected Result": ""}]).to_excel(
                    writer, sheet_name=f"page_{pid}"[:31], index=False)
    output.seek(0)
    wb = replace_br_with_newline_in_memory(openpyxl.load_workbook(output))
    final_output = BytesIO()
    wb.save(final_output)
    final_output.seek(0)
    return final_output

# -------------------------
# App UI
# -------------------------

st.set_page_config(page_title="Story Summarizer & Q&A — Multi-Source", layout="wide")
load_env_to_session()

st.title("Story Summarizer & Q&A — Multi-Source")

# -------- SIDEBAR --------
st.sidebar.header("Configuration")

source_type = st.sidebar.selectbox(
    "Data Source",
    SOURCE_TYPES,
    key="source_type",
    help="Choose the platform to ingest stories or pages from."
)

st.sidebar.markdown("---")

if source_type == "Azure DevOps":
    st.sidebar.subheader("Azure DevOps Credentials")
    st.sidebar.text_input("Organization (org)", key="org",
                          help="e.g. dpwhotfsonline or https://dev.azure.com/dpwhotfsonline")
    st.sidebar.text_input("Project", key="project")
    st.sidebar.text_input("Personal Access Token (PAT)", key="pat", type="password")
    st.sidebar.caption("PAT needs 'Work Items (Read)' permission.")

elif source_type == "Jira":
    st.sidebar.subheader("Jira Credentials")
    st.sidebar.text_input("Jira Base URL", key="jira_base_url",
                          help="e.g. https://yourcompany.atlassian.net")
    st.sidebar.text_input("Email", key="jira_email")
    st.sidebar.text_input("API Token", key="jira_token", type="password",
                          help="Generate at id.atlassian.com/manage-profile/security/api-tokens")
    st.sidebar.text_input("Default Project Key (optional)", key="jira_project",
                          help="Used to pre-fill JQL, e.g. PROJ")
    st.sidebar.checkbox("Verify SSL", key="jira_verify_ssl", value=True)

elif source_type == "Confluence":
    st.sidebar.subheader("Confluence Credentials")
    st.sidebar.text_input("Confluence Base URL", key="conf_base_url",
                          help="e.g. https://yourcompany.atlassian.net")
    st.sidebar.text_input("Email", key="conf_email")
    st.sidebar.text_input("API Token", key="conf_token", type="password")
    st.sidebar.text_input("Default Space Key (optional)", key="conf_space",
                          help="Used to pre-fill CQL, e.g. ENG")
    st.sidebar.checkbox("Verify SSL", key="conf_verify_ssl", value=True)

st.sidebar.markdown("---")
st.sidebar.write("GROQ_API_KEY: " + ("✅ Present" if st.session_state.get("groq_key") else "❌ Missing"))

# -------- TABS --------
main_tab1, main_tab2 = st.tabs(["📋 Summarizer & Q&A", "🧪 Test Case Generator"])

# =======================
# TAB 1: Summarizer & Q&A
# =======================
with main_tab1:

    # --- Input mode ---
    if source_type == "Azure DevOps":
        mode = st.radio("Input mode", ["Story IDs (comma-separated)", "WIQL Query", "Upload files (PDF/DOCX/TXT)"])
    elif source_type == "Jira":
        mode = st.radio("Input mode", ["Issue Keys (comma-separated)", "JQL Query", "Upload files (PDF/DOCX/TXT)"])
    else:  # Confluence
        mode = st.radio("Input mode", ["Page IDs (comma-separated)", "CQL Query", "Upload files (PDF/DOCX/TXT)"])

    col1, col2 = st.columns([3, 1])
    with col1:
        if mode in ("Story IDs (comma-separated)", "Issue Keys (comma-separated)", "Page IDs (comma-separated)"):
            label_map = {
                "Story IDs (comma-separated)": "Enter story/work item IDs (comma-separated)",
                "Issue Keys (comma-separated)": "Enter Jira issue keys (comma-separated)",
                "Page IDs (comma-separated)": "Enter Confluence page IDs (comma-separated)",
            }
            placeholder_map = {
                "Story IDs (comma-separated)": "e.g., 123, 456, 789",
                "Issue Keys (comma-separated)": "e.g., PROJ-123, PROJ-456",
                "Page IDs (comma-separated)": "e.g., 123456, 789012",
            }
            ids_input = st.text_input(label_map[mode], placeholder=placeholder_map[mode])
        elif mode in ("WIQL Query", "JQL Query", "CQL Query"):
            placeholder_map = {
                "WIQL Query": "SELECT [System.Id] FROM WorkItems WHERE [System.TeamProject] = @project AND ...",
                "JQL Query": "project = PROJ AND issuetype = Story AND status != Done ORDER BY created DESC",
                "CQL Query": 'space = "ENG" AND type = page AND title ~ "Feature" ORDER BY created DESC',
            }
            query_input = st.text_area(f"Enter {mode}", placeholder=placeholder_map[mode])
        else:
            uploaded_files = st.file_uploader(
                "Upload files (PDF, DOCX, TXT). Multiple allowed.",
                accept_multiple_files=True, type=["pdf", "docx", "txt", "md"]
            )

    with col2:
        st.markdown("### Options")
        append_mode = st.checkbox("Append to existing corpus", value=False)
        generate_diagram = st.checkbox("Generate Flow Diagram", value=True)
        process_btn = st.button("Fetch / Embed & Summarize")

    status = st.empty()
    embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    if process_btn:
        try:
            status.info("Collecting inputs...")
            title_text_pairs = []

            # ---- Azure DevOps ----
            if source_type == "Azure DevOps":
                org_v = st.session_state["org"]
                proj_v = st.session_state["project"]
                pat_v = st.session_state["pat"]

                if mode == "Story IDs (comma-separated)":
                    if not ids_input.strip():
                        st.warning("Please enter at least one story ID.")
                    else:
                        ids = [s.strip() for s in re.split(r"[,\s]+", ids_input.strip()) if s.strip()]
                        status.info(f"Fetching {len(ids)} work items from Azure DevOps...")
                        for sid in ids:
                            try:
                                desc, ac = azure_fetch_work_item(org_v, proj_v, sid, pat_v)
                                comments = azure_fetch_work_item_comments(org_v, proj_v, sid, pat_v)
                                comments_text = "\n".join(
                                    [f"- [{c['author']}] ({c['created']}) ⭐{c['weight']}: {c['text']}" for c in comments]
                                ) if comments else "No comments found."
                                combined = "\n\n".join(filter(None, [
                                    "Description:\n" + desc if desc else "",
                                    "Acceptance Criteria:\n" + ac if ac else "",
                                    "Comments:\n" + comments_text,
                                ]))
                                title_text_pairs.append((f"WorkItem-{sid}", combined.strip()))
                            except Exception as e:
                                st.warning(f"Failed to fetch {sid}: {e}")

                elif mode == "WIQL Query":
                    if not query_input.strip():
                        st.warning("Please provide a WIQL query.")
                    else:
                        status.info("Running WIQL...")
                        ids = azure_run_wiql(org_v, proj_v, query_input, pat_v)
                        st.info(f"WIQL returned {len(ids)} work items.")
                        for sid in ids:
                            try:
                                desc, ac = azure_fetch_work_item(org_v, proj_v, sid, pat_v)
                                combined = "\n\n".join(filter(None, [
                                    "Description:\n" + desc if desc else "",
                                    "Acceptance Criteria:\n" + ac if ac else "",
                                ]))
                                title_text_pairs.append((f"WorkItem-{sid}", combined.strip()))
                            except Exception as e:
                                st.warning(f"Failed to fetch {sid}: {e}")

                else:  # Upload
                    if not uploaded_files:
                        st.warning("Please upload at least one file.")
                    else:
                        for f in uploaded_files:
                            try:
                                title_text_pairs.append((f.name, extract_text_from_file(f)))
                            except Exception as e:
                                st.warning(f"Failed to read {f.name}: {e}")

            # ---- Jira ----
            elif source_type == "Jira":
                jira_url = st.session_state["jira_base_url"]
                jira_email = st.session_state["jira_email"]
                jira_token = st.session_state["jira_token"]
                jira_ssl = st.session_state["jira_verify_ssl"]

                if mode == "Issue Keys (comma-separated)":
                    if not ids_input.strip():
                        st.warning("Please enter at least one issue key.")
                    else:
                        keys = [s.strip() for s in re.split(r"[,\s]+", ids_input.strip()) if s.strip()]
                        status.info(f"Fetching {len(keys)} Jira issues...")
                        for key in keys:
                            try:
                                desc, ac = jira_fetch_issue(jira_url, jira_email, jira_token, key, jira_ssl)
                                title = jira_fetch_issue_title(jira_url, jira_email, jira_token, key, jira_ssl)
                                comments = jira_fetch_issue_comments(jira_url, jira_email, jira_token, key, jira_ssl)
                                comments_text = "\n".join(
                                    [f"- [{c['author']}] ({c['created']}): {c['text']}" for c in comments]
                                ) if comments else "No comments found."
                                combined = "\n\n".join(filter(None, [
                                    "Description:\n" + desc if desc else "",
                                    "Acceptance Criteria:\n" + ac if ac else "",
                                    "Comments:\n" + comments_text,
                                ]))
                                title_text_pairs.append((f"{key} - {title}", combined.strip()))
                            except Exception as e:
                                st.warning(f"Failed to fetch {key}: {e}")

                elif mode == "JQL Query":
                    if not query_input.strip():
                        st.warning("Please provide a JQL query.")
                    else:
                        status.info("Running JQL query...")
                        keys = jira_run_jql(jira_url, jira_email, jira_token, query_input, jira_ssl)
                        st.info(f"JQL returned {len(keys)} issues.")
                        for key in keys:
                            try:
                                desc, ac = jira_fetch_issue(jira_url, jira_email, jira_token, key, jira_ssl)
                                title = jira_fetch_issue_title(jira_url, jira_email, jira_token, key, jira_ssl)
                                combined = "\n\n".join(filter(None, [
                                    "Description:\n" + desc if desc else "",
                                    "Acceptance Criteria:\n" + ac if ac else "",
                                ]))
                                title_text_pairs.append((f"{key} - {title}", combined.strip()))
                            except Exception as e:
                                st.warning(f"Failed to fetch {key}: {e}")

                else:  # Upload
                    if not uploaded_files:
                        st.warning("Please upload at least one file.")
                    else:
                        for f in uploaded_files:
                            try:
                                title_text_pairs.append((f.name, extract_text_from_file(f)))
                            except Exception as e:
                                st.warning(f"Failed to read {f.name}: {e}")

            # ---- Confluence ----
            else:
                conf_url = st.session_state["conf_base_url"]
                conf_email = st.session_state["conf_email"]
                conf_token = st.session_state["conf_token"]
                conf_ssl = st.session_state["conf_verify_ssl"]

                if mode == "Page IDs (comma-separated)":
                    if not ids_input.strip():
                        st.warning("Please enter at least one page ID.")
                    else:
                        page_ids = [s.strip() for s in re.split(r"[,\s]+", ids_input.strip()) if s.strip()]
                        status.info(f"Fetching {len(page_ids)} Confluence pages...")
                        for pid in page_ids:
                            try:
                                title, content = confluence_fetch_page(conf_url, conf_email, conf_token, pid, conf_ssl)
                                title_text_pairs.append((f"Page-{pid}: {title}", content.strip()))
                            except Exception as e:
                                st.warning(f"Failed to fetch page {pid}: {e}")

                elif mode == "CQL Query":
                    if not query_input.strip():
                        st.warning("Please provide a CQL query.")
                    else:
                        status.info("Running CQL query...")
                        page_ids = confluence_search_cql(conf_url, conf_email, conf_token, query_input, conf_ssl)
                        st.info(f"CQL returned {len(page_ids)} pages.")
                        for pid in page_ids:
                            try:
                                title, content = confluence_fetch_page(conf_url, conf_email, conf_token, pid, conf_ssl)
                                title_text_pairs.append((f"Page-{pid}: {title}", content.strip()))
                            except Exception as e:
                                st.warning(f"Failed to fetch page {pid}: {e}")

                else:  # Upload
                    if not uploaded_files:
                        st.warning("Please upload at least one file.")
                    else:
                        for f in uploaded_files:
                            try:
                                title_text_pairs.append((f.name, extract_text_from_file(f)))
                            except Exception as e:
                                st.warning(f"Failed to read {f.name}: {e}")

            # ---- Embed & Summarize (common for all sources) ----
            if not title_text_pairs:
                status.error("No content collected. Fix errors above and try again.")
            else:
                st.session_state["story_data"] = title_text_pairs
                docs = build_docs_from_texts(title_text_pairs)
                if not docs:
                    status.error("No valid text extracted from inputs.")
                else:
                    status.info("Splitting & embedding documents into FAISS...")
                    split_docs = []
                    for d in docs:
                        for i, c in enumerate(strict_split(d.page_content, chunk_size=1000, chunk_overlap=200)):
                            split_docs.append(Document(page_content=c, metadata={"source": d.metadata.get("source"), "chunk": i}))
                    embeddings = HuggingFaceEmbeddings(model_name=embed_model_name)
                    new_vs = FAISS.from_documents(split_docs, embeddings)
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

                    if generate_diagram:
                        status.info("Generating flow diagram...")
                        try:
                            st.session_state["flow_diagram"] = generate_flow_diagram(title_text_pairs)
                        except Exception as e:
                            st.warning(f"Could not generate flow diagram: {e}")

                    status.info("Generating summary...")
                    combined_text = "\n\n".join([f"{title}\n\n{text[:5000]}" for title, text in title_text_pairs])
                    source_label = {"Azure DevOps": "Azure DevOps user stories", "Jira": "Jira issues", "Confluence": "Confluence pages"}.get(source_type, "items")
                    summary_prompt = f"""
You are an expert Agile analyst and QA architect.
Summarize the following {source_label} clearly in Markdown format.

Make sure to:
1. Extract main objectives from the Description and Acceptance Criteria (if present).
2. Analyze any comments or discussion notes for decisions, deferrals, or concerns.
3. Highlight any risk areas, incomplete work, or postponed validations.
4. Give a short "Discussion Summary" section for each item.

Each item summary should follow this format:

### 🧩 {{Item Title}}

**Overview**
- One or two lines explaining what this item is about.

**Layman Summary**
- Summarize the whole item in plain terms with bullet points.

**P0 Scenarios**
- The most critical scenarios that are mandatory.

**Acceptance & Completion Criteria**
- Summarize the acceptance or done conditions.

**Comment Insights**
- Mention relevant comments or discussion takeaways.

---

CONTENT START
{combined_text}
CONTENT END
"""
                    try:
                        raw_out = get_llm_instance().invoke(summary_prompt)
                        st.session_state["layman_summary"] = extract_llm_text(raw_out)
                    except Exception as e:
                        st.session_state["layman_summary"] = f"(Summary not generated: {e})"

                    status.success("Done!")
        except Exception as e:
            status.error(f"Failed: {e}")

    # Flow Diagram
    if st.session_state.get("flow_diagram"):
        st.markdown("---")
        st.header("📊 Visual Flow Diagram")
        if st.button("🔄 Regenerate", key="regen_btn"):
            if st.session_state.get("story_data"):
                try:
                    st.session_state["flow_diagram"] = generate_flow_diagram(st.session_state["story_data"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

        diagram_code = st.session_state["flow_diagram"]
        if "content='" in diagram_code:
            m = re.search(r"(graph\s+\w+.*?)(?='|$)", diagram_code, re.DOTALL)
            if m:
                diagram_code = m.group(1)
        diagram_code = diagram_code.replace("\\n", "\n")
        diagram_code = re.sub(r"```.*?```", "", diagram_code, flags=re.DOTALL).strip()

        tab1, tab2 = st.tabs(["🎯 Visual Diagram", "📝 Code"])
        with tab1:
            mermaid_html = f"""<!DOCTYPE html><html><head>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>
<style>body {{ font-family: Arial, sans-serif; margin: 20px; background: white; }} .mermaid {{ text-align: center; }}</style>
</head><body><div class="mermaid">{diagram_code}</div></body></html>"""
            import streamlit.components.v1 as components
            components.html(mermaid_html, height=500, scrolling=True)
        with tab2:
            st.code(diagram_code, language="mermaid")
            st.download_button("💾 Download Mermaid Code", data=diagram_code,
                               file_name="flow_diagram.mmd", mime="text/plain", key="download_btn")

    # Summary
    st.markdown("---")
    st.header("📄 Summary")
    if st.session_state.get("layman_summary"):
        summary = st.session_state["layman_summary"]
        if isinstance(summary, str):
            summary = summary.replace("\\n", "\n")
        st.markdown(summary, unsafe_allow_html=True)
    else:
        st.info("No summary yet.")

    # Q&A
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
                result = st.session_state["retrieval_chain"].invoke({"question": user_q})
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

    # Smart Testing Focus — Azure DevOps only
    if source_type == "Azure DevOps":
        st.markdown("---")
        st.header("🧪 Smart Testing Focus & Risk Zones")
        if st.session_state.get("story_data") and st.session_state.get("vectorstore"):
            selected_story = st.selectbox(
                "Select a story to analyze for linked PRs:",
                options=[t for t, _ in st.session_state["story_data"]],
            )
            if st.button("🔍 Analyze Linked PRs & Suggest Test Focus"):
                story_text = next((text for title, text in st.session_state["story_data"] if title == selected_story), "")
                story_id = re.sub(r"[^\d]", "", selected_story)
                with st.spinner("Analyzing PRs and generating smart testing insights..."):
                    pr_analysis = generate_pr_test_focus(
                        org=st.session_state["org"],
                        project=st.session_state["project"],
                        story_id=story_id,
                        pat=st.session_state["pat"],
                        story_text=story_text,
                    )
                    st.session_state["pr_analysis"] = pr_analysis
                    st.success("✅ Analysis complete!")
        if st.session_state.get("pr_analysis"):
            st.markdown(st.session_state["pr_analysis"], unsafe_allow_html=True)
        else:
            st.info("Select a story and click 'Analyze Linked PRs' to generate insights.")

    # Debug / Reset
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📄 Sources", key="src_btn"):
            st.json(st.session_state.get("docs_sources", []))
    with col2:
        if st.button("🎨 Regenerate Diagram", key="diag_btn"):
            if st.session_state.get("story_data"):
                st.session_state["flow_diagram"] = generate_flow_diagram(st.session_state["story_data"])
                st.rerun()
    with col3:
        if st.button("🗑️ Reset All", key="reset_btn"):
            for key in ["vectorstore", "retrieval_chain", "docs_sources", "layman_summary",
                        "chat_history", "qa_history", "flow_diagram", "story_data", "pr_analysis"]:
                st.session_state.pop(key, None)
            st.rerun()

# =======================
# TAB 2: Test Case Generator
# =======================
with main_tab2:
    st.header("🧪 AI-Powered Test Case Generator")
    st.markdown(f"Generate comprehensive test cases from **{source_type}** using AI (30–100 cases per item).")

    if not GROQ_CLIENT_AVAILABLE:
        st.error("⚠️ Groq client not installed. Run: `pip install groq`")
    if pd is None or openpyxl is None:
        st.error("⚠️ Excel libraries not installed. Run: `pip install pandas openpyxl`")

    # Input mode options based on source type
    if source_type == "Azure DevOps":
        test_mode_options = ["Story IDs (comma-separated)", "WIQL Query"]
    elif source_type == "Jira":
        test_mode_options = ["Issue Keys (comma-separated)", "JQL Query"]
    else:
        test_mode_options = ["Page IDs (comma-separated)", "CQL Query"]

    test_mode = st.radio("Input mode for test case generation:", test_mode_options, key="test_mode")

    col1, col2 = st.columns([3, 1])
    with col1:
        if test_mode in ("Story IDs (comma-separated)", "Issue Keys (comma-separated)", "Page IDs (comma-separated)"):
            placeholder_map2 = {
                "Story IDs (comma-separated)": "e.g., 123, 456, 789",
                "Issue Keys (comma-separated)": "e.g., PROJ-123, PROJ-456",
                "Page IDs (comma-separated)": "e.g., 123456, 789012",
            }
            test_ids_input = st.text_input("Enter IDs / keys", placeholder=placeholder_map2[test_mode], key="test_ids_input")
        else:
            placeholder_map3 = {
                "WIQL Query": "SELECT [System.Id] FROM WorkItems WHERE ...",
                "JQL Query": "project = PROJ AND issuetype = Story ORDER BY created DESC",
                "CQL Query": 'space = "ENG" AND type = page ORDER BY created DESC',
            }
            test_query_input = st.text_area(f"Enter {test_mode}", placeholder=placeholder_map3[test_mode], key="test_query_input")

    with col2:
        st.markdown("### Settings")
        test_model = st.selectbox("AI Model", ["llama-3.3-70b-versatile", "openai/gpt-oss-20b", "groq/compound"], key="test_model")
        generate_test_btn = st.button("🚀 Generate Test Cases", type="primary")

    test_status = st.empty()

    if generate_test_btn:
        if not st.session_state.get("groq_key"):
            test_status.error("❌ GROQ_API_KEY not set. Check your .env file.")
        elif not GROQ_CLIENT_AVAILABLE or pd is None or openpyxl is None:
            test_status.error("❌ Required libraries not installed.")
        else:
            try:
                test_status.info("🔍 Collecting IDs...")
                groq_key = st.session_state["groq_key"]
                excel_file = None

                if source_type == "Azure DevOps":
                    org_v = st.session_state["org"]
                    proj_v = st.session_state["project"]
                    pat_v = st.session_state["pat"]

                    if test_mode == "Story IDs (comma-separated)":
                        if not test_ids_input.strip():
                            test_status.warning("Please enter at least one story ID.")
                        else:
                            ids_list = [s.strip() for s in re.split(r"[,\s]+", test_ids_input.strip()) if s.strip()]
                            test_status.info(f"⚙️ Generating test cases for {len(ids_list)} stories...")
                            excel_file = generate_test_cases_excel_azure(ids_list, org_v, proj_v, pat_v, groq_key, test_model)
                    else:
                        if not test_query_input.strip():
                            test_status.warning("Please provide a WIQL query.")
                        else:
                            test_status.info("🔍 Running WIQL...")
                            ids_list = azure_run_wiql(org_v, proj_v, test_query_input, pat_v)
                            test_status.info(f"✅ Found {len(ids_list)} work items.")
                            excel_file = generate_test_cases_excel_azure(ids_list, org_v, proj_v, pat_v, groq_key, test_model)

                elif source_type == "Jira":
                    jira_url = st.session_state["jira_base_url"]
                    jira_email = st.session_state["jira_email"]
                    jira_token = st.session_state["jira_token"]
                    jira_ssl = st.session_state["jira_verify_ssl"]

                    if test_mode == "Issue Keys (comma-separated)":
                        if not test_ids_input.strip():
                            test_status.warning("Please enter at least one issue key.")
                        else:
                            keys_list = [s.strip() for s in re.split(r"[,\s]+", test_ids_input.strip()) if s.strip()]
                            test_status.info(f"⚙️ Generating test cases for {len(keys_list)} issues...")
                            excel_file = generate_test_cases_excel_jira(keys_list, jira_url, jira_email, jira_token, groq_key, test_model, jira_ssl)
                    else:
                        if not test_query_input.strip():
                            test_status.warning("Please provide a JQL query.")
                        else:
                            test_status.info("🔍 Running JQL...")
                            keys_list = jira_run_jql(jira_url, jira_email, jira_token, test_query_input, jira_ssl)
                            test_status.info(f"✅ Found {len(keys_list)} issues.")
                            excel_file = generate_test_cases_excel_jira(keys_list, jira_url, jira_email, jira_token, groq_key, test_model, jira_ssl)

                else:  # Confluence
                    conf_url = st.session_state["conf_base_url"]
                    conf_email = st.session_state["conf_email"]
                    conf_token = st.session_state["conf_token"]
                    conf_ssl = st.session_state["conf_verify_ssl"]

                    if test_mode == "Page IDs (comma-separated)":
                        if not test_ids_input.strip():
                            test_status.warning("Please enter at least one page ID.")
                        else:
                            pids_list = [s.strip() for s in re.split(r"[,\s]+", test_ids_input.strip()) if s.strip()]
                            test_status.info(f"⚙️ Generating test cases for {len(pids_list)} pages...")
                            excel_file = generate_test_cases_excel_confluence(pids_list, conf_url, conf_email, conf_token, groq_key, test_model, conf_ssl)
                    else:
                        if not test_query_input.strip():
                            test_status.warning("Please provide a CQL query.")
                        else:
                            test_status.info("🔍 Running CQL...")
                            pids_list = confluence_search_cql(conf_url, conf_email, conf_token, test_query_input, conf_ssl)
                            test_status.info(f"✅ Found {len(pids_list)} pages.")
                            excel_file = generate_test_cases_excel_confluence(pids_list, conf_url, conf_email, conf_token, groq_key, test_model, conf_ssl)

                if excel_file:
                    st.session_state["test_cases_excel"] = excel_file
                    test_status.success("✅ Test cases generated successfully!")

            except Exception as e:
                test_status.error(f"❌ Error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())

    if st.session_state.get("test_cases_excel"):
        st.markdown("---")
        st.subheader("📥 Download Test Cases")
        col1, col2 = st.columns([2, 1])
        with col1:
            st.download_button(
                label="💾 Download Excel File",
                data=st.session_state["test_cases_excel"],
                file_name=f"test_cases_{time.strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_test_cases",
            )
        with col2:
            if st.button("🗑️ Clear Results", key="clear_test_results"):
                st.session_state["test_cases_excel"] = None
                st.rerun()
        st.info("ℹ️ Each item has its own sheet. Test steps are formatted with line breaks.")

    st.markdown("---")
    with st.expander("ℹ️ How to use Test Case Generator"):
        st.markdown(f"""
### Instructions

1. **Select Data Source** in the sidebar: Azure DevOps, Jira, or Confluence
2. **Configure credentials** for the selected source
3. **Ensure GROQ_API_KEY** is set in your .env file
4. **Choose input mode** (IDs/Keys or Query language)
5. **Click "Generate Test Cases"** and wait for processing
6. **Download the Excel file**

### Supported Sources

| Source | Input | Query Language |
|--------|-------|---------------|
| Azure DevOps | Work Item IDs | WIQL |
| Jira | Issue Keys (e.g. PROJ-123) | JQL |
| Confluence | Page IDs | CQL |

### What you get (30–100 test cases per item)
- ✅ Positive & negative scenarios
- 🔄 Edge cases and boundary values
- 🔗 Integration scenarios
- 🔐 Security test cases
- ♿ Accessibility checks
- 📱 Compatibility testing
""")
