# Story Summarizer & Q&A — Multi-Source

An AI-powered tool that ingests stories and pages from **Azure DevOps**, **Jira**, or **Confluence**, summarizes them, enables Q&A over the content, generates visual flow diagrams, and produces comprehensive Excel test cases — all in a Streamlit UI.

---

## Features

- **Multi-source ingestion** — Azure DevOps work items, Jira issues, or Confluence pages
- **Query language support** — WIQL (Azure DevOps), JQL (Jira), CQL (Confluence)
- **File upload fallback** — PDF, DOCX, TXT, Markdown
- **RAG-based Q&A** — Ask questions over ingested content using FAISS + HuggingFace embeddings
- **AI summarization** — Structured summaries with P0 scenarios, acceptance criteria, and comment insights
- **Flow diagram generation** — Mermaid.js visual diagrams of item relationships
- **Test case generation** — 30–100 AI-generated test cases per item exported to Excel
- **PR analysis** (Azure DevOps only) — Smart testing focus based on linked pull request diffs

---

## Setup

### 1. Create a virtual environment

```bash
conda create -p venv python==3.12 -y
conda activate <your/path>
pip install -r requirements.txt
```

### 2. Create a `.env` file

Create a file named exactly `.env` in the project root and fill in the credentials for the source(s) you plan to use.

```env
# ── Shared ──────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
model=llama-3.3-70b-versatile       # LLM model name passed to Groq

# ── Azure DevOps (if using Azure DevOps source) ─────
AZURE_DEVOPS_ORG=yourorg            # e.g. dpwhotfsonline
AZURE_DEVOPS_PROJECT=YOURPROJECT    # e.g. DTLP
AZURE_DEVOPS_PAT=your_pat           # needs Work Items (Read) permission

# ── Jira (if using Jira source) ─────────────────────
JIRA_BASE_URL=https://yourcompany.atlassian.net
JIRA_EMAIL=you@yourcompany.com
JIRA_PAT=your_jira_api_token        # generate at id.atlassian.com/manage-profile/security/api-tokens
JIRA_PROJECT=PROJ                   # optional default project key
JIRA_VERIFY=true                    # set to false for self-hosted with self-signed certs

# ── Confluence (if using Confluence source) ─────────
CONFLUENCE_BASE_URL=https://yourcompany.atlassian.net
CONFLUENCE_EMAIL=you@yourcompany.com
CONFLUENCE_TOKEN=your_confluence_api_token
CONFLUENCE_SPACE=ENG                # optional default space key
```

> **Tip:** Jira and Confluence share the same Atlassian domain and API token, so `JIRA_BASE_URL` / `JIRA_PAT` are reused for Confluence if the Confluence-specific vars are not set.

### 3. Run the app

```bash
streamlit run main_NCS.py
```

---

## Usage

### Selecting a data source

Use the **Data Source** dropdown at the top of the sidebar to switch between Azure DevOps, Jira, and Confluence. The credential fields and input modes update automatically.

### Tab 1 — Summarizer & Q&A

| Data Source | Input modes | Query language |
|-------------|-------------|---------------|
| Azure DevOps | Work Item IDs, WIQL Query, Upload files | WIQL |
| Jira | Issue Keys (e.g. `PROJ-123`), JQL Query, Upload files | JQL |
| Confluence | Page IDs, CQL Query, Upload files | CQL |

1. Fill in credentials in the sidebar.
2. Choose an input mode and enter IDs / a query, or upload files.
3. Click **Fetch / Embed & Summarize**.
4. View the AI summary, flow diagram, and use the Q&A chat below.

**PR Analysis** (Azure DevOps only): select a story and click *Analyze Linked PRs* to get a smart testing focus report based on the code changes in linked pull requests.

### Tab 2 — Test Case Generator

1. Select the same data source and input mode.
2. Choose an AI model (default: `llama-3.3-70b-versatile`).
3. Click **Generate Test Cases**.
4. Download the Excel file — each item gets its own sheet with 30–100 test cases covering:
   - Positive & negative scenarios
   - Edge cases and boundary values
   - Integration, security, accessibility, and compatibility scenarios

---

## Environment variables reference

| Variable | Required for | Description |
|----------|-------------|-------------|
| `GROQ_API_KEY` | All | Groq API key for LLM inference |
| `HF_TOKEN` | All | HuggingFace token for embedding model |
| `model` | All | Groq model name (default: `llama-3.3-70b-versatile`) |
| `AZURE_DEVOPS_ORG` | Azure DevOps | Organization name or URL |
| `AZURE_DEVOPS_PROJECT` | Azure DevOps | Project name |
| `AZURE_DEVOPS_PAT` | Azure DevOps | Personal Access Token |
| `JIRA_BASE_URL` | Jira | e.g. `https://company.atlassian.net` |
| `JIRA_EMAIL` | Jira | Atlassian account email |
| `JIRA_PAT` | Jira | Atlassian API token |
| `JIRA_PROJECT` | Jira (optional) | Default project key |
| `JIRA_VERIFY` | Jira (optional) | `false` to skip SSL verification |
| `CONFLUENCE_BASE_URL` | Confluence | e.g. `https://company.atlassian.net` |
| `CONFLUENCE_EMAIL` | Confluence | Atlassian account email |
| `CONFLUENCE_TOKEN` | Confluence | Atlassian API token |
| `CONFLUENCE_SPACE` | Confluence (optional) | Default space key |

---

## Dependencies

Key packages (see `requirements.txt` for pinned versions):

- `streamlit` — UI framework
- `langchain`, `langchain-groq`, `langchain-huggingface`, `langchain-community` — LLM orchestration
- `faiss-cpu` — vector store for RAG
- `sentence-transformers` — local embeddings (`all-MiniLM-L6-v2`)
- `groq` — Groq client for test case generation
- `beautifulsoup4` — HTML stripping (Azure DevOps / Confluence content)
- `pandas`, `openpyxl` — Excel export
- `python-docx`, `PyPDF2` — file upload parsing
- `python-dotenv` — `.env` loading
- `requests` — API calls to all three platforms
