# Story Summarizer & QA Assistant — Multi-Source

> **Stop reading 50-page backlogs. Get AI summaries, instant Q&A, and ready-to-use test cases — in under a minute.**

Works with **Azure DevOps**, **Jira**, and **Confluence**. No copy-paste. No manual effort.

---

## The Problem

Every sprint, QA engineers and BAs waste hours doing the same things:

- Reading through long, scattered user stories and acceptance criteria
- Digging through comments to find key decisions buried in threads
- Writing test cases from scratch for stories they barely understand
- Asking devs "what does this story actually mean?" — repeatedly

This is not the job. This is the overhead before the job.

---

## The Solution

Paste your story IDs (or run a query). In under 60 seconds you get:

- A clean, structured **AI summary** of every story
- **P0 scenarios** and acceptance criteria pulled out automatically
- A **visual flow diagram** showing how stories relate
- A **Q&A chatbot** you can ask anything — "what's the edge case here?" "what did the dev say about the auth flow?"
- A downloadable **Excel sheet with 30–100 test cases** per story — covering positive, negative, edge, security, accessibility, and integration scenarios

---

## Demo

> **Add screenshots here before sharing with buyers.**
>
> Recommended shots:
> 1. Sidebar with source selector (Azure DevOps / Jira / Confluence)
> 2. The AI summary output for a real story
> 3. The Q&A chat answering a question
> 4. The generated Excel test cases
>
> Run the app (`streamlit run main_NCS.py`), open `http://localhost:8501`, and screenshot with your tool.
> Drop images into a `/docs/screenshots/` folder and replace this section with:
> `![Summary View](docs/screenshots/summary.png)`

---

## Key Features

### Multi-Source Ingestion
Connect directly to **Azure DevOps**, **Jira**, or **Confluence** — no export, no copy-paste. Fetch stories by ID, run a saved query (WIQL / JQL / CQL), or upload files (PDF, DOCX, TXT).

### AI Summarization That Actually Makes Sense
Not a word-for-word dump. The AI extracts what matters: the objective, the P0 scenarios, the acceptance criteria, and insights from dev/QA comments — formatted cleanly in Markdown.

### Ask Anything — Contextual Q&A
Once stories are loaded, ask questions in plain English. "What's the rollback plan?" "Which stories touch the payment module?" "What edge cases did the team discuss?" The chatbot searches across all ingested content.

### Flow Diagram Generation
Automatically generates a Mermaid.js diagram showing how your stories connect and flow — useful for sprint planning, demo prep, and onboarding.

### Test Case Generator (Excel Export)
Select stories, click generate. The AI produces 30–100 structured test cases per story covering:
- Positive & negative flows
- Boundary and edge cases
- Integration scenarios
- Security, accessibility, and compatibility checks

Each story gets its own Excel sheet, formatted with test steps as numbered lines.

### PR Impact Analysis *(Azure DevOps)*
Fetches linked pull requests and analyzes changed files to surface high-risk areas and suggest testing priorities — without sending source code to any external AI.

---

## Who Is This For?

| Role | How they use it |
|------|----------------|
| **QA Engineers** | Generate test cases, understand stories fast, find edge cases |
| **Business Analysts** | Summarize backlogs, prep for sprint reviews, answer stakeholder questions |
| **Test Managers** | Get sprint-level test coverage in minutes, export to Excel |
| **Developers** | Quickly understand what a story expects without reading the whole thread |

---

## How It Works

```
1. Select source  →  Azure DevOps / Jira / Confluence
2. Enter credentials + story IDs (or a query)
3. Click "Fetch / Embed & Summarize"
```

That's it. Summarization, diagram, and Q&A are ready instantly.

---

## Setup

### 1. Create a virtual environment

```bash
conda create -p venv python==3.12 -y
conda activate <your/path>
pip install -r requirements.txt
```

### 2. Create a `.env` file

```env
# ── Shared ──────────────────────────────────────────
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
model=llama-3.3-70b-versatile

# ── Azure DevOps ─────────────────────────────────────
AZURE_DEVOPS_ORG=yourorg
AZURE_DEVOPS_PROJECT=YOURPROJECT
AZURE_DEVOPS_PAT=your_pat

# ── Jira ─────────────────────────────────────────────
JIRA_BASE_URL=https://yourcompany.atlassian.net
JIRA_EMAIL=you@yourcompany.com
JIRA_PAT=your_jira_api_token
JIRA_VERIFY=true

# ── Confluence ───────────────────────────────────────
CONFLUENCE_BASE_URL=https://yourcompany.atlassian.net
CONFLUENCE_EMAIL=you@yourcompany.com
CONFLUENCE_TOKEN=your_confluence_api_token
```

> Jira and Confluence share the same Atlassian domain and API token.

### 3. Run

```bash
streamlit run main_NCS.py
```

Open **http://localhost:8501**

---

## Supported Sources & Query Languages

| Source | Fetch by | Query language | Example |
|--------|----------|---------------|---------|
| Azure DevOps | Work Item IDs | WIQL | `SELECT [System.Id] FROM WorkItems WHERE ...` |
| Jira | Issue Keys | JQL | `project = PROJ AND issuetype = Story` |
| Confluence | Page IDs | CQL | `space = "ENG" AND type = page` |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| LLM | Groq (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API cost) |
| Vector Store | FAISS (in-memory) |
| Excel Export | pandas + openpyxl |
| Source APIs | Azure DevOps REST API, Jira REST API v3, Confluence REST API |

---

## Environment Variables

| Variable | Source | Required |
|----------|--------|----------|
| `GROQ_API_KEY` | All | Yes |
| `HF_TOKEN` | All | Yes |
| `model` | All | No (default: `llama-3.3-70b-versatile`) |
| `AZURE_DEVOPS_ORG` | Azure DevOps | Yes |
| `AZURE_DEVOPS_PROJECT` | Azure DevOps | Yes |
| `AZURE_DEVOPS_PAT` | Azure DevOps | Yes |
| `JIRA_BASE_URL` | Jira | Yes |
| `JIRA_EMAIL` | Jira | Yes |
| `JIRA_PAT` | Jira | Yes |
| `JIRA_VERIFY` | Jira | No (default: `true`) |
| `CONFLUENCE_BASE_URL` | Confluence | Yes |
| `CONFLUENCE_EMAIL` | Confluence | Yes |
| `CONFLUENCE_TOKEN` | Confluence | Yes |
