"""Simple Jira Cloud integration helpers for Story_Summarizer_RAG_Q-A

Provides functions to:
- build Jira issue keys for project prefix BAW-
- fetch issue details (summary + description)
- fetch issue comments

Authentication:
- If `JIRA_EMAIL` is set in environment, the helper will try Basic auth (email:api_token)
- Otherwise it will attempt to use `Authorization: Bearer <JIRA_PAT>` header
  (If your Jira Cloud requires email+token, set `JIRA_EMAIL` in your .env)

Note: Jira Cloud REST API expects issue key like "BAW-18036". The user said the base
URL is like: https://nykmage.atlassian.net/browse/BAW-18036. We accept either the
full URL or a base URL (e.g., https://nykmage.atlassian.net).
"""

from typing import Tuple, List, Optional
import os
import re
import requests
 

def _get_requests_verify():
    """Determine the `verify` parameter for requests.get.

    Priority:
    - If environment variable JIRA_VERIFY is set to a falsey value (0/false/no), return False
    - If environment variable REQUESTS_CA_BUNDLE is set, return its path (requests will use it)
    - Otherwise return True (default certificate verification)
    """
    v = os.getenv("JIRA_VERIFY")
    if v is not None:
        if str(v).lower() in ("0", "false", "no"):
            return False
        # any other non-empty value means keep verification on
    ca_bundle = os.getenv("REQUESTS_CA_BUNDLE")
    if ca_bundle:
        return ca_bundle
    return True


def normalize_base_url(base: str) -> str:
    if not base:
        raise ValueError("Jira base URL is required")
    # If user passed full browse URL https://.../browse/BAW-18036, strip /browse/... part
    base = base.strip()
    m = re.match(r"(https?://[^/]+)", base)
    if m:
        return m.group(1)
    return base.rstrip("/")


def build_jira_issue_key(issue_id: str, prefix: str = "BAW") -> str:
    """Builds issue key like BAW-12345 from numeric or full-key input.

    Examples:
    - "18036" -> "BAW-18036"
    - "BAW-18036" -> "BAW-18036"
    - "baw-18036" -> "BAW-18036"
    """
    if not issue_id:
        raise ValueError("issue_id is required")
    issue_id = issue_id.strip()
    if issue_id.upper().startswith(f"{prefix}-"):
        return issue_id.upper()
    # if input contains non-numeric or has dashes, just join
    return f"{prefix}-{issue_id}"


def _extract_plain_from_jira_description(desc_field) -> str:
    """Try to extract reasonable plain text from Jira description field.

    Jira Cloud description may be a string or a rich content structure (Confluence "storage"/"doc").
    This function attempts a best-effort extraction: if it's a dict with nested content, pull text segments.
    """
    if not desc_field:
        return ""
    # If it's a simple string, return it
    if isinstance(desc_field, str):
        return desc_field
    # If Jira returns the new Atlassian Document Format (ADF) dict, try to extract text
    try:
        # Walk the structure and collect 'text' fields
        texts = []
        def walk(node):
            if isinstance(node, dict):
                if 'text' in node and isinstance(node['text'], str):
                    texts.append(node['text'])
                for v in node.values():
                    walk(v)
            elif isinstance(node, list):
                for item in node:
                    walk(item)
        walk(desc_field)
        return "\n".join([t for t in texts if t])
    except Exception:
        try:
            return str(desc_field)
        except Exception:
            return ""


def jira_fetch_issue(base_url: str, issue_id: str, pat: Optional[str] = None, email: Optional[str] = None) -> Tuple[str, str]:
    """Fetch Jira issue summary and description.

    Args:
        base_url: Jira base URL, e.g., https://nykmage.atlassian.net or full browse URL
        issue_id: numeric id or full key (e.g., "18036" or "BAW-18036")
        pat: API token (if None, will read from env JIRA_PAT)
        email: Jira account email (optional). If provided, Basic auth (email:pat) will be used.

    Returns:
        (summary, description_text)

    Raises:
        RuntimeError on non-200 responses with informative message.
    """
    pat = pat or os.getenv("JIRA_PAT")
    if not pat:
        raise ValueError("JIRA_PAT token is required (pass as argument or set in environment)")

    base = normalize_base_url(base_url)
    key = build_jira_issue_key(issue_id)
    url = f"{base}/rest/api/3/issue/{key}"

    headers = {"Accept": "application/json"}
    auth = None
    # Prefer email+token Basic auth if email provided
    if email or os.getenv("JIRA_EMAIL"):
        user = email or os.getenv("JIRA_EMAIL")
        auth = (user, pat)
    else:
        # Try Bearer token as fallback
        headers["Authorization"] = f"Bearer {pat}"

    verify = _get_requests_verify()
    resp = requests.get(url, headers=headers, auth=auth, verify=verify)
    if resp.status_code == 404:
        raise RuntimeError(f"Jira issue {key} not found at {base}")
    if resp.status_code in (401, 403):
        raise RuntimeError(f"Authentication failed when fetching Jira issue {key}. Check JIRA_PAT and JIRA_EMAIL (if required). Status: {resp.status_code}")
    if resp.status_code != 200:
        raise RuntimeError(f"Failed to fetch Jira issue {key}: {resp.status_code} - {resp.text[:300]}")

    data = resp.json()
    fields = data.get("fields", {})
    summary = fields.get("summary", "")
    description = fields.get("description")
    description_text = _extract_plain_from_jira_description(description)

    return summary or key, description_text


def jira_fetch_comments(base_url: str, issue_id: str, pat: Optional[str] = None, email: Optional[str] = None) -> List[dict]:
    """Fetch comments for a Jira issue.

    Returns a list of dicts: [{"author": str, "text": str, "created": str}]
    """
    pat = pat or os.getenv("JIRA_PAT")
    if not pat:
        raise ValueError("JIRA_PAT token is required (pass as argument or set in environment)")

    base = normalize_base_url(base_url)
    key = build_jira_issue_key(issue_id)
    url = f"{base}/rest/api/3/issue/{key}/comment"

    headers = {"Accept": "application/json"}
    auth = None
    if email or os.getenv("JIRA_EMAIL"):
        user = email or os.getenv("JIRA_EMAIL")
        auth = (user, pat)
    else:
        headers["Authorization"] = f"Bearer {pat}"

    comments = []
    start_at = 0
    max_results = 50

    while True:
        params = {"startAt": start_at, "maxResults": max_results}
        verify = _get_requests_verify()
        resp = requests.get(url, headers=headers, auth=auth, params=params, verify=verify)
        if resp.status_code == 404:
            # No comments or issue not found
            break
        if resp.status_code in (401, 403):
            raise RuntimeError(f"Authentication failed when fetching Jira comments for {key}. Status: {resp.status_code}")
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to fetch Jira comments for {key}: {resp.status_code} - {resp.text[:300]}")

        data = resp.json()
        values = data.get("comments", [])
        for c in values:
            author = c.get("author", {}).get("displayName", "Unknown")
            body = c.get("body")
            # Jira comment body may be ADF or string - best-effort
            text = _extract_plain_from_jira_description(body)
            created = c.get("created")
            comments.append({"author": author, "text": text, "created": created})

        # Pagination
        if data.get("total") is None:
            break
        start_at += len(values)
        if start_at >= data.get("total", 0):
            break

    return comments

def jira_fetch_linked_prs(base_url: str, issue_key: str, pat: str, email: str = None) -> List[dict]:
    """
    Fetch Pull Requests linked to a Jira issue.
    Uses the issue's remotelinks to find GitHub/Bitbucket PRs.
    Returns: list of dicts [{ 'url': str, 'title': str, 'author': str, 'status': str }]
    """
    try:
        base = normalize_base_url(base_url)
        
        # Method 1: Try remotelinks API (more reliable)
        url = f"{base}/rest/api/3/issue/{issue_key}/remotelink"
        
        headers = {"Accept": "application/json"}
        auth = None
        if email or os.getenv("JIRA_EMAIL"):
            user = email or os.getenv("JIRA_EMAIL")
            auth = (user, pat)
        else:
            headers["Authorization"] = f"Bearer {pat}"
        
        resp = requests.get(url, headers=headers, auth=auth, verify=_get_requests_verify())
        
        if resp.status_code == 200:
            data = resp.json()
            prs = []
            
            for link in data:
                obj = link.get("object", {})
                url_val = obj.get("url", "")
                title = obj.get("title", "")
                
                # Check if it's a PR link (GitHub, Bitbucket, GitLab, Azure DevOps)
                if any(pr_host in url_val.lower() for pr_host in ["github.com/", "bitbucket.org/", "gitlab.com/", "dev.azure.com/"]):
                    # Extract status from the link summary or icon
                    status_obj = obj.get("status", {})
                    status = status_obj.get("resolved", False)
                    status_text = "Merged" if status else "Open"
                    
                    # Try to get icon which sometimes indicates status
                    icon = status_obj.get("icon", {})
                    if icon:
                        icon_title = icon.get("title", "").lower()
                        if "merged" in icon_title or "closed" in icon_title:
                            status_text = "Merged"
                        elif "open" in icon_title:
                            status_text = "Open"
                    
                    prs.append({
                        "url": url_val,
                        "title": title or url_val.split("/")[-1],  # Use last part of URL as fallback
                        "author": "Unknown",  # Remote links don't always have author info
                        "status": status_text
                    })
            
            if prs:
                return prs
        
        # Method 2: Try dev-status API (fallback, may not work for all Jira instances)
        try:
            dev_url = f"{base}/rest/dev-status/1.0/issue/detail"
            
            # Try different application types
            for app_type in ["stash", "github", "bitbucket", "gitlab"]:
                params = {
                    "issueIdOrKey": issue_key,
                    "applicationType": app_type,
                    "dataType": "pullrequest"
                }
                
                resp = requests.get(dev_url, headers=headers, auth=auth, params=params, verify=_get_requests_verify())
                
                if resp.status_code == 200:
                    data = resp.json()
                    prs = []
                    
                    for detail in data.get("detail", []):
                        for pr in detail.get("pullRequests", []):
                            prs.append({
                                "url": pr.get("url", ""),
                                "title": pr.get("name", ""),
                                "author": pr.get("author", {}).get("name", "Unknown"),
                                "status": pr.get("status", "Unknown")
                            })
                    
                    if prs:
                        return prs
        except Exception:
            pass  # dev-status API not available, continue
        
        # If no PRs found via APIs, return empty list
        return []
        
    except Exception as e:
        # Return empty list with warning instead of raising
        import warnings
        warnings.warn(f"Could not fetch PRs for {issue_key}: {e}")
        return []