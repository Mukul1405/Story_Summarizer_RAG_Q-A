#!/usr/bin/env python3
"""
Small connectivity tester for the Jira helpers.
Usage:
  python run_jira_connectivity.py [ISSUE_ID] [--no-verify]

Reads JIRA_BASE_URL, JIRA_PAT, JIRA_EMAIL, REQUESTS_CA_BUNDLE and JIRA_VERIFY from environment (.env supported).
If ISSUE_ID omitted, defaults to 'BAW-17958' (example)
"""
import os
import sys
import argparse
from dotenv import load_dotenv

load_dotenv()

try:
    from jira_integration import jira_fetch_issue, build_jira_issue_key, jira_fetch_linked_prs
except Exception as e:
    print("Failed to import jira_integration module:", e)
    sys.exit(2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("issue", nargs="?", default=os.getenv("TEST_JIRA_ISSUE", "BAW-17958"), help="Issue id or key to fetch")
    p.add_argument("--no-verify", action="store_true", help="Run requests with verify=False (insecure, dev only)")
    args = p.parse_args()

    base = os.getenv("JIRA_BASE_URL")
    pat = os.getenv("JIRA_PAT")
    email = os.getenv("JIRA_EMAIL")

    if args.no_verify:
        os.environ["JIRA_VERIFY"] = "false"

    if not base or not pat:
        print("Please set JIRA_BASE_URL and JIRA_PAT in your environment or .env file before running this script.")
        sys.exit(1)

    issue = args.issue
    print(f"Testing Jira connectivity to: {base} (issue={issue})")
    try:
        summary, description = jira_fetch_issue(base, issue, pat=pat, email=email)
        print("Fetch successful:")
        print("SUMMARY:", summary)
        print("DESCRIPTION (first 300 chars):\n", (description or "(empty)")[:300])
        prs = jira_fetch_linked_prs(base, issue, pat=pat, email=email)
        if prs:
            print("Linked PRs:")
            for pr in prs:
                print(f"- {pr['status']}: {pr['title']} ({pr['url']}) by {pr['author']}")
        else:
            print("No PRs linked to this Jira issue.")
        sys.exit(0)
    except Exception as e:
        print("Failed to fetch", issue, ":", repr(e))
        # Helpful env hints
        print("\nHints:")
        print(" - If you are on a corporate network doing TLS interception, set REQUESTS_CA_BUNDLE to a .pem file that contains your company CA and restart Streamlit/script.")
        print(" - For a quick insecure test, re-run with --no-verify (not recommended for production)")
        sys.exit(3)
    


if __name__ == '__main__':
    main()