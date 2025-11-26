# ========================================================================
#  Agentic Healthcare Assistant
#  UPDATED: PubMed + NCBI + WHO Live Search Integration
# ========================================================================

import os
import uuid
from datetime import datetime, date
from typing import List, Dict, Any, Optional
import time

import streamlit as st
import requests
import numpy as np
import pandas as pd

from pypdf import PdfReader
from docx import Document

try:
    import openai
except ImportError:
    openai = None


# ========================================================================
# >>> NEW: PubMed + WHO Integration (Place directly after imports)
# ========================================================================

NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
WHO_BASE = "https://ghoapi.azureedge.net/api/"

def pubmed_search(query, max_results=5):
    """Search PubMed using NCBI ESearch + ESummary."""
    try:
        # Step 1: Search PMIDs
        esearch_url = NCBI_BASE + "esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json"
        }
        r = requests.get(esearch_url, params=params)
        data = r.json()
        pmids = data.get("esearchresult", {}).get("idlist", [])

        if not pmids:
            return []

        # Step 2: Fetch Details
        esummary_url = NCBI_BASE + "esummary.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "json"
        }
        r = requests.get(esummary_url, params=params)
        summaries = r.json().get("result", {})

        results = []
        for pmid in pmids:
            info = summaries.get(pmid, {})
            if info:
                results.append({
                    "pmid": pmid,
                    "title": info.get("title"),
                    "authors": [a.get("name") for a in info.get("authors", [])],
                    "journal": info.get("fulljournalname"),
                    "pubdate": info.get("pubdate"),
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                })
        return results

    except Exception as e:
        return [{"error": str(e)}]


def who_search(indicator):
    """Retrieve data from WHO Global Health Observatory."""
    try:
        r = requests.get(WHO_BASE + indicator)
        if r.status_code != 200:
            return {"error": "Invalid WHO indicator or endpoint."}
        return r.json().get("value", [])
    except Exception as e:
        return {"error": str(e)}

# <<< END NEW CODE
# ========================================================================


# ========================================================================
# (Your original code continues below — unchanged)
# CONFIG & UTILITIES
# ========================================================================

st.set_page_config(
    page_title="Agentic Healthcare Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
EVAL_MODEL_NAME = os.getenv("OPENAI_EVAL_MODEL_NAME", MODEL_NAME)
EMBED_MODEL_NAME = os.getenv("OPENAI_EMBED_MODEL_NAME", "text-embedding-3-small")

...
# (ALL your original functions remain EXACTLY the same)
...
# No deletions, no changes to existing logic
...


# ========================================================================
# >>> NEW TABS ADDED HERE (Inside main())
# ========================================================================

def main():
    st.title("🏥 Agentic Healthcare Assistant")

    # (Your original sidebar code is untouched)

    # ============================
    # Updated Tab Definitions
    # ============================
    tab_chat, tab_appts, tab_files, tab_pubmed, tab_who, tab_metrics, tab_debug = st.tabs(
        [
            "💬 Chat Assistant",
            "📅 Appointments",
            "📂 Patient Files & Search",
            "🔎 PubMed Search",
            "🌍 WHO Search",
            "📊 Metrics & Logs",
            "🛠 Debug Trace",
        ]
    )

    # Existing tabs render unchanged:
    with tab_chat:
        render_chat_tab()

    with tab_appts:
        render_appointments_tab()

    with tab_files:
        render_files_tab()

    # ===================================================================
    # PubMed UI (NEW)
    # ===================================================================
    with tab_pubmed:
        st.subheader("🔎 Live PubMed / MEDLINE Search")

        query = st.text_input("Enter a medical keyword or topic:")

        if st.button("Search PubMed"):
            results = pubmed_search(query)

            if not results:
                st.warning("No PubMed results found.")
            else:
                for item in results:
                    st.markdown(f"### {item['title']}")
                    st.write(f"**Journal:** {item['journal']}")
                    st.write(f"**Date:** {item['pubdate']}")
                    st.write(f"**Authors:** {', '.join(item['authors'])}")
                    st.write(f"[View on PubMed →]({item['url']})")
                    st.markdown("---")

    # ===================================================================
    # WHO UI (NEW)
    # ===================================================================
    with tab_who:
        st.subheader("🌍 WHO Global Health Observatory Search")

        indicator = st.text_input("Enter WHO Indicator Code (Example: WHOSIS_000015)")

        if st.button("Search WHO"):
            results = who_search(indicator)
            st.json(results)

    # Existing tabs continue unchanged
    with tab_metrics:
        render_metrics_tab()

    with tab_debug:
        render_debug_tab()


if __name__ == "__main__":
    main()


