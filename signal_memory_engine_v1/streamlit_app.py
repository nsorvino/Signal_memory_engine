import os
from urllib.parse import urljoin

import matplotlib.pyplot as plt
import requests
import streamlit as st

from dashboard import show_dashboard

# Basic Streamlit UI for Signal Memory RAG backend with drift visualization

# Basic Streamlit UI for Signal Memory RAG backend with drift visualization

# ---- Backend reachability helpers -------------------------------------------

PROBE_PATHS = ("/healthz", "/health", "/docs", "/")  # try a few common endpoints
CANDIDATE_URLS = (
    os.getenv("BACKEND_URL"),  # if provided via environment/compose
    "http://backend:8000",  # works from inside Docker
    "http://localhost:8000",  # works from host
)


def backend_is_running(base_url: str, timeout: float = 0.75) -> bool:
    """Return True if any probe path responds (2xx or 4xx is fine)."""
    if not base_url:
        return False
    for p in PROBE_PATHS:
        try:
            r = requests.get(urljoin(base_url.rstrip("/") + "/", p.lstrip("/")), timeout=timeout)
            if 200 <= r.status_code < 500:
                return True
        except requests.RequestException:
            continue
    return False


def pick_backend() -> str:
    """Pick the first reachable candidate URL; otherwise fall back to localhost."""
    for url in CANDIDATE_URLS:
        if url and backend_is_running(url):
            return url
    # usable fallback so user can edit it
    for url in CANDIDATE_URLS:
        if url:
            return url
    return "http://localhost:8000"


# ---- Plotting / UI helpers ---------------------------------------------------


def plot_drift(scores: dict):
    """
    Create a radial bar chart showing each agent's top similarity score using matplotlib.
    """
    labels = list(scores.keys())
    values = list(scores.values())
    N = len(values)

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")

    angles = [n / float(N) * 2 * 3.14159265 for n in range(N)]
    width = 2 * 3.14159265 / N

    ax.bar(angles, values, width=width)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_title("Agent Drift Scores", pad=20)
    plt.tight_layout()
    return fig


def display_flag(flag: str, suggestion: str):
    """
    Show flag with color coding and suggestion.
    """
    if flag == "stable":
        st.success(f"Flag: {flag}")
    elif flag == "drifting":
        st.warning(f"Flag: {flag}")
    else:  # concern
        st.error(f"Flag: {flag}")
    st.write(f"Suggestion: {suggestion}")


def main():
    page = st.sidebar.radio("Page", ["Interface", "Dashboard"])
    if page == "Dashboard":
        show_dashboard()
        return
    st.title("Signal Memory RAG Interface")
    st.sidebar.header("Settings")

    # Detect once at startup
    initial_url = pick_backend()
    backend_url = st.sidebar.text_input("Backend URL", initial_url, key="backend_url")

    mode = st.sidebar.selectbox("Mode", ["Single-Agent", "Multi-Agent"])
    k = st.sidebar.slider("Number of chunks (k)", min_value=1, max_value=10, value=3)

    st.header("Enter your query")
    query = st.text_area("Enter your question:", height=100)

    if st.button("Submit"):
        if not query.strip():
            st.warning("Please enter a query.")
            return

        endpoint = "/query" if mode == "Single-Agent" else "/multi_query"
        url = f"{backend_url}{endpoint}"
        payload = {"query": query, "k": k}

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            st.error(f"Request failed: {e}")
            return

        # Single-Agent
        if mode == "Single-Agent":
            st.subheader("Answer")
            st.write(data.get("answer", ""))

            st.subheader("Memory Chunks & Scores")
            for chunk in data.get("chunks", []):
                st.markdown(f"- **{chunk['score']:.3f}**: {chunk['content']}")

            st.subheader("Flag & Suggestion")
            display_flag(data.get("flag", ""), data.get("suggestion", ""))

            # Drift Gauge
            top_score = max((c["score"] for c in data.get("chunks", [])), default=0.0)
            st.subheader("Drift Gauge")
            fig = plot_drift({mode: top_score})
            st.pyplot(fig)

        # Multi-Agent
        else:
            st.subheader("Multi-Agent Responses")
            agents = data.get("agents", {})
            drift_scores = {}

            for role, result in agents.items():
                st.markdown(f"### {role}")

                st.markdown("**Answer:**")
                st.write(result.get("answer", ""))

                st.markdown("**Chunks & Scores:**")
                for chunk in result.get("chunks", []):
                    st.markdown(f"- **{chunk['score']:.3f}**: {chunk['content']}")

                st.markdown("**Flag & Suggestion:**")
                display_flag(result.get("flag", ""), result.get("suggestion", ""))

                # collect for drift chart
                top = max((c["score"] for c in result.get("chunks", [])), default=0.0)
                drift_scores[role] = top

            st.subheader("Drift Visualization")
            fig = plot_drift(drift_scores)
            st.pyplot(fig)


if __name__ == "__main__":
    main()
