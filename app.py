"""Streamlit UI for the Infinity Knowledge Mesh project."""

from __future__ import annotations

from collections import Counter
import json
from typing import Dict, List

import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from crawler import crawl
from entities import extract_entities
from graph_builder import add_entities, add_page_context, clear_graph, get_graph
from utils import domain_of, normalize_url

st.set_page_config(
    page_title="Infinity Knowledge Mesh",
    layout="wide",
    page_icon="KM",
)

DEFAULT_URL = "https://en.wikipedia.org/wiki/OpenAI"
EXAMPLE_URLS = {
    "OpenAI (Wikipedia)": "https://en.wikipedia.org/wiki/OpenAI",
    "NASA Artemis update": "https://www.nasa.gov/mission/artemis/",
    "Streamlit components guide": "https://docs.streamlit.io/library/components",
    "BBC technology article": "https://www.bbc.com/news/technology",
}
TYPE_COLORS: Dict[str, str] = {
    "page": "#45b7d1",
    "ORG": "#a155b9",
    "PERSON": "#f0a500",
    "GPE": "#ff6f61",
    "MISC": "#62d2a2",
}


def _inject_styles() -> None:
    """Add a light custom theme for a more polished feel."""
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&display=swap');
            .stApp {background: radial-gradient(circle at 20% 18%, #10182a 0, #0b1020 38%, #070a16 100%); font-family: 'Space Grotesk', sans-serif;}
            h1, h2, h3, h4, h5, h6 {font-family: 'Space Grotesk', sans-serif;}
            .km-hero {
                padding: 1.5rem 1.75rem;
                margin: 0.2rem 0 0.75rem 0;
                border-radius: 18px;
                background: linear-gradient(135deg, rgba(69,183,209,0.28), rgba(118,106,255,0.18));
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 18px 42px rgba(0,0,0,0.30);
            }
            .km-hero h1 {margin-bottom: 0.2rem; color: #f7f9fb;}
            .km-hero p {margin-top: 0; color: #d9e4ff;}
            .km-pill {
                display: inline-flex;
                padding: 0.25rem 0.7rem;
                margin-bottom: 0.5rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.12);
                color: #eaf3ff;
                font-weight: 600;
                letter-spacing: 0.3px;
                text-transform: uppercase;
                font-size: 0.75rem;
            }
            .stTabs [data-baseweb="tab-list"] {gap: 0.5rem;}
            .stTabs [data-baseweb="tab"] {background: rgba(255,255,255,0.05); border-radius: 12px; border: 1px solid rgba(255,255,255,0.04);}
            .stTabs [aria-selected="true"] {background: rgba(118,106,255,0.14); border: 1px solid rgba(118,106,255,0.45);}
            .stMetric {background: rgba(255,255,255,0.05); padding: 0.85rem 0.7rem; border-radius: 12px; border: 1px solid rgba(255,255,255,0.07);}
            .stMetric > div:nth-child(1) {color: #dbe8ff;}
            .small-note {color: #9fb2d8; font-size: 0.9rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _hero() -> None:
    st.markdown(
        """
        <div class="km-hero">
            <div class="km-pill">Live Web &rarr; NLP &rarr; Graph</div>
            <h1>Infinity Knowledge Mesh</h1>
            <p>Feed any URL, extract the entities it talks about, and explore how the page connects to people, places, and links.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    _inject_styles()
    _hero()
    st.caption("Crawl a live page, extract named entities, and explore the resulting graph.")

    if "results" not in st.session_state:
        st.session_state["results"] = None  # cached tuple of (text, links, entities, include_links, same_domain_only)
    if "error_msg" not in st.session_state:
        st.session_state["error_msg"] = ""

    with st.sidebar:
        st.header("Controls")
        if "target_url" not in st.session_state:
            st.session_state["target_url"] = DEFAULT_URL

        example_choice = st.selectbox(
            "Try a preset",
            options=["Custom"] + list(EXAMPLE_URLS.keys()),
            help="Select to quickly load a known-good page.",
        )
        if example_choice != "Custom":
            preset_url = EXAMPLE_URLS[example_choice]
            if st.session_state["target_url"] != preset_url:
                st.session_state["target_url"] = preset_url

        url = st.text_input("Target URL", key="target_url")
        entity_limit = st.slider("Max entities", min_value=10, max_value=200, value=60, step=10)
        include_links = st.checkbox("Add discovered hyperlinks", value=True)
        same_domain_only = st.checkbox("Keep links on the same domain only", value=False)
        show_text = st.checkbox("Show extracted text preview", value=False)
        layout_choice = st.radio(
            "Graph layout",
            ["Force Atlas", "Barnes-Hut", "Hierarchical"],
            index=0,
            horizontal=True,
            help="Pick a physics layout for the network visualization.",
        )
        submitted = st.button("Build Knowledge Mesh", type="primary", use_container_width=True)
        st.markdown("Try a news article or docs page to see different graphs.")

    should_seed = st.session_state["results"] is None

    if submitted or should_seed:
        with st.spinner("Crawling page and extracting structure..."):
            try:
                text, links, entities = run_pipeline(
                    url=url if submitted else DEFAULT_URL,
                    entity_limit=entity_limit,
                    include_links=include_links,
                    same_domain_only=same_domain_only,
                )
                st.session_state["results"] = (
                    text,
                    links,
                    entities,
                    include_links,
                    same_domain_only,
                )
                st.session_state["error_msg"] = ""
            except Exception as exc:  # pragma: no cover - only triggered in UI
                st.session_state["results"] = None
                st.session_state["error_msg"] = str(exc)

    if st.session_state["error_msg"]:
        st.error(f"Unable to build the knowledge mesh: {st.session_state['error_msg']}")
        return

    cached = st.session_state["results"]
    if not cached:
        st.info("Provide a URL and click the button to begin.")
        return

    text, links, entities, cached_links_flag, cached_same_domain = cached
    st.success("Knowledge mesh ready! Scroll to explore the graph and insights.")

    overview_tab, graph_tab, text_tab = st.tabs(
        ["Overview", "Interactive Graph", "Extracted Text"]
    )

    with overview_tab:
        _render_metrics(
            text,
            entities,
            links,
            same_domain_only=cached_same_domain,
            include_links=cached_links_flag,
        )
        _render_top_entities(entities)
        _render_top_domains(links, cached_same_domain, cached_links_flag)

    with graph_tab:
        _render_graph(layout_choice=layout_choice)

    with text_tab:
        if show_text:
            st.subheader("Extracted Text (first 1,500 characters)")
            st.write(text[:1500] + ("..." if len(text) > 1500 else ""))
        else:
            st.info("Enable 'Show extracted text preview' in the sidebar to view the source text.")


def run_pipeline(url: str, entity_limit: int, include_links: bool, same_domain_only: bool):
    clear_graph()
    text, links = crawl(url, same_domain_only=same_domain_only)
    entities = extract_entities(text)
    if entity_limit:
        entities = entities[:entity_limit]
    add_entities(entities)
    links_for_graph = links if include_links else []
    add_page_context(url, entities, links_for_graph)
    return text, links_for_graph, entities


def _render_metrics(
    text: str,
    entities: List[Dict[str, str]],
    links: List[str],
    same_domain_only: bool,
    include_links: bool,
) -> None:
    graph = get_graph()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Characters", f"{len(text):,}")
    col2.metric("Unique entities", f"{len({ent['text'] for ent in entities}):,}")
    col3.metric("Graph nodes", f"{graph.number_of_nodes():,}")
    col4.metric(
        "Outgoing links",
        f"{len(links):,}",
        help="Links added to the graph after filtering"
        + (" (same domain only)" if same_domain_only else "")
        + (". Link edges disabled via sidebar toggle." if not include_links else ""),
    )
    if not include_links:
        st.caption("Link edges are disabled. Enable them in the sidebar to see outbound domains and link edges.")
    elif same_domain_only:
        st.caption("Outbound links filtered to the same domain as the source page.")


def _render_top_entities(entities: List[Dict[str, str]]) -> None:
    if not entities:
        st.warning("No entities extracted from the supplied URL.")
        return

    counter = Counter(ent["text"] for ent in entities)
    top_rows = []
    for entity, count in counter.most_common(15):
        label = next((ent["label"] for ent in entities if ent["text"] == entity), "MISC")
        top_rows.append({"Entity": entity, "Label": label, "Mentions": count})

    st.subheader("Top Entities")
    st.dataframe(top_rows, hide_index=True, use_container_width=True)


def _render_top_domains(links: List[str], same_domain_only: bool, include_links: bool) -> None:
    if not include_links:
        st.info("Outbound link edges are disabled.")
        return

    if same_domain_only:
        st.info("Outbound links restricted to the same domain.")
        return

    if not links:
        st.info("No outbound links discovered.")
        return

    domains = _link_domain_counts(links)
    if not domains:
        st.info("No outbound domains to display.")
        return

    st.subheader("Top Outbound Domains")
    rows = [{"Domain": domain, "Links": count} for domain, count in domains.most_common(10)]
    st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_graph(*, layout_choice: str = "Force Atlas") -> None:
    graph = get_graph()
    if graph.number_of_nodes() == 0:
        st.info("Graph is empty.")
        return

    base_options = {
        "nodes": {
            "font": {"face": "Space Grotesk", "size": 16},
            "shape": "dot",
            "scaling": {"min": 8, "max": 22},
        },
        "edges": {"color": "#8fb3ff", "smooth": {"type": "dynamic"}},
        "physics": {"stabilization": True},
    }

    hierarchical_options = {
        "layout": {
            "hierarchical": {
                "enabled": True,
                "direction": "UD",
                "sortMethod": "hubsize",
                "levelSeparation": 140,
            }
        },
        "nodes": {
            "font": {"face": "Space Grotesk", "size": 16},
            "shape": "dot",
            "scaling": {"min": 8, "max": 22},
        },
        "edges": {"smooth": False, "color": "#8fb3ff"},
        "physics": {
            "hierarchicalRepulsion": {
                "nodeDistance": 130,
                "springLength": 140,
            },
            "stabilization": True,
        },
    }

    net = Network(
        height="600px",
        width="100%",
        bgcolor="#0e1117",
        font_color="#ffffff",
        directed=True,
    )
    if layout_choice == "Barnes-Hut":
        net.barnes_hut()
        net.set_options(json.dumps(base_options))
    elif layout_choice == "Hierarchical":
        net.set_options(json.dumps(hierarchical_options))
    else:
        net.force_atlas_2based()
        net.set_options(json.dumps(base_options))

    for node, data in graph.nodes(data=True):
        node_type = data.get("type", "MISC")
        color = TYPE_COLORS.get(node_type, "#6c63ff")
        label = node if node_type != "page" else _trim_url_label(node)
        net.add_node(
            node,
            label=label,
            title=f"{node_type}: {node}",
            color=color,
        )

    for src, dst, data in graph.edges(data=True):
        net.add_edge(src, dst, title=data.get("relation", "related"))

    html = net.generate_html()
    components.html(html, height=650, scrolling=True)


def _trim_url_label(url: str) -> str:
    normalized = normalize_url(url)
    if not normalized:
        return url
    return normalized.replace("https://", "").replace("http://", "")


def _link_domain_counts(links: List[str]) -> Counter:
    counts: Counter = Counter()
    for link in links:
        domain = domain_of(link)
        if domain:
            counts[domain] += 1
    return counts


if __name__ == "__main__":
    main()
