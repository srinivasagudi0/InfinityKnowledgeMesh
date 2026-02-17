"""Knowledge graph builder module using NetworkX."""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List

import networkx as nx

from utils import normalize_url

logger = logging.getLogger(__name__)

# Initialize directed graph
graph = nx.DiGraph()

MENTIONS_RELATION = "mentions"
LINKS_RELATION = "links_to"


def add_entities(entities: List[Dict[str, str]]) -> None:
    """
    Add entities as nodes to the knowledge graph.
    
    Args:
        entities: A list of entity dictionaries with 'text' and 'label' keys.
    """
    if not entities:
        logger.warning("No entities to add to the graph")
        return
    
    added = 0
    for ent in entities:
        text = ent.get("text", "").strip()
        label = ent.get("label", "MISC")
        if not text:
            logger.debug("Skipping entity with empty text: %s", ent)
            continue
        graph.add_node(text, type=label)
        added += 1
    
    logger.info("Added %d entities to the graph", added)


def add_relation(src: str, rel: str, dst: str) -> None:
    """
    Add a relationship edge between two nodes in the graph.
    
    Args:
        src: The source node.
        rel: The relationship type.
        dst: The destination node.
    """
    if not src or not dst:
        return
    graph.add_edge(src, dst, relation=rel)
    logger.debug("Added relation: %s --[%s]--> %s", src, rel, dst)


def add_page_context(source_url: str, entities: Iterable[Dict[str, str]], links: Iterable[str]) -> None:
    """
    Enrich the graph with relationships specific to a crawled page.
    
    Args:
        source_url: URL of the crawled page.
        entities: Iterable of entity dictionaries.
        links: Iterable of URLs found on the page.
    """
    normalized_source = normalize_url(source_url)
    if not normalized_source:
        logger.warning("Skipping context build due to invalid source URL: %s", source_url)
        return

    graph.add_node(normalized_source, type="page", role="source")

    for ent in entities:
        text = ent.get("text", "").strip()
        if not text:
            continue
        graph.add_edge(normalized_source, text, relation=MENTIONS_RELATION)

    for link in links:
        normalized_link = normalize_url(link)
        if not normalized_link or normalized_link == normalized_source:
            continue
        graph.add_node(normalized_link, type="page")
        graph.add_edge(normalized_source, normalized_link, relation=LINKS_RELATION)


def get_graph() -> nx.DiGraph:
    """
    Get the current knowledge graph.
    
    Returns:
        The NetworkX directed graph object.
    """
    return graph


def clear_graph() -> None:
    """
    Clear all nodes and edges from the graph.
    """
    graph.clear()
    logger.info("Graph cleared")
