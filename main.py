"""Main entry point for the Infinity Knowledge Mesh application."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from typing import List

from crawler import crawl
from entities import extract_entities
from graph_builder import add_entities, add_page_context, clear_graph, get_graph
from utils import domain_of

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Crawl a URL, extract entities, and summarize the knowledge graph."""
    args = _parse_args()

    try:
        clear_graph()
        logger.info("Starting Infinity Knowledge Mesh")
        text, links = crawl(
            args.url,
            same_domain_only=args.same_domain_only,
        )
        logger.info("Crawled %d characters and %d links", len(text), len(links))

        entities = extract_entities(text)
        if args.max_entities:
            entities = entities[: args.max_entities]
        logger.info("Processed %d entities", len(entities))

        add_entities(entities)
        add_page_context(args.url, entities, links if not args.skip_links else [])

        _print_summary(
            entities,
            links,
            top_n=args.top_entities,
            top_domains=args.top_domains,
            same_domain_only=args.same_domain_only,
        )
    except Exception:
        logger.exception("Application failed")
        sys.exit(1)  # noqa: TRY004


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a lightweight knowledge graph from any URL."
    )
    parser.add_argument(
        "-u",
        "--url",
        default="https://en.wikipedia.org/wiki/OpenAI",
        help="Target URL to crawl (default: %(default)s)",
    )
    parser.add_argument(
        "-m",
        "--max-entities",
        type=int,
        default=50,
        help="Limit the number of entities to include in the graph.",
    )
    parser.add_argument(
        "--skip-links",
        action="store_true",
        help="Skip linking out to other pages discovered during crawling.",
    )
    parser.add_argument(
        "--same-domain-only",
        action="store_true",
        help="Keep only outbound links on the same domain as the crawled page.",
    )
    parser.add_argument(
        "-t",
        "--top-entities",
        type=int,
        default=10,
        help="How many of the most frequent entities to display in the summary.",
    )
    parser.add_argument(
        "--top-domains",
        type=int,
        default=5,
        help="How many outbound link domains to list in the summary.",
    )
    return parser.parse_args()


def _print_summary(
    entities: List[dict],
    links: List[str],
    *,
    top_n: int,
    top_domains: int,
    same_domain_only: bool,
) -> None:
    G = get_graph()
    print("\nKnowledge Graph Statistics")
    print("--------------------------")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")
    print(f"Outgoing links retained: {len(links)}{' (same domain only)' if same_domain_only else ''}")
    if entities:
        counter = Counter(ent["text"] for ent in entities)
        print("\nTop entities")
        for entity, count in counter.most_common(top_n):
            print(f"  {entity}: {count}")
    if links and not same_domain_only:
        domains = [domain_of(link) for link in links]
        domain_counts = Counter(domain for domain in domains if domain)
        if domain_counts:
            print("\nTop outbound link domains")
            for domain, count in domain_counts.most_common(top_domains):
                print(f"  {domain}: {count}")


if __name__ == "__main__":
    main()
