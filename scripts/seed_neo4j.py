"""Seeding utility for Neo4j graph database.

Creates the default facility topology nodes (Zones, Equipment, Sensors, Connections)
required for the Hazard Propagation and Digital Twin modules to function.
"""

import asyncio
import logging

from app.core.settings import get_settings
from app.hazard_propagation.repositories.neo4j_graph_repo import (
    Neo4jGraphRepository,
)
from tests.integration.test_neo4j_hazard_propagation import seed_graph

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("seed_neo4j")


async def main():
    settings = get_settings()
    logger.info("Connecting to Neo4j at %s...", settings.neo4j_uri)

    repo = Neo4jGraphRepository(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
    )

    try:
        # Clean existing data
        logger.info("Cleaning up existing Neo4j data...")
        await repo._run("MATCH (n) DETACH DELETE n")

        # Seed graph
        logger.info("Seeding default facility graph...")
        counts = await seed_graph(repo)
        logger.info("✓ Successfully seeded graph: %s", counts)

    except Exception as e:
        logger.error("Failed to seed Neo4j: %s", e)
    finally:
        await repo.close()
        logger.info("Connection closed")


if __name__ == "__main__":
    asyncio.run(main())
