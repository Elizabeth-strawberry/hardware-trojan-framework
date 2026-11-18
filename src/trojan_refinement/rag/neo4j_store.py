"""Namespaced netlist storage and bounded, parameterized Cypher path queries."""

import base64
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from urllib import parse, request, error


@dataclass(frozen=True)
class Neo4jConfig:
    http_uri: str = "http://127.0.0.1:7474"
    database: str = "neo4j"
    user: str = "neo4j"
    password_env: str = "NEO4J_PASSWORD"
    timeout_seconds: float = 60

    def __post_init__(self):
        url = parse.urlsplit(self.http_uri)
        local = url.hostname in ("localhost", "127.0.0.1", "::1")
        if (
            url.scheme not in ("http", "https")
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
            or url.path not in ("", "/")
            or (url.scheme == "http" and not local)
        ):
            raise ValueError("Neo4j requires HTTPS or a loopback HTTP endpoint")
        if (
            not re.fullmatch(r"[A-Za-z0-9_.-]+", self.database)
            or not self.password_env.isidentifier()
        ):
            raise ValueError("Invalid Neo4j database or credential variable")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("Neo4j timeout must be positive and finite")

    @classmethod
    def from_environment(cls):
        return cls(
            http_uri=os.getenv("NEO4J_HTTP_URI", "http://127.0.0.1:7474"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            user=os.getenv("NEO4J_USER", "neo4j"),
        )


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Neo4j redirects are not followed")


class Neo4jGraphStore:
    namespace = "trojan_refinement_v1"

    def __init__(self, config=None):
        self.config = config or Neo4jConfig.from_environment()
        password = os.environ.get(self.config.password_env, "")
        if not password:
            raise ValueError("Set " + self.config.password_env)
        self.authorization = (
            "Basic " + base64.b64encode((self.config.user + ":" + password).encode()).decode()
        )
        handlers = [NoRedirect()]
        if parse.urlsplit(self.config.http_uri).hostname in ("localhost", "127.0.0.1", "::1"):
            handlers.append(request.ProxyHandler({}))
        self.opener = request.build_opener(*handlers)

    def transaction(self, statements):
        body = {
            "statements": [
                {"statement": statement, "parameters": parameters}
                for statement, parameters in statements
            ]
        }
        req = request.Request(
            self.config.http_uri.rstrip("/") + "/db/" + self.config.database + "/tx/commit",
            json.dumps(body, allow_nan=False).encode(),
            {"Content-Type": "application/json", "Authorization": self.authorization},
        )
        try:
            with self.opener.open(req, timeout=self.config.timeout_seconds) as response:
                result = json.load(response)
        except (error.URLError, OSError) as exc:
            raise RuntimeError("Neo4j request failed") from exc
        if result.get("errors"):
            raise RuntimeError(
                "Neo4j query failed: "
                + ", ".join(e.get("code", "unknown") for e in result["errors"])
            )
        return [
            [dict(zip(group["columns"], row["row"])) for row in group["data"]]
            for group in result["results"]
        ]

    def query(self, statement, parameters):
        return self.transaction([(statement, parameters)])[0]

    def import_graph(self, graph, graph_id):
        from ..graph_mentor.data import validate_graph

        graph = validate_graph(graph)
        digest = hashlib.sha256(json.dumps(graph, sort_keys=True).encode()).hexdigest()
        key = self.namespace + ":" + hashlib.sha256(graph_id.encode()).hexdigest() + ":" + digest
        nodes = [
            {"id": n["id"], "cell_type": n["cell_type"], "family": n["family"]}
            for n in graph["nodes"]
        ]
        edges = [
            e for e in graph["edges"] if e["target_port"].upper() not in ("CLK", "CK", "CLOCK")
        ]
        self.query(
            "CREATE CONSTRAINT trojan_refinement_graph_key IF NOT EXISTS FOR (g:TrojanRefinementGraph) REQUIRE g.key IS UNIQUE",
            {},
        )
        self.transaction(
            [
                ("MATCH (n:TrojanRefinementCell {graph_key:$key}) DETACH DELETE n", {"key": key}),
                (
                    "MERGE (g:TrojanRefinementGraph {key:$key}) SET g.namespace=$namespace, g.content_sha256=$digest",
                    {"key": key, "namespace": self.namespace, "digest": digest},
                ),
                (
                    "MATCH (g:TrojanRefinementGraph {key:$key}) UNWIND $nodes AS row "
                    "CREATE (n:TrojanRefinementCell {graph_key:$key, id:row.id, cell_type:row.cell_type, family:row.family}) "
                    "CREATE (g)-[:HAS_CELL]->(n)",
                    {"key": key, "nodes": nodes},
                ),
                (
                    "UNWIND $edges AS row MATCH (a:TrojanRefinementCell {graph_key:$key,id:row.source}), "
                    "(b:TrojanRefinementCell {graph_key:$key,id:row.target}) "
                    "CREATE (a)-[:CONNECTS {source_port:row.source_port,target_port:row.target_port,signal:row.signal}]->(b)",
                    {"key": key, "edges": edges},
                ),
            ]
        )
        counts = self.query(
            "MATCH (n:TrojanRefinementCell {graph_key:$key}) "
            "OPTIONAL MATCH (n)-[r:CONNECTS]->() RETURN count(DISTINCT n) AS nodes, count(r) AS edges",
            {"key": key},
        )
        if counts != [{"nodes": len(nodes), "edges": len(edges)}]:
            raise RuntimeError("Neo4j graph import did not preserve the expected nodes and edges")
        return key

    def bounded_paths(
        self, graph_key, *, sources=(), targets=(), target_port=None, max_hops=3, limit=1000
    ):
        if (
            type(max_hops) is not int
            or not 1 <= max_hops <= 32
            or type(limit) is not int
            or not 1 <= limit <= 2000
        ):
            raise ValueError("Path depth must be 1..32 and result limit 1..2000")
        if not sources and not targets:
            raise ValueError("A bounded path query requires source or target anchors")
        query = (
            f"MATCH p=(a:TrojanRefinementCell)-[:CONNECTS*1..{max_hops}]->(b:TrojanRefinementCell) "
            "WHERE all(n IN nodes(p) WHERE n.graph_key=$key) "
            "AND (size($sources)=0 OR a.id IN $sources) AND (size($targets)=0 OR b.id IN $targets) "
            "AND ($target_port IS NULL OR last(relationships(p)).target_port=$target_port) "
            "RETURN [n IN nodes(p) | n.id] AS nodes, "
            "[r IN relationships(p) | {source_port:r.source_port,target_port:r.target_port,signal:r.signal}] AS edges "
            "ORDER BY length(p), [n IN nodes(p) | n.id] LIMIT $limit"
        )
        rows = self.query(
            query,
            {
                "key": graph_key,
                "sources": list(sources),
                "targets": list(targets),
                "target_port": target_port,
                "limit": limit + 1,
            },
        )
        return {"paths": rows[:limit], "truncated": len(rows) > limit, "max_hops": max_hops}

    def delete_graph(self, graph_key):
        if not graph_key.startswith(self.namespace + ":"):
            raise ValueError("Cannot delete a graph outside the application namespace")
        self.query(
            "MATCH (g:TrojanRefinementGraph {key:$key}) OPTIONAL MATCH (g)-[:HAS_CELL]->(n:TrojanRefinementCell) DETACH DELETE n, g",
            {"key": graph_key},
        )
