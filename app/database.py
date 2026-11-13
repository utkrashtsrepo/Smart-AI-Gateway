"""Asynchronous SQLite database persistence for LLMOps request logging and analytics."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import aiosqlite

from app.config import get_settings
from app.schemas import GatewayMetricsResponse, RequestLogEntry


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS request_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    routed_model TEXT NOT NULL,
    route_reason TEXT NOT NULL,
    complexity_score REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    cache_hit INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_message TEXT,
    retries INTEGER NOT NULL DEFAULT 0,
    fallback_used INTEGER NOT NULL DEFAULT 0,
    tags TEXT
);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON request_logs(timestamp);
CREATE INDEX IF NOT EXISTS idx_logs_status ON request_logs(status);
CREATE INDEX IF NOT EXISTS idx_logs_routed_model ON request_logs(routed_model);
"""


class DatabaseManager:
    """Manages SQLite database connections and lifecycle."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_settings().DATABASE_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Initialize database tables and indexes."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(CREATE_TABLE_SQL)
            await db.commit()

    async def log_request(
        self,
        endpoint: str,
        requested_model: str,
        routed_model: str,
        route_reason: str,
        complexity_score: float,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        cost_usd: float,
        cache_hit: bool,
        status: str = "success",
        error_message: Optional[str] = None,
        retries: int = 0,
        fallback_used: bool = False,
        tags: Optional[List[str]] = None,
    ) -> int:
        """Insert a single request log row asynchronously."""
        now = datetime.now(timezone.utc).isoformat()
        total_tokens = input_tokens + output_tokens
        tags_json = json.dumps(tags or [])

        query = """
        INSERT INTO request_logs (
            timestamp, endpoint, requested_model, routed_model, route_reason,
            complexity_score, input_tokens, output_tokens, total_tokens,
            latency_ms, cost_usd, cache_hit, status, error_message,
            retries, fallback_used, tags
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                query,
                (
                    now,
                    endpoint,
                    requested_model,
                    routed_model,
                    route_reason,
                    complexity_score,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    latency_ms,
                    cost_usd,
                    1 if cache_hit else 0,
                    status,
                    error_message,
                    retries,
                    1 if fallback_used else 0,
                    tags_json,
                ),
            )
            await db.commit()
            return cursor.lastrowid or 0

    async def get_metrics(self) -> GatewayMetricsResponse:
        """Calculate high-level LLMOps metrics across all logged requests."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            # Overall summary
            summary_query = """
            SELECT 
                COUNT(*) as total_requests,
                SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful_requests,
                SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) as failed_requests,
                COALESCE(SUM(total_tokens), 0) as total_tokens,
                COALESCE(SUM(cost_usd), 0.0) as total_cost_usd,
                COALESCE(AVG(latency_ms), 0.0) as avg_latency_ms,
                SUM(CASE WHEN cache_hit = 1 THEN 1 ELSE 0 END) as cache_hits
            FROM request_logs;
            """
            cursor = await db.execute(summary_query)
            row = await cursor.fetchone()

            total_requests = row["total_requests"] or 0
            successful_requests = row["successful_requests"] or 0
            failed_requests = row["failed_requests"] or 0
            total_tokens = row["total_tokens"] or 0
            total_cost_usd = round(float(row["total_cost_usd"] or 0.0), 6)
            avg_latency_ms = round(float(row["avg_latency_ms"] or 0.0), 2)
            cache_hits = row["cache_hits"] or 0

            cache_hit_rate = round((cache_hits / total_requests * 100), 2) if total_requests > 0 else 0.0
            error_rate = round((failed_requests / total_requests * 100), 2) if total_requests > 0 else 0.0

            # Route distribution
            routes_query = """
            SELECT routed_model, COUNT(*) as count, SUM(cost_usd) as model_cost
            FROM request_logs
            GROUP BY routed_model;
            """
            cursor = await db.execute(routes_query)
            route_rows = await cursor.fetchall()

            route_distribution: Dict[str, int] = {}
            cost_by_model: Dict[str, float] = {}
            for r in route_rows:
                model = r["routed_model"]
                route_distribution[model] = r["count"]
                cost_by_model[model] = round(float(r["model_cost"] or 0.0), 6)

            return GatewayMetricsResponse(
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                error_rate_pct=error_rate,
                total_tokens=total_tokens,
                total_cost_usd=total_cost_usd,
                avg_latency_ms=avg_latency_ms,
                cache_hits=cache_hits,
                cache_hit_rate_pct=cache_hit_rate,
                route_distribution=route_distribution,
                cost_by_model=cost_by_model,
            )

    async def get_logs(self, limit: int = 50, offset: int = 0) -> List[RequestLogEntry]:
        """Fetch recent request logs."""
        query = """
        SELECT id, timestamp, endpoint, requested_model, routed_model, route_reason,
               complexity_score, input_tokens, output_tokens, total_tokens,
               latency_ms, cost_usd, cache_hit, status, error_message, retries, fallback_used
        FROM request_logs
        ORDER BY id DESC
        LIMIT ? OFFSET ?;
        """
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, (limit, offset))
            rows = await cursor.fetchall()

            entries = []
            for r in rows:
                entries.append(
                    RequestLogEntry(
                        id=r["id"],
                        timestamp=r["timestamp"],
                        endpoint=r["endpoint"],
                        requested_model=r["requested_model"],
                        routed_model=r["routed_model"],
                        route_reason=r["route_reason"],
                        complexity_score=r["complexity_score"],
                        input_tokens=r["input_tokens"],
                        output_tokens=r["output_tokens"],
                        total_tokens=r["total_tokens"],
                        latency_ms=r["latency_ms"],
                        cost_usd=r["cost_usd"],
                        cache_hit=bool(r["cache_hit"]),
                        status=r["status"],
                        error_message=r["error_message"],
                        retries=r["retries"],
                        fallback_used=bool(r["fallback_used"]),
                    )
                )
            return entries


# Default singleton instance
db_manager = DatabaseManager()

