"""
Storage backends for MLOps Sentinel.

Provides:
- StorageBackend — abstract base class
- PredictionRecord — immutable data record
- InMemoryStorage — thread-safe in-process store (zero config, no persistence)
- SQLiteStorage — SQLAlchemy-backed SQLite store for persistence
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PredictionRecord
# ---------------------------------------------------------------------------


@dataclass
class PredictionRecord:
    """A single logged prediction event."""

    id: str
    model_name: str
    timestamp: datetime
    features: Dict[str, Any]
    prediction: Any
    actual: Optional[Any] = None
    latency_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "features": self.features,
            "prediction": self.prediction,
            "actual": self.actual,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class StorageBackend(ABC):
    """Abstract interface for Sentinel storage backends."""

    @abstractmethod
    def save(self, record: PredictionRecord) -> None:
        """Persist a prediction record."""

    @abstractmethod
    def get(self, record_id: str) -> Optional[PredictionRecord]:
        """Retrieve a record by ID."""

    @abstractmethod
    def update_actual(self, record_id: str, actual: Any) -> bool:
        """Update the actual label for an existing record."""

    @abstractmethod
    def query(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        has_actual: Optional[bool] = None,
    ) -> List[PredictionRecord]:
        """Query records with optional filters."""

    @abstractmethod
    def count(self, model_name: Optional[str] = None) -> int:
        """Return total record count, optionally filtered by model."""

    @abstractmethod
    def clear(self, model_name: Optional[str] = None) -> int:
        """Delete records.  Returns the number deleted."""


# ---------------------------------------------------------------------------
# InMemoryStorage
# ---------------------------------------------------------------------------


class InMemoryStorage(StorageBackend):
    """
    Thread-safe in-process storage using a plain list.

    Suitable for development, testing, and low-volume production use.
    Data is lost when the process exits.

    Parameters
    ----------
    max_records : int
        Maximum number of records to keep (oldest are discarded).
    """

    def __init__(self, max_records: int = 50_000) -> None:
        self._records: List[PredictionRecord] = []
        self._index: Dict[str, int] = {}  # id → list index
        self._lock = threading.RLock()
        self.max_records = max_records

    def save(self, record: PredictionRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._index[record.id] = len(self._records) - 1

            # Evict oldest records if we exceed the cap
            if len(self._records) > self.max_records:
                evicted = self._records.pop(0)
                self._index.pop(evicted.id, None)
                # Rebuild the index after eviction
                self._index = {r.id: i for i, r in enumerate(self._records)}

    def get(self, record_id: str) -> Optional[PredictionRecord]:
        with self._lock:
            idx = self._index.get(record_id)
            if idx is None:
                return None
            return self._records[idx]

    def update_actual(self, record_id: str, actual: Any) -> bool:
        with self._lock:
            idx = self._index.get(record_id)
            if idx is None:
                return False
            rec = self._records[idx]
            # PredictionRecord is a plain dataclass (not frozen), so mutate directly
            rec.actual = actual
            return True

    def query(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        has_actual: Optional[bool] = None,
    ) -> List[PredictionRecord]:
        with self._lock:
            result = list(self._records)

        if model_name is not None:
            result = [r for r in result if r.model_name == model_name]
        if since is not None:
            result = [r for r in result if r.timestamp >= since]
        if until is not None:
            result = [r for r in result if r.timestamp <= until]
        if has_actual is True:
            result = [r for r in result if r.actual is not None]
        elif has_actual is False:
            result = [r for r in result if r.actual is None]

        if limit is not None:
            result = result[-limit:]

        return result

    def count(self, model_name: Optional[str] = None) -> int:
        with self._lock:
            if model_name is None:
                return len(self._records)
            return sum(1 for r in self._records if r.model_name == model_name)

    def clear(self, model_name: Optional[str] = None) -> int:
        with self._lock:
            before = len(self._records)
            if model_name is None:
                self._records.clear()
                self._index.clear()
            else:
                self._records = [r for r in self._records if r.model_name != model_name]
                self._index = {r.id: i for i, r in enumerate(self._records)}
            return before - len(self._records)


# ---------------------------------------------------------------------------
# SQLiteStorage
# ---------------------------------------------------------------------------


class SQLiteStorage(StorageBackend):
    """
    Persistent storage backed by SQLite via SQLAlchemy Core.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file (e.g., ``"sentinel.db"``).
        Use ``":memory:"`` for an in-process DB.
    """

    def __init__(self, db_path: str = "sentinel.db") -> None:
        try:
            from sqlalchemy import (
                Column,
                DateTime,
                Float,
                MetaData,
                String,
                Table,
                Text,
                create_engine,
                insert,
                select,
                update,
                delete,
                func,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SQLiteStorage requires SQLAlchemy. "
                "Install it with: pip install sqlalchemy"
            ) from exc

        self._engine = create_engine(f"sqlite:///{db_path}", echo=False)
        self._metadata = MetaData()

        self._table = Table(
            "predictions",
            self._metadata,
            Column("id", String(64), primary_key=True),
            Column("model_name", String(256), nullable=False, index=True),
            Column("timestamp", DateTime, nullable=False, index=True),
            Column("features_json", Text, nullable=False),
            Column("prediction_json", Text, nullable=False),
            Column("actual_json", Text, nullable=True),
            Column("latency_ms", Float, nullable=True),
            Column("metadata_json", Text, nullable=False, default="{}"),
        )
        self._metadata.create_all(self._engine)
        self._lock = threading.Lock()

        # Store references to SA symbols
        self._insert = insert
        self._select = select
        self._update = update
        self._delete = delete
        self._func = func

        logger.info("SQLiteStorage initialised: %s", db_path)

    def save(self, record: PredictionRecord) -> None:
        stmt = self._insert(self._table).values(
            id=record.id,
            model_name=record.model_name,
            timestamp=record.timestamp,
            features_json=json.dumps(record.features),
            prediction_json=json.dumps(record.prediction),
            actual_json=json.dumps(record.actual) if record.actual is not None else None,
            latency_ms=record.latency_ms,
            metadata_json=json.dumps(record.metadata),
        )
        with self._lock:
            with self._engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()

    def get(self, record_id: str) -> Optional[PredictionRecord]:
        stmt = self._select(self._table).where(self._table.c.id == record_id)
        with self._engine.connect() as conn:
            row = conn.execute(stmt).fetchone()
        return self._row_to_record(row) if row else None

    def update_actual(self, record_id: str, actual: Any) -> bool:
        stmt = (
            self._update(self._table)
            .where(self._table.c.id == record_id)
            .values(actual_json=json.dumps(actual))
        )
        with self._lock:
            with self._engine.connect() as conn:
                result = conn.execute(stmt)
                conn.commit()
        return result.rowcount > 0

    def query(
        self,
        model_name: Optional[str] = None,
        limit: Optional[int] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        has_actual: Optional[bool] = None,
    ) -> List[PredictionRecord]:
        stmt = self._select(self._table).order_by(self._table.c.timestamp.asc())

        if model_name is not None:
            stmt = stmt.where(self._table.c.model_name == model_name)
        if since is not None:
            stmt = stmt.where(self._table.c.timestamp >= since)
        if until is not None:
            stmt = stmt.where(self._table.c.timestamp <= until)
        if has_actual is True:
            stmt = stmt.where(self._table.c.actual_json.isnot(None))
        elif has_actual is False:
            stmt = stmt.where(self._table.c.actual_json.is_(None))
        if limit is not None:
            # Take the most recent N
            stmt = stmt.order_by(self._table.c.timestamp.desc()).limit(limit)

        with self._engine.connect() as conn:
            rows = conn.execute(stmt).fetchall()

        return [self._row_to_record(r) for r in rows]

    def count(self, model_name: Optional[str] = None) -> int:
        stmt = self._select(self._func.count()).select_from(self._table)
        if model_name is not None:
            stmt = stmt.where(self._table.c.model_name == model_name)
        with self._engine.connect() as conn:
            return conn.execute(stmt).scalar() or 0

    def clear(self, model_name: Optional[str] = None) -> int:
        before = self.count(model_name)
        stmt = self._delete(self._table)
        if model_name is not None:
            stmt = stmt.where(self._table.c.model_name == model_name)
        with self._lock:
            with self._engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
        return before

    @staticmethod
    def _row_to_record(row: Any) -> PredictionRecord:
        # SQLite discards timezone info; timestamps are stored as UTC,
        # so restore awareness on the way out for safe comparisons.
        ts = row.timestamp
        if ts is not None and ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return PredictionRecord(
            id=row.id,
            model_name=row.model_name,
            timestamp=ts,
            features=json.loads(row.features_json),
            prediction=json.loads(row.prediction_json),
            actual=json.loads(row.actual_json) if row.actual_json is not None else None,
            latency_ms=row.latency_ms,
            metadata=json.loads(row.metadata_json) if row.metadata_json else {},
        )
