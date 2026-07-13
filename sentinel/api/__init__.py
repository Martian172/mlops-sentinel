"""Multi-tenant monitoring API for MLOps Sentinel.

Exposes a versioned REST API (``/api/v1``) that lets anyone register a model,
stream predictions to it over HTTP, and pull drift / performance / alert
reports back — monitoring-as-a-service, no Python import required on the
client side.
"""
from sentinel.api.registry import MonitorRegistry, registry
from sentinel.api.routes import router

__all__ = ["MonitorRegistry", "registry", "router"]
