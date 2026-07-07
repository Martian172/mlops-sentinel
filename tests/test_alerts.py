"""Tests for sentinel.core.alerts."""
import pytest
from sentinel.core.alerts import AlertManager, Alert, AlertRule


class TestAlertManager:
    def test_init(self):
        mgr = AlertManager()
        assert mgr is not None

    def test_add_rule(self):
        mgr = AlertManager()
        rule = AlertRule(name="test", metric="accuracy", threshold=0.9, condition="less_than", severity="WARNING")
        mgr.add_rule(rule)
        assert len(mgr.rules) == 1

    def test_create_alert(self):
        alert = Alert(severity="WARNING", message="Test alert", source="test")
        assert alert.severity == "WARNING"
        assert "Test alert" in alert.message

    def test_send_log_alert(self):
        mgr = AlertManager()
        # Should not raise
        mgr.send_alert(Alert(severity="INFO", message="Test", source="test"))
