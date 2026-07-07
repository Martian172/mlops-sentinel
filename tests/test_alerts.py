"""Tests for sentinel.core.alerts."""
from datetime import datetime

from sentinel.core.alerts import Alert, AlertManager, AlertRule, AlertSeverity


def make_alert(severity=AlertSeverity.WARNING, message="Test alert"):
    return Alert(
        id="test-1",
        title="Test",
        message=message,
        severity=severity,
        timestamp=datetime.utcnow(),
        model_name="test-model",
    )


class TestAlertManager:
    def test_init(self):
        mgr = AlertManager()
        assert mgr is not None

    def test_add_rule(self):
        mgr = AlertManager()
        rule = AlertRule(
            name="low-accuracy",
            metric="accuracy",
            threshold=0.9,
            comparison="lt",
            severity="WARNING",
        )
        mgr.add_rule(rule)
        assert len(mgr._rules) == 1

    def test_remove_rule(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(name="r1", metric="accuracy", threshold=0.9))
        assert mgr.remove_rule("r1") is True
        assert mgr.remove_rule("does-not-exist") is False

    def test_create_alert(self):
        alert = make_alert()
        assert alert.severity == AlertSeverity.WARNING
        assert "Test alert" in alert.message

    def test_fire_without_channels(self):
        mgr = AlertManager()
        # No channels configured — should not raise, delivers to 0 channels
        delivered = mgr.fire(make_alert(severity=AlertSeverity.INFO))
        assert delivered == 0

    def test_evaluate_rules_triggers(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="low-accuracy",
            metric="accuracy",
            threshold=0.9,
            comparison="lt",
            severity="CRITICAL",
        ))
        fired = mgr.evaluate_rules({"accuracy": 0.5, "model_name": "test-model"})
        assert len(fired) == 1
        assert fired[0].severity == AlertSeverity.CRITICAL

    def test_evaluate_rules_no_trigger(self):
        mgr = AlertManager()
        mgr.add_rule(AlertRule(
            name="low-accuracy",
            metric="accuracy",
            threshold=0.9,
            comparison="lt",
        ))
        fired = mgr.evaluate_rules({"accuracy": 0.95, "model_name": "test-model"})
        assert fired == []
