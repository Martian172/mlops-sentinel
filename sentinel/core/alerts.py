"""
Alerting system for MLOps Sentinel.

Provides:
- AlertManager — orchestrates alert evaluation and dispatch
- SlackAlertChannel — sends alerts to Slack via webhooks
- EmailAlertChannel — sends alerts via SMTP
- WebhookAlertChannel — generic HTTP webhook
- AlertRule — configurable threshold-based rule
- Alert — immutable alert event dataclass
"""

from __future__ import annotations

import asyncio
import json
import logging
import smtplib
import socket
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class AlertSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Alert:
    """An immutable alert event."""

    id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    model_name: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------


@dataclass
class AlertRule:
    """
    A threshold-based alert rule.

    Parameters
    ----------
    name : str
        Unique rule identifier.
    metric : str
        Key in the metrics dict to check (e.g., ``"accuracy"``, ``"drift_score"``).
    threshold : float
        Threshold value.
    severity : str or AlertSeverity
        Alert severity when rule fires.
    comparison : str
        ``"gt"`` (greater than), ``"lt"`` (less than), ``"gte"``, ``"lte"``, ``"eq"``.
    cooldown_seconds : int
        Minimum seconds between repeated alerts for the same rule.
    enabled : bool
        Whether the rule is active.
    """

    name: str
    metric: str
    threshold: float
    severity: str = "WARNING"
    comparison: str = "gt"
    cooldown_seconds: int = 300
    enabled: bool = True
    _last_fired: Optional[datetime] = field(default=None, repr=False, compare=False)

    def evaluate(self, value: float) -> bool:
        """Return ``True`` if ``value`` violates the threshold."""
        ops: Dict[str, Callable[[float, float], bool]] = {
            "gt": lambda v, t: v > t,
            "lt": lambda v, t: v < t,
            "gte": lambda v, t: v >= t,
            "lte": lambda v, t: v <= t,
            "eq": lambda v, t: v == t,
        }
        op = ops.get(self.comparison, ops["gt"])
        return op(value, self.threshold)

    def can_fire(self) -> bool:
        """Return ``True`` if the cooldown period has elapsed."""
        if self._last_fired is None:
            return True
        elapsed = (datetime.utcnow() - self._last_fired).total_seconds()
        return elapsed >= self.cooldown_seconds

    def mark_fired(self) -> None:
        """Record the time the rule last fired."""
        object.__setattr__(self, "_last_fired", datetime.utcnow())


# ---------------------------------------------------------------------------
# Alert channels
# ---------------------------------------------------------------------------


class BaseAlertChannel:
    """Abstract base class for alert channels."""

    def send(self, alert: Alert) -> bool:
        """
        Send an alert through this channel.

        Returns
        -------
        bool
            ``True`` if delivery succeeded.
        """
        raise NotImplementedError

    def _build_message_text(self, alert: Alert) -> str:
        severity_emoji = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
        }.get(alert.severity, "🔔")

        return (
            f"{severity_emoji} [{alert.severity.value}] {alert.title}\n"
            f"Model: {alert.model_name}\n"
            f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
            f"Message: {alert.message}\n"
            + (
                f"Metric: {alert.metric_name} = {alert.metric_value:.4f} "
                f"(threshold: {alert.threshold})\n"
                if alert.metric_value is not None
                else ""
            )
        )


class SlackAlertChannel(BaseAlertChannel):
    """
    Send alerts to a Slack channel via incoming webhook.

    Parameters
    ----------
    webhook_url : str
        Slack incoming webhook URL.
    channel : str, optional
        Override the default channel configured in Slack.
    username : str
        Bot display name.
    timeout_seconds : int
        HTTP request timeout.
    """

    SEVERITY_COLORS = {
        AlertSeverity.INFO: "#36a64f",
        AlertSeverity.WARNING: "#ffaa00",
        AlertSeverity.CRITICAL: "#ff0000",
    }

    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "MLOps Sentinel",
        timeout_seconds: int = 10,
    ) -> None:
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.timeout_seconds = timeout_seconds

    def send(self, alert: Alert) -> bool:
        color = self.SEVERITY_COLORS.get(alert.severity, "#888888")
        fields = [
            {"title": "Model", "value": alert.model_name, "short": True},
            {"title": "Severity", "value": alert.severity.value, "short": True},
            {
                "title": "Timestamp",
                "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "short": True,
            },
        ]

        if alert.metric_name and alert.metric_value is not None:
            fields.append(
                {
                    "title": "Metric",
                    "value": f"{alert.metric_name} = {alert.metric_value:.4f}",
                    "short": True,
                }
            )
            fields.append(
                {
                    "title": "Threshold",
                    "value": str(alert.threshold),
                    "short": True,
                }
            )

        payload: Dict[str, Any] = {
            "username": self.username,
            "attachments": [
                {
                    "color": color,
                    "title": alert.title,
                    "text": alert.message,
                    "fields": fields,
                    "footer": "MLOps Sentinel",
                    "ts": int(alert.timestamp.timestamp()),
                }
            ],
        }
        if self.channel:
            payload["channel"] = self.channel

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                success = resp.status == 200
            if success:
                logger.info("Slack alert sent: %s", alert.title)
            return success
        except urllib.error.URLError as exc:
            logger.error("Slack alert failed: %s", exc)
            return False
        except Exception as exc:
            logger.error("Slack alert unexpected error: %s", exc)
            return False


class EmailAlertChannel(BaseAlertChannel):
    """
    Send alerts via SMTP email.

    Parameters
    ----------
    smtp_host : str
        SMTP server hostname.
    smtp_port : int
        SMTP server port (587 for STARTTLS, 465 for SSL).
    username : str
        SMTP username / sender address.
    password : str
        SMTP password or app password.
    recipients : list of str
        List of recipient email addresses.
    use_tls : bool
        Use STARTTLS if ``True``, SSL if ``False``.
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        recipients: List[str],
        use_tls: bool = True,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.recipients = recipients
        self.use_tls = use_tls

    def send(self, alert: Alert) -> bool:
        subject = f"[{alert.severity.value}] MLOps Sentinel: {alert.title}"
        body_text = self._build_message_text(alert)
        body_html = self._build_html_body(alert)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = self.username
        msg["To"] = ", ".join(self.recipients)
        msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(body_html, "html"))

        try:
            if self.use_tls:
                with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=15) as server:
                    server.ehlo()
                    server.starttls()
                    server.login(self.username, self.password)
                    server.sendmail(
                        self.username, self.recipients, msg.as_string()
                    )
            else:
                with smtplib.SMTP_SSL(
                    self.smtp_host, self.smtp_port, timeout=15
                ) as server:
                    server.login(self.username, self.password)
                    server.sendmail(
                        self.username, self.recipients, msg.as_string()
                    )
            logger.info("Email alert sent to %s", self.recipients)
            return True
        except smtplib.SMTPException as exc:
            logger.error("Email alert failed: %s", exc)
            return False
        except socket.timeout:
            logger.error("Email alert timed out")
            return False

    def _build_html_body(self, alert: Alert) -> str:
        severity_color = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffaa00",
            AlertSeverity.CRITICAL: "#ff0000",
        }.get(alert.severity, "#888888")

        metric_row = ""
        if alert.metric_name and alert.metric_value is not None:
            metric_row = f"""
            <tr>
              <td style="padding:4px;color:#666;">Metric</td>
              <td style="padding:4px;"><b>{alert.metric_name}</b> = {alert.metric_value:.4f}
                (threshold: {alert.threshold})</td>
            </tr>"""

        return f"""
        <html><body style="font-family:Arial,sans-serif;background:#f5f5f5;padding:20px;">
          <div style="max-width:600px;margin:auto;background:#fff;border-radius:8px;
                      border-left:6px solid {severity_color};padding:24px;">
            <h2 style="color:{severity_color};margin-top:0;">{alert.title}</h2>
            <p style="color:#333;">{alert.message}</p>
            <table style="width:100%;border-collapse:collapse;">
              <tr><td style="padding:4px;color:#666;">Model</td>
                  <td style="padding:4px;"><b>{alert.model_name}</b></td></tr>
              <tr><td style="padding:4px;color:#666;">Severity</td>
                  <td style="padding:4px;"><b style="color:{severity_color};">
                    {alert.severity.value}</b></td></tr>
              <tr><td style="padding:4px;color:#666;">Time</td>
                  <td style="padding:4px;">{alert.timestamp.strftime(
                      "%Y-%m-%d %H:%M:%S UTC")}</td></tr>
              {metric_row}
            </table>
            <p style="color:#999;font-size:12px;margin-top:20px;">
              Sent by MLOps Sentinel</p>
          </div>
        </body></html>"""


class WebhookAlertChannel(BaseAlertChannel):
    """
    Send alerts to an arbitrary HTTP webhook.

    The alert payload is POSTed as JSON.

    Parameters
    ----------
    url : str
        Webhook endpoint URL.
    headers : dict, optional
        Extra HTTP headers (e.g., ``{"Authorization": "Bearer token"}``).
    timeout_seconds : int
        Request timeout.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: int = 10,
    ) -> None:
        self.url = url
        self.extra_headers = headers or {}
        self.timeout_seconds = timeout_seconds

    def send(self, alert: Alert) -> bool:
        payload = alert.to_dict()
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.extra_headers}

        try:
            req = urllib.request.Request(self.url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=self.timeout_seconds) as resp:
                success = 200 <= resp.status < 300
            if success:
                logger.info("Webhook alert sent to %s", self.url)
            return success
        except Exception as exc:
            logger.error("Webhook alert failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class AlertManager:
    """
    Orchestrates alert rule evaluation and dispatch to configured channels.

    Usage
    -----
    .. code-block:: python

        manager = AlertManager()
        manager.add_channel(SlackAlertChannel(webhook_url="..."))
        manager.add_rule(AlertRule("low_accuracy", "accuracy", 0.85, comparison="lt"))

        # Evaluate all rules against current metrics
        manager.evaluate_rules({"accuracy": 0.72, "drift_score": 0.18})

        # Manually fire an alert
        manager.fire(Alert(...))
    """

    def __init__(self, max_history: int = 500) -> None:
        self._channels: List[BaseAlertChannel] = []
        self._rules: List[AlertRule] = []
        self._history: List[Alert] = []
        self.max_history = max_history
        self._lock = threading.Lock()
        self._counter = 0

    def add_channel(self, channel: BaseAlertChannel) -> None:
        """Register an alert delivery channel."""
        self._channels.append(channel)
        logger.info("Alert channel added: %s", type(channel).__name__)

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules.append(rule)
        logger.info("Alert rule added: %s", rule.name)

    def remove_rule(self, name: str) -> bool:
        """Remove a rule by name. Returns True if found."""
        before = len(self._rules)
        self._rules = [r for r in self._rules if r.name != name]
        return len(self._rules) < before

    def fire(self, alert: Alert) -> int:
        """
        Dispatch an alert to all registered channels.

        Returns
        -------
        int
            Number of channels that successfully delivered the alert.
        """
        with self._lock:
            self._history.append(alert)
            if len(self._history) > self.max_history:
                self._history = self._history[-self.max_history :]

        successes = 0
        for channel in self._channels:
            try:
                if channel.send(alert):
                    successes += 1
            except Exception as exc:
                logger.error("Channel %s failed: %s", type(channel).__name__, exc)

        logger.info(
            "Alert fired: %s | %s | delivered to %d/%d channels",
            alert.severity.value,
            alert.title,
            successes,
            len(self._channels),
        )
        return successes

    def evaluate_rules(self, metrics: Dict[str, Any]) -> List[Alert]:
        """
        Evaluate all registered rules against a metrics snapshot.

        Parameters
        ----------
        metrics : dict
            Mapping of metric name → current value.

        Returns
        -------
        list of Alert
            Alerts that were fired.
        """
        fired: List[Alert] = []

        for rule in self._rules:
            if not rule.enabled:
                continue
            value = metrics.get(rule.metric)
            if value is None:
                continue

            try:
                value = float(value)
            except (TypeError, ValueError):
                continue

            if rule.evaluate(value) and rule.can_fire():
                model_name = metrics.get("model_name", "unknown")
                alert = self._build_rule_alert(rule, value, str(model_name))
                rule.mark_fired()
                self.fire(alert)
                fired.append(alert)

        return fired

    def alert_on_drift(self, drift_report: Any) -> Optional[Alert]:
        """
        Fire a drift alert from a :class:`~sentinel.core.drift.DriftReport`.

        Parameters
        ----------
        drift_report : DriftReport
        """
        if not drift_report.is_drifted:
            return None

        severity = (
            AlertSeverity.CRITICAL
            if drift_report.drift_score > 0.4
            else AlertSeverity.WARNING
        )
        self._counter += 1
        alert = Alert(
            id=f"drift-{self._counter}",
            title="Data Drift Detected",
            message=(
                f"Drift score {drift_report.drift_score:.4f} exceeded threshold. "
                f"Drifted features: {', '.join(drift_report.drifted_features) or 'none'}."
            ),
            severity=severity,
            timestamp=datetime.utcnow(),
            model_name=drift_report.model_name,
            metric_name="drift_score",
            metric_value=drift_report.drift_score,
            threshold=0.15,
        )
        self.fire(alert)
        return alert

    def alert_on_performance_degradation(
        self,
        model_name: str,
        metric_name: str,
        current_value: float,
        baseline_value: float,
        threshold_delta: float = 0.05,
        severity: str = "WARNING",
    ) -> Optional[Alert]:
        """Fire an alert when performance drops by more than ``threshold_delta``."""
        delta = baseline_value - current_value
        if delta <= threshold_delta:
            return None

        self._counter += 1
        alert = Alert(
            id=f"perf-{self._counter}",
            title=f"Performance Degradation: {metric_name}",
            message=(
                f"{metric_name} dropped from {baseline_value:.4f} to "
                f"{current_value:.4f} (delta: {delta:.4f})."
            ),
            severity=AlertSeverity(severity),
            timestamp=datetime.utcnow(),
            model_name=model_name,
            metric_name=metric_name,
            metric_value=current_value,
            threshold=baseline_value - threshold_delta,
        )
        self.fire(alert)
        return alert

    def get_history(
        self,
        limit: int = 50,
        severity: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return recent alert history.

        Parameters
        ----------
        limit : int
            Maximum number of alerts to return.
        severity : str, optional
            Filter by severity (``"INFO"``, ``"WARNING"``, ``"CRITICAL"``).
        """
        with self._lock:
            history = list(reversed(self._history))

        if severity:
            history = [a for a in history if a.severity.value == severity.upper()]

        return [a.to_dict() for a in history[:limit]]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_rule_alert(
        self, rule: AlertRule, value: float, model_name: str
    ) -> Alert:
        self._counter += 1
        ops_desc = {
            "gt": "exceeded",
            "lt": "fell below",
            "gte": "reached or exceeded",
            "lte": "reached or fell below",
            "eq": "equalled",
        }
        verb = ops_desc.get(rule.comparison, "violated")

        return Alert(
            id=f"rule-{self._counter}",
            title=f"Alert Rule Triggered: {rule.name}",
            message=(
                f"Metric '{rule.metric}' {verb} threshold {rule.threshold}. "
                f"Current value: {value:.4f}."
            ),
            severity=AlertSeverity(rule.severity),
            timestamp=datetime.utcnow(),
            model_name=model_name,
            metric_name=rule.metric,
            metric_value=value,
            threshold=rule.threshold,
        )
