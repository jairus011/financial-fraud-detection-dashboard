import pytest
from streamlit.testing.v1 import AppTest

from src.config import ROOT


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()


@pytest.mark.parametrize("page", ["Executive Overview", "Fraud Analysis", "Transaction Patterns",
                                 "Risk/Anomaly Analysis", "Model Performance", "Transaction Prediction"])
def test_dashboard_pages_render(app, page):
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception
    assert app.title[0].value == page


def test_single_prediction_form(app):
    app.sidebar.radio[0].set_value("Transaction Prediction").run()
    next(button for button in app.button if button.label == "Score transaction").click().run()
    assert not app.exception
    assert any(m.label == "Fraud model score" for m in app.metric)


def test_alert_simulation(app):
    app.sidebar.radio[0].set_value("Risk/Anomaly Analysis").run()
    next(button for button in app.button if button.label == "Simulate high-risk alerts").click().run()
    assert not app.exception
    assert any("No email" in success.value for success in app.success)


def test_filter_changes_kpis_but_not_test_metrics(app):
    app.sidebar.radio[0].set_value("Executive Overview").run()
    app.sidebar.selectbox[0].set_value("test").run()
    total = next(m.value for m in app.metric if m.label == "Total transactions")
    assert total == "1,000"
    app.sidebar.radio[0].set_value("Model Performance").run()
    assert next(m.value for m in app.metric if m.label == "Recall") == "50.00%"
    app.sidebar.selectbox[0].set_value("All transactions").run()


def test_empty_filters_have_clear_message(app):
    app.sidebar.radio[0].set_value("Executive Overview").run()
    app.sidebar.selectbox[0].set_value("train").run()
    app.sidebar.date_input[0].set_value(("2024-02-01", "2024-02-02")).run()
    assert not app.exception
    assert any("No transactions" in message.value for message in app.info)
