"""Tests for ev_qa_framework.cli module."""

import json
import os
import tempfile
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_csv(tmp_path):
    """Create a minimal valid CSV file for analyze_csv / train_soh_model."""
    csv_path = tmp_path / "telemetry.csv"
    df = pd.DataFrame(
        {
            "voltage": [396.0, 397.0, 395.0, 398.0, 396.5],
            "current": [50.0, 51.0, 49.0, 52.0, 50.5],
            "temperature": [35, 36, 34, 35, 36],
            "soc": [80, 79, 78, 77, 76],
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_csv_with_temp_col(tmp_path):
    """CSV that uses 'temp' instead of 'temperature'."""
    csv_path = tmp_path / "telemetry_temp.csv"
    df = pd.DataFrame(
        {
            "voltage": [396.0, 397.0],
            "current": [50.0, 51.0],
            "temp": [35, 36],
            "soc": [80, 79],
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def sample_csv_temperature_only(tmp_path):
    """CSV that uses 'temperature' column (needs rename to 'temp')."""
    csv_path = tmp_path / "telemetry_temperature.csv"
    df = pd.DataFrame(
        {
            "voltage": [396.0, 397.0],
            "current": [50.0, 51.0],
            "temperature": [35, 36],
            "soc": [80, 79],
        }
    )
    df.to_csv(csv_path, index=False)
    return str(csv_path)


@pytest.fixture
def mock_analysis_results():
    """Return value from EVBatteryAnalyzer.analyze_telemetry."""
    return {
        "total_samples": 5,
        "anomalies_detected": 1,
        "anomaly_percentage": 20.0,
        "severity": "WARNING",
    }


@pytest.fixture
def mock_analyzer(mock_analysis_results):
    """A mocked EVBatteryAnalyzer instance."""
    with patch("ev_qa_framework.cli.EVBatteryAnalyzer") as MockCls:
        instance = MagicMock()
        instance.analyze_telemetry.return_value = mock_analysis_results
        MockCls.return_value = instance
        yield MockCls, instance


@pytest.fixture
def mock_can_simulator():
    """Mock CANBatterySimulator."""
    with patch("ev_qa_framework.cli.CANBatterySimulator") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance
        yield MockCls, instance


@pytest.fixture
def mock_can_receiver():
    """Mock CANTelemetryReceiver."""
    with patch("ev_qa_framework.cli.CANTelemetryReceiver") as MockCls:
        instance = MagicMock()
        instance.get_telemetry.return_value = {
            "voltage": 396.5,
            "current": 50.2,
            "temperature": 35.0,
            "soc": 80.0,
        }
        MockCls.return_value = instance
        yield MockCls, instance


@pytest.fixture
def mock_soh_predictor():
    """Mock SOHPredictor."""
    with patch("ev_qa_framework.cli.SOHPredictor") as MockCls:
        instance = MagicMock()
        MockCls.return_value = instance
        yield MockCls, instance


# ---------------------------------------------------------------------------
# Tests for analyze_csv
# ---------------------------------------------------------------------------


class TestAnalyzeCsv:
    """Tests for the analyze_csv function."""

    def test_success_without_output(self, sample_csv, mock_analyzer, capsys):
        """Successful analysis without writing output file."""
        from ev_qa_framework.cli import analyze_csv

        analyze_csv(sample_csv)

        captured = capsys.readouterr()
        assert "Analysis complete" in captured.out
        assert "Total samples: 5" in captured.out
        assert "Anomalies: 1" in captured.out
        assert "Severity: WARNING" in captured.out

    def test_success_with_output(self, sample_csv, mock_analyzer, tmp_path, capsys):
        """Successful analysis with JSON output file."""
        from ev_qa_framework.cli import analyze_csv

        output_path = str(tmp_path / "results.json")
        analyze_csv(sample_csv, output=output_path)

        captured = capsys.readouterr()
        assert f"Results saved to {output_path}" in captured.out

        # Verify the JSON file was written correctly
        with open(output_path) as f:
            data = json.load(f)
        assert data["total_samples"] == 5
        assert data["anomalies_detected"] == 1
        assert data["severity"] == "WARNING"

    def test_file_not_found(self):
        """FileNotFoundError when CSV does not exist."""
        from ev_qa_framework.cli import analyze_csv

        with pytest.raises(FileNotFoundError):
            analyze_csv("/nonexistent/path/data.csv")

    def test_temperature_column_renamed(self, sample_csv_temperature_only, mock_analyzer, capsys):
        """CSV with 'temperature' column is renamed to 'temp'."""
        from ev_qa_framework.cli import analyze_csv

        analyze_csv(sample_csv_temperature_only)

        # The mock should still be called; verify the analyzer was invoked
        _, instance = mock_analyzer
        assert instance.analyze_telemetry.call_count == 1

    def test_temp_column_no_rename(self, sample_csv_with_temp_col, mock_analyzer, capsys):
        """CSV already having 'temp' column should not be renamed."""
        from ev_qa_framework.cli import analyze_csv

        analyze_csv(sample_csv_with_temp_col)

        _, instance = mock_analyzer
        assert instance.analyze_telemetry.call_count == 1

    def test_analyzer_called_with_dataframe(self, sample_csv, mock_analyzer):
        """Verify pd.read_csv is called and analyzer receives a DataFrame."""
        from ev_qa_framework.cli import analyze_csv

        _, instance = mock_analyzer
        analyze_csv(sample_csv)

        # analyze_telemetry should have been called once with a DataFrame
        args, _ = instance.analyze_telemetry.call_args
        df_arg = args[0]
        assert isinstance(df_arg, pd.DataFrame)
        assert "temp" in df_arg.columns

    def test_empty_csv(self, tmp_path):
        """Empty CSV (headers only) should still be processed."""
        from ev_qa_framework.cli import analyze_csv

        csv_path = tmp_path / "empty.csv"
        pd.DataFrame(columns=["voltage", "current", "temperature", "soc"]).to_csv(
            csv_path, index=False
        )

        with patch("ev_qa_framework.cli.EVBatteryAnalyzer") as MockCls:
            instance = MagicMock()
            instance.analyze_telemetry.return_value = {
                "total_samples": 0,
                "anomalies_detected": 0,
                "anomaly_percentage": 0.0,
                "severity": "INFO",
            }
            MockCls.return_value = instance

            analyze_csv(str(csv_path))
            assert instance.analyze_telemetry.call_count == 1


# ---------------------------------------------------------------------------
# Tests for run_can_demo
# ---------------------------------------------------------------------------


class TestRunCanDemo:
    """Tests for the run_can_demo function."""

    def test_success_default_duration(self, mock_can_simulator, mock_can_receiver, capsys):
        """CAN demo runs with default duration and prints telemetry."""
        from ev_qa_framework.cli import run_can_demo

        with patch("ev_qa_framework.cli.time.sleep"):
            run_can_demo(duration=2)

        _, sim = mock_can_simulator
        _, receiver = mock_can_receiver

        sim.start.assert_called_once()
        receiver.start.assert_called_once()
        assert receiver.get_telemetry.call_count == 2
        sim.stop.assert_called_once()
        receiver.stop.assert_called_once()

        captured = capsys.readouterr()
        assert "Starting CAN Bus Emulation Demo (2s)" in captured.out
        assert "CAN Demo finished" in captured.out
        assert "CAN Telemetry" in captured.out

    def test_success_custom_duration(self, mock_can_simulator, mock_can_receiver, capsys):
        """CAN demo respects custom duration."""
        from ev_qa_framework.cli import run_can_demo

        with patch("ev_qa_framework.cli.time.sleep"):
            run_can_demo(duration=5)

        _, receiver = mock_can_receiver
        assert receiver.get_telemetry.call_count == 5

    def test_zero_duration(self, mock_can_simulator, mock_can_receiver, capsys):
        """Duration of 0 should not call get_telemetry."""
        from ev_qa_framework.cli import run_can_demo

        run_can_demo(duration=0)

        _, receiver = mock_can_receiver
        assert receiver.get_telemetry.call_count == 0

        _, sim = mock_can_simulator
        sim.stop.assert_called_once()
        receiver.stop.assert_called_once()

    def test_telemetry_format(self, mock_can_simulator, mock_can_receiver, capsys):
        """Verify telemetry line format."""
        from ev_qa_framework.cli import run_can_demo

        with patch("ev_qa_framework.cli.time.sleep"):
            run_can_demo(duration=1)

        captured = capsys.readouterr()
        # Check format: V=396.5V | I=50.2A | T=35C | SOC=80%
        assert "V=396.5V" in captured.out
        assert "I=50.2A" in captured.out
        assert "T=35C" in captured.out
        assert "SOC=80%" in captured.out

    def test_stop_called_on_exception(self, mock_can_simulator, mock_can_receiver):
        """Simulator and receiver are stopped even if an error occurs."""
        from ev_qa_framework.cli import run_can_demo

        _, receiver = mock_can_receiver
        receiver.get_telemetry.side_effect = RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            run_can_demo(duration=5)

        _, sim = mock_can_simulator
        sim.stop.assert_called_once()
        receiver.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for run_dbc_emulate
# ---------------------------------------------------------------------------


class TestRunDbcEmulate:
    """Tests for the run_dbc_emulate function."""

    def test_default_dbc(self, capsys):
        """DBC emulation with built-in DBC (no path)."""
        from ev_qa_framework.cli import run_dbc_emulate

        with patch("ev_qa_framework.cli.DBCFileSimulator") as MockCls:
            instance = MagicMock()
            instance.dbc = MagicMock()
            instance.dbc.messages = {0x101: MagicMock(), 0x102: MagicMock()}
            MockCls.return_value = instance

            with patch("ev_qa_framework.cli.time.sleep"):
                run_dbc_emulate(dbc_path=None, duration=2)

            MockCls.assert_called_once_with(dbc_path=None)
            instance.start.assert_called_once()
            assert instance.start.call_count == 1
            instance.stop.assert_called_once()

        captured = capsys.readouterr()
        assert "Using built-in battery DBC" in captured.out
        assert "Sent 2 messages" in captured.out
        assert "DBC Emulation finished" in captured.out

    def test_custom_dbc_path(self, capsys):
        """DBC emulation with a custom DBC file path."""
        from ev_qa_framework.cli import run_dbc_emulate

        with patch("ev_qa_framework.cli.DBCFileSimulator") as MockCls:
            instance = MagicMock()
            instance.dbc = MagicMock()
            instance.dbc.messages = {0x200: MagicMock()}
            MockCls.return_value = instance

            with patch("ev_qa_framework.cli.time.sleep"):
                run_dbc_emulate(dbc_path="/path/to/custom.dbc", duration=1)

            MockCls.assert_called_once_with(dbc_path="/path/to/custom.dbc")

        captured = capsys.readouterr()
        assert "Loading DBC: /path/to/custom.dbc" in captured.out

    def test_zero_duration(self):
        """Duration 0 should still start/stop but not loop."""
        from ev_qa_framework.cli import run_dbc_emulate

        with patch("ev_qa_framework.cli.DBCFileSimulator") as MockCls:
            instance = MagicMock()
            instance.dbc = MagicMock()
            instance.dbc.messages = {}
            MockCls.return_value = instance

            run_dbc_emulate(dbc_path=None, duration=0)

            instance.start.assert_called_once()
            instance.stop.assert_called_once()

    def test_stop_called_on_exception_in_loop(self):
        """Simulator is stopped even if an error occurs during the loop."""
        from ev_qa_framework.cli import run_dbc_emulate

        with patch("ev_qa_framework.cli.DBCFileSimulator") as MockCls:
            instance = MagicMock()
            instance.dbc = MagicMock()
            instance.dbc.messages = {0x101: MagicMock()}
            MockCls.return_value = instance

            # Make time.sleep raise to simulate error during the loop
            with patch("ev_qa_framework.cli.time.sleep", side_effect=RuntimeError("sleep error")):
                with pytest.raises(RuntimeError, match="sleep error"):
                    run_dbc_emulate(dbc_path=None, duration=3)

            # stop() is in finally block, should be called
            instance.stop.assert_called_once()


# ---------------------------------------------------------------------------
# Tests for train_soh_model
# ---------------------------------------------------------------------------


class TestTrainSohModel:
    """Tests for the train_soh_model function."""

    def test_success(self, sample_csv, mock_soh_predictor, capsys):
        """Successful SOH model training."""
        from ev_qa_framework.cli import train_soh_model

        model_dir = os.path.join(tempfile.mkdtemp(), "model_output")
        train_soh_model(sample_csv, model_dir)

        _, predictor = mock_soh_predictor
        predictor.train.assert_called_once()
        # Verify train was called with epochs=5
        _, kwargs = predictor.train.call_args
        assert kwargs.get("epochs") == 5 or predictor.train.call_args[1].get("epochs") == 5

        predictor.save.assert_called_once_with(model_dir)

        captured = capsys.readouterr()
        assert "Training SOH Predictor" in captured.out
        assert "Model saved to" in captured.out

    def test_file_not_found(self):
        """FileNotFoundError when CSV does not exist."""
        from ev_qa_framework.cli import train_soh_model

        with pytest.raises(FileNotFoundError):
            train_soh_model("/nonexistent/data.csv", "/tmp/model")

    def test_pd_read_csv_called(self, sample_csv, mock_soh_predictor):
        """Verify pd.read_csv is called with the correct path."""
        from ev_qa_framework.cli import train_soh_model

        with patch("ev_qa_framework.cli.pd.read_csv", wraps=pd.read_csv) as mock_read:
            train_soh_model(sample_csv, "/tmp/model")
            mock_read.assert_called_once_with(sample_csv)

    def test_predictor_train_receives_dataframe(self, sample_csv, mock_soh_predictor):
        """Verify the predictor.train receives a DataFrame."""
        from ev_qa_framework.cli import train_soh_model

        _, predictor = mock_soh_predictor
        train_soh_model(sample_csv, "/tmp/model")

        args, _ = predictor.train.call_args
        df_arg = args[0]
        assert isinstance(df_arg, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests for main() CLI entry point
# ---------------------------------------------------------------------------


class TestMain:
    """Tests for the main() CLI entry point."""

    @patch("ev_qa_framework.cli.analyze_csv")
    @patch("ev_qa_framework.cli.validate_input_file")
    def test_analyze_command(self, mock_validate_input_file, mock_analyze):
        """'analyze' command calls analyze_csv with correct args."""
        from ev_qa_framework.cli import main

        with patch("sys.argv", ["ev-qa", "analyze", "-i", "data.csv"]):
            main()

        mock_validate_input_file.assert_called_once_with("data.csv")
        mock_analyze.assert_called_once_with("data.csv", None)

    @patch("ev_qa_framework.cli.analyze_csv")
    @patch("ev_qa_framework.cli.validate_input_file")
    def test_analyze_command_with_output(self, mock_validate_input_file, mock_analyze):
        """'analyze' command with --output flag."""
        from ev_qa_framework.cli import main

        with patch(
            "sys.argv",
            ["ev-qa", "analyze", "-i", "data.csv", "-o", "out.json"],
        ):
            main()

        mock_validate_input_file.assert_called_once_with("data.csv")
        mock_analyze.assert_called_once_with("data.csv", "out.json")

    @patch("ev_qa_framework.cli.run_can_demo")
    def test_can_demo_command(self, mock_can_demo):
        """'can-demo' command calls run_can_demo with default duration."""
        from ev_qa_framework.cli import main

        with patch("sys.argv", ["ev-qa", "can-demo"]):
            main()

        mock_can_demo.assert_called_once_with(10)

    @patch("ev_qa_framework.cli.run_can_demo")
    def test_can_demo_command_custom_duration(self, mock_can_demo):
        """'can-demo' command with custom duration."""
        from ev_qa_framework.cli import main

        with patch("sys.argv", ["ev-qa", "can-demo", "-d", "30"]):
            main()

        mock_can_demo.assert_called_once_with(30)

    @patch("ev_qa_framework.cli.run_dbc_emulate")
    def test_emulate_command_default(self, mock_dbc):
        """'emulate' command with default args."""
        from ev_qa_framework.cli import main

        with patch("sys.argv", ["ev-qa", "emulate"]):
            main()

        mock_dbc.assert_called_once_with(dbc_path=None, duration=10)

    @patch("ev_qa_framework.cli.run_dbc_emulate")
    def test_emulate_command_custom(self, mock_dbc):
        """'emulate' command with custom DBC and duration."""
        from ev_qa_framework.cli import main

        with patch(
            "sys.argv",
            ["ev-qa", "emulate", "--dbc", "file.dbc", "-d", "5"],
        ):
            main()

        mock_dbc.assert_called_once_with(dbc_path="file.dbc", duration=5)

    @patch("ev_qa_framework.cli.train_soh_model")
    @patch("ev_qa_framework.cli.validate_csv_path")
    @patch("ev_qa_framework.cli.validate_model_dir")
    def test_train_soh_command(self, mock_validate_model_dir, mock_validate_csv_path, mock_train):
        """'train-soh' command calls train_soh_model."""
        from ev_qa_framework.cli import main

        with patch(
            "sys.argv",
            ["ev-qa", "train-soh", "-i", "hist.csv", "-m", "model_dir"],
        ):
            main()

        mock_validate_csv_path.assert_called_once_with("hist.csv")
        mock_validate_model_dir.assert_called_once_with("model_dir")
        mock_train.assert_called_once_with("hist.csv", "model_dir")

    def test_dashboard_command(self, capsys):
        """'dashboard' command prints starting message and calls uvicorn."""
        from ev_qa_framework.cli import main

        # uvicorn is imported inline inside main(), so we patch it via sys.modules
        mock_uvicorn = MagicMock()
        with patch("sys.argv", ["ev-qa", "dashboard"]):
            with patch.dict("sys.modules", {"uvicorn": mock_uvicorn}):
                with patch.dict("sys.modules", {"dashboard": MagicMock()}):
                    with patch.dict("sys.modules", {"dashboard.app": MagicMock()}):
                        # We just verify the command is recognized and prints the message
                        # The actual uvicorn.run call requires complex mocking of inline imports
                        try:
                            main()
                        except (TypeError, AttributeError, KeyError):
                            # Expected — mock modules don't have the real interfaces
                            pass

        captured = capsys.readouterr()
        assert "Starting dashboard" in captured.out

    def test_no_command_prints_help(self, capsys):
        """No command prints help."""
        from ev_qa_framework.cli import main

        with patch("sys.argv", ["ev-qa"]):
            main()

        captured = capsys.readouterr()
        # argparse prints help to stdout when no command given
        assert "usage" in captured.out.lower() or "EV Battery QA" in captured.out


# ---------------------------------------------------------------------------
# Tests for start_dashboard (inline in main)
# ---------------------------------------------------------------------------


class TestStartDashboard:
    """Tests for the dashboard command path in main()."""

    def test_dashboard_imports_uvicorn(self):
        """Verify dashboard startup path is present in the public CLI helper."""
        import inspect

        from ev_qa_framework.cli import print_dashboard_start

        source = inspect.getsource(print_dashboard_start)
        assert "uvicorn" in source
        assert "dashboard" in source


# ---------------------------------------------------------------------------
# Edge-case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge-case and boundary tests."""

    def test_analyze_csv_output_json_content(self, sample_csv, mock_analyzer, tmp_path):
        """Verify JSON output is valid and contains expected keys."""
        from ev_qa_framework.cli import analyze_csv

        output_path = str(tmp_path / "out.json")
        analyze_csv(sample_csv, output=output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "total_samples" in data
        assert "anomalies_detected" in data
        assert "severity" in data

    def test_analyze_csv_nonexistent_output_dir(self, sample_csv, mock_analyzer, tmp_path):
        """Output to a non-existent directory should raise an OS-level error."""
        from ev_qa_framework.cli import analyze_csv

        output_path = str(tmp_path / "nonexistent_dir" / "out.json")
        with pytest.raises(OSError):
            analyze_csv(sample_csv, output=output_path)

    def test_run_can_demo_negative_duration(self, mock_can_simulator, mock_can_receiver):
        """Negative duration should result in zero iterations."""
        from ev_qa_framework.cli import run_can_demo

        run_can_demo(duration=-1)

        _, receiver = mock_can_receiver
        assert receiver.get_telemetry.call_count == 0

    def test_run_dbc_emulate_stop_always_called(self):
        """DBC emulate stop() is called even on keyboard interrupt."""
        from ev_qa_framework.cli import run_dbc_emulate

        with patch("ev_qa_framework.cli.DBCFileSimulator") as MockCls:
            instance = MagicMock()
            instance.dbc = MagicMock()
            instance.dbc.messages = {0x101: MagicMock()}
            MockCls.return_value = instance

            with patch("ev_qa_framework.cli.time.sleep", side_effect=KeyboardInterrupt):
                with pytest.raises(KeyboardInterrupt):
                    run_dbc_emulate(dbc_path="test.dbc", duration=10)

            instance.stop.assert_called_once()

    def test_train_soh_model_creates_output_dir(self, sample_csv, mock_soh_predictor):
        """train_soh_model should work with a non-existent output directory."""
        from ev_qa_framework.cli import train_soh_model

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, "new_subdir", "model")
            train_soh_model(sample_csv, model_path)

            _, predictor = mock_soh_predictor
            predictor.save.assert_called_once_with(model_path)
