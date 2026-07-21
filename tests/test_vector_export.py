"""Tests for Vector CANoe/CANalyzer export module."""

from __future__ import annotations

import pytest

from ev_qa_framework.vector_export import VectorExporter


@pytest.fixture
def exporter():
    """Create a VectorExporter instance."""
    return VectorExporter()


@pytest.fixture
def sample_trace():
    """Sample CAN trace data for testing."""
    return [
        {
            "timestamp": 1.0,
            "can_id": 0x123,
            "data": bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08]),
            "channel": 0,
        },
        {
            "timestamp": 1.5,
            "can_id": 0x456,
            "data": bytes([0xAA, 0xBB, 0xCC]),
            "channel": 1,
        },
        {
            "timestamp": 2.0,
            "can_id": 0x789,
            "data": bytes([]),
            "channel": 0,
        },
    ]


@pytest.fixture
def sample_results():
    """Sample test results for testing."""
    return {
        "test_name": "Battery Voltage Test",
        "test_id": "TEST-001",
        "status": "PASS",
        "timestamp": "2024-01-15T10:30:00",
        "duration_ms": 123.45,
        "details": "All voltage readings within tolerance",
        "measurements": [
            {
                "name": "voltage",
                "value": 405.3,
                "unit": "V",
                "limit_min": 350.0,
                "limit_max": 420.0,
            },
            {
                "name": "current",
                "value": 2.5,
                "unit": "A",
                "limit_min": 0.0,
                "limit_max": 5.0,
            },
        ],
    }


class TestASCExport:
    """Test ASC export functionality."""

    def test_export_creates_file(self, exporter, sample_trace, tmp_path):
        """ASC export should create a file."""
        output = tmp_path / "test.asc"
        result = exporter.export_asc(sample_trace, output)
        assert result.exists()
        assert result == output

    def test_export_content_format(self, exporter, sample_trace, tmp_path):
        """ASC export should produce correct format."""
        output = tmp_path / "test.asc"
        exporter.export_asc(sample_trace, output)

        content = output.read_text(encoding="utf-8")
        lines = content.strip().split("\n")

        # Check header
        assert lines[0].startswith("date ")
        assert "base hex" in lines[1]
        assert "internal events" in lines[2]
        assert "no direction" in lines[3]

        # Check data lines (skip header)
        data_lines = lines[4:]
        assert len(data_lines) == 3

        # First message
        parts = data_lines[0].split()
        assert parts[0] == "0"  # channel
        assert parts[2] == "123"  # can_id (hex)
        assert parts[3] == "8"  # dlc
        assert parts[4] == "0102030405060708"  # data hex

        # Second message
        parts = data_lines[1].split()
        assert parts[0] == "1"  # channel
        assert parts[2] == "456"  # can_id (hex)
        assert parts[3] == "3"  # dlc
        assert parts[4] == "AABBCC"  # data hex

    def test_export_empty_trace(self, exporter, tmp_path):
        """ASC export should handle empty trace."""
        output = tmp_path / "empty.asc"
        result = exporter.export_asc([], output)
        assert result.exists()

        content = output.read_text(encoding="utf-8")
        lines = content.strip().split("\n")
        # Only header lines, no data
        assert len(lines) == 4


class TestASCImport:
    """Test ASC import functionality."""

    def test_import_roundtrip(self, exporter, sample_trace, tmp_path):
        """Export then import should preserve data."""
        output = tmp_path / "roundtrip.asc"
        exporter.export_asc(sample_trace, output)

        imported = exporter.import_asc(output)
        assert len(imported) == len(sample_trace)

        for orig, imp in zip(sample_trace, imported):
            assert imp["can_id"] == orig["can_id"]
            assert imp["channel"] == orig["channel"]
            assert imp["data"] == orig["data"]
            # Timestamp should be close (within 1 microsecond tolerance)
            assert abs(imp["timestamp"] - orig["timestamp"]) < 1e-6

    def test_import_ignores_headers(self, exporter, tmp_path):
        """Import should skip header lines."""
        output = tmp_path / "test.asc"
        # Create minimal ASC file
        content = (
            "date Mon Jan 15 10:30:00.000000 PM 2024\n"
            "base hex  timestamps absolute\n"
            "internal events log\n"
            "no direction\n"
            "  0  000F4240  123  8  0102030405060708\n"
        )
        output.write_text(content, encoding="utf-8")

        imported = exporter.import_asc(output)
        assert len(imported) == 1
        assert imported[0]["can_id"] == 0x123

    def test_import_handles_unparseable_lines(self, exporter, tmp_path):
        """Import should skip unparseable lines gracefully."""
        output = tmp_path / "test.asc"
        content = (
            "date Mon Jan 15 10:30:00.000000 PM 2024\n"
            "base hex  timestamps absolute\n"
            "internal events log\n"
            "no direction\n"
            "  invalid line\n"
            "  0  000F4240  123  8  0102030405060708\n"
        )
        output.write_text(content, encoding="utf-8")

        imported = exporter.import_asc(output)
        assert len(imported) == 1


class TestBLFExport:
    """Test BLF export functionality."""

    def test_export_blf_fallback_to_asc(self, exporter, sample_trace, tmp_path):
        """BLF export should fall back to ASC if BLFWriter unavailable."""
        from ev_qa_framework.vector_export import _BLF_AVAILABLE

        output = tmp_path / "test.blf"
        result = exporter.export_blf(sample_trace, output)

        if not _BLF_AVAILABLE:
            # Should fall back to ASC
            assert result.suffix == ".asc"
            assert result.exists()
        else:
            assert result.suffix == ".blf"
            assert result.exists()


class TestTestVectorExport:
    """Test test vector CSV export."""

    def test_export_creates_csv(self, exporter, sample_results, tmp_path):
        """Test vector export should create CSV file."""
        output = tmp_path / "test_vector.csv"
        result = exporter.export_test_vector(sample_results, output)
        assert result.exists()
        assert result.suffix == ".csv"

    def test_export_content(self, exporter, sample_results, tmp_path):
        """Test vector export should contain correct data."""
        output = tmp_path / "test_vector.csv"
        exporter.export_test_vector(sample_results, output)

        content = output.read_text(encoding="utf-8")
        assert "Test Vector Export" in content
        assert "Battery Voltage Test" in content
        assert "TEST-001" in content
        assert "PASS" in content
        assert "voltage" in content
        assert "405.3" in content

    def test_export_minimal_results(self, exporter, tmp_path):
        """Test vector export should handle minimal results."""
        output = tmp_path / "minimal.csv"
        result = exporter.export_test_vector({}, output)
        assert result.exists()

        content = result.read_text(encoding="utf-8")
        assert "unknown" in content  # default test_name
