"""Tests for DBCParser."""

import pytest
import os
import tempfile
from ev_qa_framework.dbc_parser import DBCParser, builtin_dbc, battery_dbc_content


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_dbc():
    """Create a minimal temporary DBC file."""
    content = '''VERSION "1.0"

BS_:

BU_: TEST_ECU

BO_ 257 TestMessage: 8 TEST_ECU
 SG_ Voltage : 0|16@1+ (0.1,0) [0|1000] "V" TEST_ECU
 SG_ Current : 16|16@1- (0.1,0) [-500|500] "A" TEST_ECU

CM_ BO_ 257 "Test message";
CM_ SG_ 257 Voltage "Battery voltage";
'''
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False)
    tmp.write(content)
    tmp.close()
    yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def builtin():
    return builtin_dbc()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDBCParser:
    """DBC parsing tests."""

    def test_parse_minimal(self, minimal_dbc):
        """Can parse a minimal DBC file."""
        parser = DBCParser(minimal_dbc)
        assert len(parser.messages) == 1
        assert 257 in parser.messages

    def test_message_properties(self, minimal_dbc):
        """Parsed message has correct properties."""
        parser = DBCParser(minimal_dbc)
        msg = parser.get_message(257)
        assert msg is not None
        assert msg.name == "TestMessage"
        assert msg.dlc == 8
        assert msg.transmitter == "TEST_ECU"
        assert msg.is_extended is False

    def test_signal_count(self, minimal_dbc):
        """Message has correct number of signals."""
        parser = DBCParser(minimal_dbc)
        msg = parser.get_message(257)
        assert len(msg.signals) == 2

    def test_signal_properties(self, minimal_dbc):
        """Signal has correct properties."""
        parser = DBCParser(minimal_dbc)
        sig = parser.get_signal(257, "Voltage")
        assert sig is not None
        assert sig.name == "Voltage"
        assert sig.start_bit == 0
        assert sig.length == 16
        assert sig.byte_order == "Intel"
        assert sig.signed is False
        assert sig.scale == 0.1
        assert sig.offset == 0.0
        assert sig.unit == "V"

    def test_get_message_by_name(self, minimal_dbc):
        """Lookup by symbolic name works."""
        parser = DBCParser(minimal_dbc)
        msg = parser.get_message_by_name("TestMessage")
        assert msg is not None
        assert msg.id == 257

    def test_decode_unsigned(self, minimal_dbc):
        """Decoding unsigned signal from raw bytes."""
        parser = DBCParser(minimal_dbc)
        # Voltage: 0|16@1+ scale=0.1, raw 4000 -> 400.0V
        data = bytes([0xA0, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        decoded = parser.decode(257, data)
        assert "Voltage" in decoded
        assert decoded["Voltage"] == pytest.approx(400.0, abs=0.1)

    def test_decode_signed(self, minimal_dbc):
        """Decoding signed signal (current, negative)."""
        parser = DBCParser(minimal_dbc)
        # Current: 16|16@1- scale=0.1, raw -1000 -> -100.0A
        # -1000 in 16-bit 2's complement = 0xFC18
        # bytes: 0x18, 0xFC in little-endian at byte 2,3
        data = bytes([0x00, 0x00, 0x18, 0xFC, 0x00, 0x00, 0x00, 0x00])
        decoded = parser.decode(257, data)
        assert "Current" in decoded
        assert decoded["Current"] == pytest.approx(-100.0, abs=0.1)

    def test_get_signal_value(self, minimal_dbc):
        """Named signal extraction works."""
        parser = DBCParser(minimal_dbc)
        data = bytes([0xA0, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
        val = parser.get_signal_value(257, data, "Voltage")
        assert val == pytest.approx(400.0, abs=0.1)

    def test_list_messages(self, minimal_dbc):
        """list_messages returns all parsed messages."""
        parser = DBCParser(minimal_dbc)
        msgs = parser.list_messages()
        assert len(msgs) == 1

    def test_decode_unknown_id(self, minimal_dbc):
        """Decoding unknown ID returns empty dict."""
        parser = DBCParser(minimal_dbc)
        assert parser.decode(999, b"") == {}


class TestBuiltinDBC:
    """Built-in battery DBC tests."""

    def test_builtin_has_standard_messages(self, builtin):
        """Built-in DBC has all expected messages."""
        expected_ids = [257, 258, 259, 65270, 65271, 65272, 65273]
        for eid in expected_ids:
            assert eid in builtin.messages, f"Missing message ID {eid}"

    def test_builtin_j1939_ids(self, builtin):
        """J1939 PGNs are parsed correctly (29-bit)."""
        msg = builtin.get_message(65270)
        assert msg is not None
        assert msg.is_extended is True

    def test_builtin_can2_ids(self, builtin):
        """CAN 2.0B messages are NOT extended."""
        msg = builtin.get_message(257)
        assert msg is not None
        assert msg.is_extended is False

    def test_builtin_comments(self, builtin):
        """Comments are parsed."""
        msg = builtin.get_message(257)
        assert msg.comment == "Battery voltage and current (CAN 2.0B 0x101)"

    def test_builtin_signal_comments(self, builtin):
        """Signal-level comments are parsed."""
        sig = builtin.get_signal(257, "Voltage")
        assert sig is not None
        assert sig.comment == "Battery pack voltage (0.1V resolution)"

    def test_builtin_decode_can2(self, builtin):
        """Decode CAN 2.0B VoltageCurrent message."""
        # Voltage = 396.5V -> raw 3965 = 0x0F7D
        # Current = 125.3A -> raw 1253 = 0x04E5
        data = bytes([0x7D, 0x0F, 0xE5, 0x04, 0x00, 0x00, 0x00, 0x00])
        decoded = builtin.decode(257, data)
        assert decoded.get("Voltage") == pytest.approx(396.5, abs=0.1)
        assert decoded.get("Current") == pytest.approx(125.3, abs=0.1)

    def test_builtin_decode_j1939(self, builtin):
        """Decode J1939 BatteryTemperature message."""
        # Cell temps: 35, 36, 37, 38°C
        data = bytes([35, 36, 37, 38, 0, 0, 0, 0])
        decoded = builtin.decode(65270, data)
        assert decoded.get("CellTemp_1") == 35.0
        assert decoded.get("CellTemp_2") == 36.0
        assert decoded.get("CellTemp_3") == 37.0
        assert decoded.get("CellTemp_4") == 38.0


class TestBatteryDBCContent:
    """battery_dbc_content integrity checks."""

    def test_content_is_valid_dbc(self):
        """battery_dbc_content can be parsed."""
        import tempfile
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False)
        tmp.write(battery_dbc_content())
        tmp.close()
        try:
            parser = DBCParser(tmp.name)
            assert len(parser.messages) == 7
        finally:
            os.unlink(tmp.name)

    def test_raw_to_physical(self):
        """Signal raw_to_physical conversion."""
        from ev_qa_framework.dbc_parser import Signal
        sig = Signal("Test", 0, 16, "Intel", False, 0.1, 0.0, 0, 1000, "V")
        assert sig.raw_to_physical(4000) == 400.0
        assert sig.raw_to_physical(0) == 0.0

    def test_physical_to_raw(self):
        """Signal physical_to_raw conversion."""
        from ev_qa_framework.dbc_parser import Signal
        sig = Signal("Test", 0, 16, "Intel", False, 0.1, 0.0, 0, 1000, "V")
        assert sig.physical_to_raw(400.0) == 4000
        assert sig.physical_to_raw(0.0) == 0
