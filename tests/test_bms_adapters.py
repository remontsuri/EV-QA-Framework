"""Tests for manufacturer-specific BMS adapters.

Covers Tesla, BYD, and Nio adapter decode functions, ensuring they
produce valid BMSTelemetry-compatible dicts from known CAN data bytes.
"""

from __future__ import annotations

import struct

import pytest

from ev_qa_framework.bms_adapters.base import (
    clamp,
    unpack_i16_be,
    unpack_i16_le,
    unpack_u8,
    unpack_u16_be,
    unpack_u16_le,
    unpack_u32_be,
)
from ev_qa_framework.bms_adapters.byd import (
    BYDBMSAdapter,
    decode_cells,
    decode_current_temp,
    decode_voltage,
)
from ev_qa_framework.bms_adapters.nio import (
    NioBMSAdapter,
    decode_current,
    decode_soh,
)
from ev_qa_framework.bms_adapters.nio import (
    decode_temperature as nio_decode_temperature,
)
from ev_qa_framework.bms_adapters.nio import (
    decode_voltage as nio_decode_voltage,
)
from ev_qa_framework.bms_adapters.tesla import (
    TeslaBMSAdapter,
    decode_cell_stats,
    decode_soc,
    decode_voltage_current,
)
from ev_qa_framework.bms_adapters.tesla import (
    decode_temperature as tesla_decode_temperature,
)

# ═══════════════════════════════════════════════════════════════════
# Base Helper Tests
# ═══════════════════════════════════════════════════════════════════


class TestBaseHelpers:
    def test_unpack_u8(self):
        assert unpack_u8(b"\x00") == 0
        assert unpack_u8(b"\xFF") == 255
        assert unpack_u8(b"\x80") == 128

    def test_unpack_u16_be(self):
        assert unpack_u16_be(b"\x00\x00") == 0
        assert unpack_u16_be(b"\xFF\xFF") == 65535
        assert unpack_u16_be(b"\x01\x00") == 256

    def test_unpack_u16_le(self):
        assert unpack_u16_le(b"\x00\x00") == 0
        assert unpack_u16_le(b"\xFF\xFF") == 65535
        assert unpack_u16_le(b"\x00\x01") == 256

    def test_unpack_i16_be(self):
        assert unpack_i16_be(b"\x00\x00") == 0
        assert unpack_i16_be(b"\xFF\xFF") == -1
        assert unpack_i16_be(b"\x80\x00") == -32768

    def test_unpack_i16_le(self):
        assert unpack_i16_le(b"\x00\x00") == 0
        assert unpack_i16_le(b"\xFF\xFF") == -1
        assert unpack_i16_le(b"\x00\x80") == -32768

    def test_unpack_u32_be(self):
        assert unpack_u32_be(b"\x00\x00\x00\x00") == 0
        assert unpack_u32_be(b"\xFF\xFF\xFF\xFF") == 0xFFFFFFFF

    def test_clamp(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(-1.0, 0.0, 10.0) == 0.0
        assert clamp(15.0, 0.0, 10.0) == 10.0
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0


# ═══════════════════════════════════════════════════════════════════
# Tesla Adapter Tests
# ═══════════════════════════════════════════════════════════════════


class TestTeslaDecodeVoltageCurrent:
    def test_normal_values(self):
        # 400.00V = 40000 = 0x9C40, 50.00A = 5000 = 0x1388
        data = struct.pack(">HH", 40000, 5000) + b"\x00" * 4
        v, i = decode_voltage_current(data)
        assert v == pytest.approx(400.0, abs=0.01)
        assert i == pytest.approx(50.0, abs=0.01)

    def test_negative_current(self):
        # -30.00A = -3000 as signed i16 = 0xF448
        data = struct.pack(">Hh", 35000, -3000) + b"\x00" * 4
        v, i = decode_voltage_current(data)
        assert v == pytest.approx(350.0, abs=0.01)
        assert i == pytest.approx(-30.0, abs=0.01)

    def test_zero_values(self):
        data = b"\x00" * 8
        v, i = decode_voltage_current(data)
        assert v == 0.0
        assert i == 0.0

    def test_max_values(self):
        # Max unsigned u16 = 65535 → 655.35V
        # Max signed i16 = 32767 → 327.67A
        data = struct.pack(">Hh", 65535, 32767)
        v, i = decode_voltage_current(data)
        assert v == pytest.approx(655.35, abs=0.01)
        assert i == pytest.approx(327.67, abs=0.01)


class TestTeslaDecodeSoc:
    def test_50_percent(self):
        # 50% SOC = 100 in 0.5% units
        data = bytes([100]) + b"\x00" * 7
        assert decode_soc(data) == 50.0

    def test_100_percent(self):
        data = bytes([200]) + b"\x00" * 7
        assert decode_soc(data) == 100.0

    def test_0_percent(self):
        data = bytes([0]) + b"\x00" * 7
        assert decode_soc(data) == 0.0

    def test_25_5_percent(self):
        # 25.5% = 51 in 0.5% units
        data = bytes([51]) + b"\x00" * 7
        assert decode_soc(data) == pytest.approx(25.5, abs=0.01)

    def test_max_byte_clamps(self):
        data = bytes([255]) + b"\x00" * 7
        # 255 * 0.5 = 127.5, clamped to 100
        assert decode_soc(data) == 100.0


class TestTeslaDecodeTemperature:
    def test_normal_values(self):
        # Max=45°C → byte 85, Min=20°C → byte 60, Avg=30°C → byte 70
        data = bytes([85, 60, 70]) + b"\x00" * 5
        t_max, t_min, t_avg = tesla_decode_temperature(data)
        assert t_max == 45.0
        assert t_min == 20.0
        assert t_avg == 30.0

    def test_offset_zero(self):
        # Byte 0 = -40°C, byte 40 = 0°C
        data = bytes([0, 40, 20]) + b"\x00" * 5
        t_max, t_min, t_avg = tesla_decode_temperature(data)
        assert t_max == -40.0
        assert t_min == 0.0
        assert t_avg == -20.0

    def test_max_temperature(self):
        # Byte 255 = 215°C
        data = bytes([255, 255, 255]) + b"\x00" * 5
        t_max, t_min, t_avg = tesla_decode_temperature(data)
        assert t_max == 215.0
        assert t_min == 215.0
        assert t_avg == 215.0


class TestTeslaDecodeCellStats:
    def test_normal_values(self):
        # min=3.200V → 3200, max=3.450V → 3450, delta=0.250V → 250
        data = struct.pack(">HHH", 3200, 3450, 250) + b"\x00" * 2
        v_min, v_max, delta = decode_cell_stats(data)
        assert v_min == pytest.approx(3.2, abs=0.001)
        assert v_max == pytest.approx(3.45, abs=0.001)
        assert delta == pytest.approx(0.25, abs=0.001)

    def test_zero_values(self):
        data = b"\x00" * 8
        v_min, v_max, delta = decode_cell_stats(data)
        assert v_min == 0.0
        assert v_max == 0.0
        assert delta == 0.0

    def test_max_values(self):
        data = struct.pack(">HHH", 65535, 65535, 65535) + b"\x00" * 2
        v_min, v_max, delta = decode_cell_stats(data)
        assert v_min == pytest.approx(65.535, abs=0.001)
        assert v_max == pytest.approx(65.535, abs=0.001)
        assert delta == pytest.approx(65.535, abs=0.001)


class TestTeslaAdapter:
    def test_not_connected_returns_empty(self):
        adapter = TeslaBMSAdapter()
        assert adapter.is_connected is False
        t = adapter.read_telemetry()
        assert t.protocol == "tesla_can"
        assert t.pack_voltage is None

    def test_health_check_disconnected(self):
        adapter = TeslaBMSAdapter()
        h = adapter.health_check()
        assert h["manufacturer"] == "tesla"
        assert h["connected"] is False
        assert h["status"] == "disconnected"

    def test_manufacturer_info(self):
        adapter = TeslaBMSAdapter()
        info = adapter.get_manufacturer_info()
        assert info["manufacturer"] == "tesla"
        assert "can_ids" in info

    def test_context_manager(self):
        adapter = TeslaBMSAdapter()
        with adapter:
            assert adapter.is_connected is False  # no CAN hardware


# ═══════════════════════════════════════════════════════════════════
# BYD Adapter Tests
# ═══════════════════════════════════════════════════════════════════


class TestBYDDecodeVoltage:
    def test_normal_value(self):
        # 400.0V = 4000 in 0.1V units
        data = struct.pack(">H", 4000) + b"\x00" * 6
        assert decode_voltage(data) == pytest.approx(400.0, abs=0.01)

    def test_zero(self):
        data = b"\x00" * 8
        assert decode_voltage(data) == 0.0

    def test_max(self):
        data = struct.pack(">H", 65535)
        assert decode_voltage(data) == pytest.approx(6553.5, abs=0.01)


class TestBYDDecodeCurrentTemp:
    def test_discharge_current(self):
        # 100.0A = 1000 in 0.1A units
        # Temp: 45, 20, 30 °C → bytes 85, 60, 70 (offset -40)
        data = struct.pack(">H", 1000) + bytes([85, 60, 70]) + b"\x00" * 3
        current, t_max, t_min, t_avg = decode_current_temp(data)
        assert current == pytest.approx(100.0, abs=0.01)
        assert t_max == 45.0
        assert t_min == 20.0
        assert t_avg == 30.0

    def test_charge_current(self):
        # -50.0A = -500 as signed → 0xFE0C in u16
        raw = struct.pack(">H", 0xFE0C)
        data = raw + bytes([40, 40, 40]) + b"\x00" * 3
        current, t_max, t_min, t_avg = decode_current_temp(data)
        assert current == pytest.approx(-50.0, abs=0.01)

    def test_zero_current(self):
        data = struct.pack(">H", 0) + bytes([40, 40, 40]) + b"\x00" * 3
        current, t_max, t_min, t_avg = decode_current_temp(data)
        assert current == 0.0

    def test_all_zeros(self):
        data = b"\x00" * 8
        current, t_max, t_min, t_avg = decode_current_temp(data)
        assert current == 0.0
        assert t_max == -40.0
        assert t_min == -40.0
        assert t_avg == -40.0


class TestBYDDecodeCells:
    def test_four_cells(self):
        # 3.200V, 3.201V, 3.202V, 3.203V
        data = struct.pack(">HHHH", 3200, 3201, 3202, 3203)
        cells = decode_cells(data)
        assert len(cells) == 4
        assert cells[0] == pytest.approx(3.200, abs=0.001)
        assert cells[1] == pytest.approx(3.201, abs=0.001)
        assert cells[2] == pytest.approx(3.202, abs=0.001)
        assert cells[3] == pytest.approx(3.203, abs=0.001)

    def test_partial_data(self):
        # Only 2 cells in 4 bytes
        data = struct.pack(">HH", 3300, 3400)
        cells = decode_cells(data)
        assert len(cells) == 2

    def test_empty_data(self):
        cells = decode_cells(b"")
        assert cells == []

    def test_zero_cells(self):
        data = b"\x00" * 8
        cells = decode_cells(data)
        assert all(c == 0.0 for c in cells)


class TestBYDAdapter:
    def test_not_connected_returns_empty(self):
        adapter = BYDBMSAdapter()
        assert adapter.is_connected is False
        t = adapter.read_telemetry()
        assert t.protocol == "byd_can"
        assert t.pack_voltage is None

    def test_health_check_disconnected(self):
        adapter = BYDBMSAdapter()
        h = adapter.health_check()
        assert h["manufacturer"] == "byd"
        assert h["connected"] is False

    def test_manufacturer_info(self):
        adapter = BYDBMSAdapter()
        info = adapter.get_manufacturer_info()
        assert info["manufacturer"] == "byd"


# ═══════════════════════════════════════════════════════════════════
# Nio Adapter Tests
# ═══════════════════════════════════════════════════════════════════


class TestNioDecodeVoltage:
    def test_normal_value(self):
        # 400.0V = 4000 in 0.1V units, little-endian
        data = struct.pack("<H", 4000) + b"\x00" * 6
        assert nio_decode_voltage(data) == pytest.approx(400.0, abs=0.01)

    def test_zero(self):
        data = b"\x00" * 8
        assert nio_decode_voltage(data) == 0.0

    def test_max(self):
        data = struct.pack("<H", 65535)
        assert nio_decode_voltage(data) == pytest.approx(6553.5, abs=0.01)


class TestNioDecodeCurrent:
    def test_discharge(self):
        # 100.0A = 1000 in 0.1A units, little-endian
        data = struct.pack("<h", 1000) + b"\x00" * 6
        assert decode_current(data) == pytest.approx(100.0, abs=0.01)

    def test_charge(self):
        # -50.0A
        data = struct.pack("<h", -500) + b"\x00" * 6
        assert decode_current(data) == pytest.approx(-50.0, abs=0.01)

    def test_zero(self):
        data = b"\x00" * 8
        assert decode_current(data) == 0.0


class TestNioDecodeTemperature:
    def test_normal_values(self):
        data = bytes([85, 60, 70]) + b"\x00" * 5
        t_max, t_min, t_avg = nio_decode_temperature(data)
        assert t_max == 45.0
        assert t_min == 20.0
        assert t_avg == 30.0

    def test_all_zeros(self):
        data = b"\x00" * 8
        t_max, t_min, t_avg = nio_decode_temperature(data)
        assert t_max == -40.0
        assert t_min == -40.0
        assert t_avg == -40.0

    def test_max_temperature(self):
        data = bytes([255, 255, 255]) + b"\x00" * 5
        t_max, t_min, t_avg = nio_decode_temperature(data)
        assert t_max == 215.0


class TestNioDecodeSoh:
    def test_normal_value(self):
        data = bytes([95]) + b"\x00" * 7
        assert decode_soh(data) == 95.0

    def test_zero(self):
        data = bytes([0]) + b"\x00" * 7
        assert decode_soh(data) == 0.0

    def test_100_percent(self):
        data = bytes([100]) + b"\x00" * 7
        assert decode_soh(data) == 100.0

    def test_max_byte(self):
        data = bytes([255]) + b"\x00" * 7
        # 255 clamped to 100
        assert decode_soh(data) == 100.0


class TestNioAdapter:
    def test_not_connected_returns_empty(self):
        adapter = NioBMSAdapter()
        assert adapter.is_connected is False
        t = adapter.read_telemetry()
        assert t.protocol == "nio_can"
        assert t.pack_voltage is None

    def test_health_check_disconnected(self):
        adapter = NioBMSAdapter()
        h = adapter.health_check()
        assert h["manufacturer"] == "nio"
        assert h["connected"] is False

    def test_manufacturer_info(self):
        adapter = NioBMSAdapter()
        info = adapter.get_manufacturer_info()
        assert info["manufacturer"] == "nio"


# ═══════════════════════════════════════════════════════════════════
# BMSTelemetry Compatibility Tests
# ═══════════════════════════════════════════════════════════════════


class TestBMSTelemetryCompatibility:
    """Ensure all decode outputs can populate a valid BMSTelemetry dict."""

    def test_tesla_decode_populates_telemetry(self):
        from ev_qa_framework.bms_protocol import BMSTelemetry

        v, i = decode_voltage_current(struct.pack(">HH", 40000, 5000) + b"\x00" * 4)
        soc = decode_soc(bytes([160]) + b"\x00" * 7)
        t_max, t_min, t_avg = tesla_decode_temperature(
            bytes([85, 60, 70]) + b"\x00" * 5
        )
        v_min, v_max, delta = decode_cell_stats(
            struct.pack(">HHH", 3200, 3450, 250) + b"\x00" * 2
        )

        t = BMSTelemetry(
            pack_voltage=v,
            pack_current=i,
            soc=soc,
            temperature_max=t_max,
            temperature_min=t_min,
            temperature_avg=t_avg,
            cell_voltage_min=v_min,
            cell_voltage_max=v_max,
            cell_voltage_delta=delta,
            protocol="tesla_can",
        )
        d = t.to_dict()
        assert d["pack_voltage"] == pytest.approx(400.0, abs=0.01)
        assert d["pack_current"] == pytest.approx(50.0, abs=0.01)
        assert d["soc"] == 80.0
        assert d["temperature_max"] == 45.0

    def test_byd_decode_populates_telemetry(self):
        from ev_qa_framework.bms_protocol import BMSTelemetry

        v = decode_voltage(struct.pack(">H", 4000) + b"\x00" * 6)
        current, t_max, t_min, t_avg = decode_current_temp(
            struct.pack(">H", 1000) + bytes([85, 60, 70]) + b"\x00" * 3
        )
        cells = decode_cells(struct.pack(">HHHH", 3200, 3201, 3202, 3203))

        t = BMSTelemetry(
            pack_voltage=v,
            pack_current=current,
            temperature_max=t_max,
            temperature_min=t_min,
            temperature_avg=t_avg,
            cell_voltages=cells,
            cell_voltage_min=min(cells),
            cell_voltage_max=max(cells),
            cell_voltage_delta=max(cells) - min(cells),
            protocol="byd_can",
        )
        d = t.to_dict()
        assert d["pack_voltage"] == pytest.approx(400.0, abs=0.01)
        assert d["pack_current"] == pytest.approx(100.0, abs=0.01)
        assert len(d["cell_voltages"]) == 4

    def test_nio_decode_populates_telemetry(self):
        from ev_qa_framework.bms_protocol import BMSTelemetry

        v = nio_decode_voltage(struct.pack("<H", 4000) + b"\x00" * 6)
        i = decode_current(struct.pack("<h", 1000) + b"\x00" * 6)
        t_max, t_min, t_avg = nio_decode_temperature(
            bytes([85, 60, 70]) + b"\x00" * 5
        )
        soh = decode_soh(bytes([95]) + b"\x00" * 7)

        t = BMSTelemetry(
            pack_voltage=v,
            pack_current=i,
            soh=soh,
            temperature_max=t_max,
            temperature_min=t_min,
            temperature_avg=t_avg,
            protocol="nio_can",
        )
        d = t.to_dict()
        assert d["pack_voltage"] == pytest.approx(400.0, abs=0.01)
        assert d["pack_current"] == pytest.approx(100.0, abs=0.01)
        assert d["soh"] == 95.0


# ═══════════════════════════════════════════════════════════════════
# Lazy Import Tests
# ═══════════════════════════════════════════════════════════════════


class TestLazyImports:
    def test_import_no_can(self):
        """Adapters import successfully without python-can installed."""
        from ev_qa_framework.bms_adapters import (
            BYDBMSAdapter,
            NioBMSAdapter,
            TeslaBMSAdapter,
        )

        assert TeslaBMSAdapter is not None
        assert BYDBMSAdapter is not None
        assert NioBMSAdapter is not None

    def test_top_level_lazy_import(self):
        """Adapters are accessible via top-level lazy imports."""
        import ev_qa_framework as ev

        assert ev.TeslaBMSAdapter is not None
        assert ev.BYDBMSAdapter is not None
        assert ev.NioBMSAdapter is not None
