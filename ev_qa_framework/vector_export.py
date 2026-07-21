"""Vector CANoe/CANalyzer export module.

Provides export and import of CAN traces and test results in formats
readable by Vector tools (ASC, BLF, test vector CSV).
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Try to import python-can BLFWriter
try:
    from can.io import BLFWriter

    _BLF_AVAILABLE = True
except ImportError:
    _BLF_AVAILABLE = False


class VectorExporter:
    """Export and import CAN traces in Vector CANoe/CANalyzer formats."""

    def export_asc(
        self,
        trace_data: list[dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Export CAN trace to ASC (ASCII) format.

        Each line: ``<channel> <timestamp> <can_id> <data_hex>``

        Args:
            trace_data: List of dicts with keys:
                - timestamp (float): seconds since epoch
                - can_id (int): CAN arbitration ID
                - data (bytes): CAN payload
                - channel (int, optional): CAN channel (default 0)
            output_path: Destination file path.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)
        lines: list[str] = []

        # ASC header
        lines.append("date " + datetime.now().strftime("%a %b %d %I:%M:%S.%f %p %Y"))
        lines.append("base hex  timestamps absolute")
        lines.append("internal events log")
        lines.append("no direction")

        for msg in trace_data:
            ts = msg.get("timestamp", 0.0)
            can_id = msg.get("can_id", 0)
            data = msg.get("data", b"")
            channel = msg.get("channel", 0)

            data_hex = data.hex().upper()
            # Pad hex to even length if needed
            if len(data_hex) % 2 != 0:
                data_hex = "0" + data_hex

            # Format: channel timestamp_hex can_id dlc data_hex
            ts_hex = f"{int(ts * 1_000_000):08X}"
            lines.append(
                f"  {channel}  {ts_hex}  {can_id:03X}  {len(data)}  {data_hex}"
            )

        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Exported %d messages to ASC: %s", len(trace_data), output_path)
        return output_path

    def export_blf(
        self,
        trace_data: list[dict[str, Any]],
        output_path: str | Path,
    ) -> Path:
        """Export CAN trace to BLF format if python-can is available.

        Falls back to ASC format if BLFWriter is not installed.

        Args:
            trace_data: List of dicts (same format as export_asc).
            output_path: Destination file path (.blf or .asc).

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)

        if not _BLF_AVAILABLE:
            logger.warning(
                "python-can BLFWriter not available; falling back to ASC format"
            )
            fallback_path = output_path.with_suffix(".asc")
            return self.export_asc(trace_data, fallback_path)

        # Write BLF using python-can
        from can import Message as CANMessage

        with BLFWriter(str(output_path)) as writer:
            for msg in trace_data:
                can_msg = CANMessage(
                    timestamp=msg.get("timestamp", 0.0),
                    arbitration_id=msg.get("can_id", 0),
                    data=msg.get("data", b""),
                    channel=msg.get("channel", 0),
                    is_extended_id=False,
                )
                writer.on_message_received(can_msg)

        logger.info("Exported %d messages to BLF: %s", len(trace_data), output_path)
        return output_path

    def export_test_vector(
        self,
        results: dict[str, Any],
        output_path: str | Path,
    ) -> Path:
        """Export test results as Vector test vector CSV format.

        Args:
            results: Test results dictionary. Expected structure:
                {
                    "test_name": str,
                    "test_id": str,
                    "status": "PASS" | "FAIL" | "SKIP" | "ERROR",
                    "timestamp": str (ISO format),
                    "duration_ms": float,
                    "details": str (optional),
                    "measurements": [
                        {"name": str, "value": Any, "unit": str, "limit_min": float, "limit_max": float}
                    ]
                }
            output_path: Destination CSV file path.

        Returns:
            Path to the written file.
        """
        output_path = Path(output_path)

        test_name = results.get("test_name", "unknown")
        test_id = results.get("test_id", "")
        status = results.get("status", "UNKNOWN")
        timestamp = results.get("timestamp", datetime.now().isoformat())
        duration_ms = results.get("duration_ms", 0.0)
        details = results.get("details", "")
        measurements = results.get("measurements", [])

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Vector test vector header
            writer.writerow(["Test Vector Export"])
            writer.writerow(["Test Name", test_name])
            writer.writerow(["Test ID", test_id])
            writer.writerow(["Status", status])
            writer.writerow(["Timestamp", timestamp])
            writer.writerow(["Duration (ms)", f"{duration_ms:.2f}"])
            writer.writerow(["Details", details])
            writer.writerow([])

            # Measurements section
            if measurements:
                writer.writerow(["Measurements"])
                writer.writerow(["Name", "Value", "Unit", "Limit Min", "Limit Max"])
                for m in measurements:
                    writer.writerow([
                        m.get("name", ""),
                        m.get("value", ""),
                        m.get("unit", ""),
                        m.get("limit_min", ""),
                        m.get("limit_max", ""),
                    ])

        logger.info("Exported test vector to: %s", output_path)
        return output_path

    def import_asc(self, file_path: str | Path) -> list[dict[str, Any]]:
        """Import an ASC file into a list of message dicts.

        Args:
            file_path: Path to the ASC file.

        Returns:
            List of dicts with keys: timestamp, can_id, data, channel.
        """
        file_path = Path(file_path)
        messages: list[dict[str, Any]] = []

        lines = file_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("date ") or line.startswith("base ") or line.startswith("internal ") or line.startswith("no "):
                continue

            # Try to parse a data line: channel timestamp_hex can_id dlc data_hex
            parts = line.split()
            if len(parts) >= 4:
                try:
                    channel = int(parts[0])
                    ts_hex = parts[1]
                    can_id = int(parts[2], 16)
                    dlc = int(parts[3])
                    data_hex = parts[4] if len(parts) > 4 else ""

                    # Convert hex timestamp (microseconds) to seconds
                    ts = int(ts_hex, 16) / 1_000_000.0

                    # Convert hex data to bytes
                    data = bytes.fromhex(data_hex[: dlc * 2]) if data_hex else b""

                    messages.append({
                        "timestamp": ts,
                        "can_id": can_id,
                        "data": data,
                        "channel": channel,
                    })
                except (ValueError, IndexError):
                    # Skip unparseable lines
                    logger.debug("Skipping unparseable ASC line: %s", line)
                    continue

        logger.info("Imported %d messages from ASC: %s", len(messages), file_path)
        return messages
