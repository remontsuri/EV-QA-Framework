"""
DBC File Parser — import Vector CANdb (.dbc) definitions.

Supports CAN 2.0B and J1939 (29-bit extended) message definitions.
Compatible with DBC files exported from Vector CANdb++, SavvyCAN, and BUSMASTER.

Usage:
    dbc = DBCParser("path/to/file.dbc")
    msg_def = dbc.get_message(0x101)
    decoded = dbc.decode(can_id=0x101, data=b"...")
    print(dbc.get_signal_value(0x101, b"...", "Voltage"))

DBC format: https://vector.com/candb-format
"""

import re
import math
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """A single CAN signal defined in a DBC file."""

    name: str
    start_bit: int
    length: int
    byte_order: str          # 'Intel' (little-endian) or 'Motorola' (big-endian)
    signed: bool
    scale: float
    offset: float
    min_val: float
    max_val: float
    unit: str
    receiver: List[str] = field(default_factory=list)
    comment: str = ""

    def raw_to_physical(self, raw: int) -> float:
        """Convert raw integer value to physical (engineering) value."""
        return raw * self.scale + self.offset

    def physical_to_raw(self, physical: float) -> int:
        """Convert physical value back to raw integer."""
        return int(round((physical - self.offset) / self.scale))


@dataclass
class Message:
    """A CAN message frame definition from DBC."""

    id: int
    name: str
    dlc: int                # data length code (bytes)
    transmitter: str
    signals: Dict[str, Signal] = field(default_factory=dict)
    comment: str = ""
    is_extended: bool = False   # True = 29-bit (J1939 style)


# ---------------------------------------------------------------------------
# DBC parser
# ---------------------------------------------------------------------------

class DBCParser:
    """
    Parse a Vector CANdb (.dbc) file and provide message/signal lookups.

    Thread-safe after construction.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.messages: Dict[int, Message] = {}      # keyed by CAN ID
        self._by_name: Dict[str, Message] = {}      # keyed by message name
        self.version: str = ""
        self.comments: Dict[str, str] = {}           # node->comment, etc.
        self._raw = ""

        with open(filepath, "r", encoding="latin-1") as f:
            self._raw = f.read()

        self._parse()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_message(self, can_id: int) -> Optional[Message]:
        """Look up a message definition by its CAN ID."""
        return self.messages.get(can_id)

    def get_message_by_name(self, name: str) -> Optional[Message]:
        """Look up a message by its symbolic name."""
        return self._by_name.get(name)

    def list_messages(self) -> List[Message]:
        """Return all parsed messages."""
        return list(self.messages.values())

    def get_signal(
        self, can_id: int, signal_name: str
    ) -> Optional[Signal]:
        """Look up a specific signal within a message."""
        msg = self.messages.get(can_id)
        if msg is None:
            return None
        return msg.signals.get(signal_name)

    def decode(
        self, can_id: int, data: bytes
    ) -> Dict[str, float]:
        """
        Decode raw CAN data bytes into physical signal values.

        Parameters
        ----------
        can_id : int
            CAN arbitration ID (11 or 29 bit).
        data : bytes
            Exactly 8 bytes of CAN payload (or less for short frames).

        Returns
        -------
        dict mapping signal name -> physical value.
        """
        msg = self.messages.get(can_id)
        if msg is None:
            return {}

        result: Dict[str, float] = {}
        for sig_name, sig in msg.signals.items():
            raw = self._extract_raw(data, sig)
            result[sig_name] = sig.raw_to_physical(raw)
        return result

    def get_signal_value(
        self, can_id: int, data: bytes, signal_name: str
    ) -> Optional[float]:
        """Decode a single named signal from raw CAN data."""
        msg = self.messages.get(can_id)
        if msg is None:
            return None
        sig = msg.signals.get(signal_name)
        if sig is None:
            return None
        raw = self._extract_raw(data, sig)
        return sig.raw_to_physical(raw)

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _parse(self):
        lines = self._raw.splitlines()

        # Pass 1: collect multi-line comments (CM_)
        # We'll handle CM_ after primary definitions.

        # Pass 2: parse BO_, SG_, and related
        for i, line in enumerate(lines):
            stripped = line.strip()

            if stripped.startswith("VERSION"):
                self.version = stripped.split('"')[1] if '"' in stripped else ""

            elif stripped.startswith("BO_ "):
                self._parse_message(line, lines, i)

        # Pass 3: parse CM_ (comments)
        self._parse_comments(lines)

    # ------------------------------------------------------------------
    def _parse_message(self, line: str, all_lines: List[str], lineno: int):
        """
        Parse a BO_ line and its following SG_ (signal) lines.

        BO_ format:
          BO_ <id> <name>: <dlc> <transmitter>
          SG_ <name> <start_bit>|<length>@<byte_order>+<sign> (<scale>,<offset>) [<min>|<max>] "<unit>" <receivers>
        """
        # BO_ 101 VoltageMessage: 8 TransmitterName
        pattern = r"BO_\s+(\d+)\s+(\S+)\s*:\s*(\d+)\s+(\S+)"
        m = re.match(pattern, line)
        if not m:
            return

        can_id = int(m.group(1))
        msg_name = m.group(2)
        dlc = int(m.group(3))
        transmitter = m.group(4)

        is_extended = can_id > 0x7FF   # 29-bit if > 11-bit max

        msg = Message(
            id=can_id,
            name=msg_name,
            dlc=dlc,
            transmitter=transmitter,
            is_extended=is_extended,
        )

        # Collect SG_ lines that follow this BO_
        idx = lineno + 1
        while idx < len(all_lines):
            s = all_lines[idx].strip()
            if s.startswith("SG_ "):
                sig = self._parse_signal(s)
                if sig is not None:
                    msg.signals[sig.name] = sig
            elif s.startswith("BO_ "):
                break   # next message starts
            idx += 1

        self.messages[can_id] = msg
        self._by_name[msg_name] = msg

    # ------------------------------------------------------------------
    def _parse_signal(self, line: str) -> Optional[Signal]:
        """
        Parse a single SG_ line.

        Examples:
          SG_ Voltage : 0|16@1+ (0.1,0) [0|1000] "V" Vector__XXX
          SG_ Temperature m0 : 8|8@1+ (1,0) [-40|150] "degC" Vector__XXX

        Note: multiplexed signals (m0, m1) are parsed but the multiplexor
        info is not used for decoding — user must pick the right message ID.
        """
        # Strip multiplexor indicator like "m0" after the name
        # SG_ Name m0 : ...
        line_clean = re.sub(r"\s+m\d+\s*:", ":", line)

        pattern = (
            r"SG_\s+"
            r"(\S+)\s*:\s*"              # signal name
            r"(\d+)\|(\d+)@(\d)([\+\-])\s*"  # start|len@byteorder+sign
            r"\(([^,]+),([^)]+)\)\s*"     # (scale, offset)
            r"\[([^|]*)\|([^\]]*)\]\s*"   # [min|max]
            r'"([^"]*)"\s*'               # "unit"
            r"(\S*)"                       # receivers (optional)
        )
        m = re.match(pattern, line_clean)
        if not m:
            # Try without [min|max] (some DBCs omit)
            pattern2 = (
                r"SG_\s+"
                r"(\S+)\s*:\s*"
                r"(\d+)\|(\d+)@(\d)([\+\-])\s*"
                r"\(([^,]+),([^)]+)\)\s*"
                r'"([^"]*)"\s*'
                r"(\S*)"
            )
            m = re.match(pattern2, line_clean)
            if not m:
                return None

            name = m.group(1)
            start_bit = int(m.group(2))
            length = int(m.group(3))
            byte_order = "Intel" if m.group(4) == "1" else "Motorola"
            signed = m.group(5) == "-"
            scale = float(m.group(6))
            offset = float(m.group(7))
            unit = m.group(8) if len(m.groups()) >= 8 else ""
            receivers = m.group(9).split() if len(m.groups()) >= 9 and m.group(9) else []
            min_val = 0.0
            max_val = 0.0
        else:
            name = m.group(1)
            start_bit = int(m.group(2))
            length = int(m.group(3))
            byte_order = "Intel" if m.group(4) == "1" else "Motorola"
            signed = m.group(5) == "-"
            scale = float(m.group(6))
            offset = float(m.group(7))
            min_val_str = m.group(8)
            max_val_str = m.group(9)
            unit = m.group(10)
            receivers = m.group(11).split() if m.group(11) else []

            try:
                min_val = float(min_val_str) if min_val_str else 0.0
            except ValueError:
                min_val = 0.0
            try:
                max_val = float(max_val_str) if max_val_str else 0.0
            except ValueError:
                max_val = 0.0

        return Signal(
            name=name,
            start_bit=start_bit,
            length=length,
            byte_order=byte_order,
            signed=signed,
            scale=scale,
            offset=offset,
            min_val=min_val,
            max_val=max_val,
            unit=unit,
            receiver=receivers,
        )

    # ------------------------------------------------------------------
    def _parse_comments(self, lines: List[str]):
        """Parse CM_ entries for message and signal comments."""
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("CM_ "):
                continue

            # CM_ BO_ <id> "<comment>";
            m = re.match(r'CM_\s+BO_\s+(\d+)\s+"([^"]*)"\s*;', stripped)
            if m:
                can_id = int(m.group(1))
                comment = m.group(2)
                if can_id in self.messages:
                    self.messages[can_id].comment = comment
                continue

            # CM_ SG_ <id> <signal_name> "<comment>";
            m = re.match(
                r'CM_\s+SG_\s+(\d+)\s+(\S+)\s+"([^"]*)"\s*;', stripped
            )
            if m:
                can_id = int(m.group(1))
                sig_name = m.group(2)
                comment = m.group(3)
                msg = self.messages.get(can_id)
                if msg and sig_name in msg.signals:
                    msg.signals[sig_name].comment = comment

    # ------------------------------------------------------------------
    # Raw value extraction from CAN data
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_raw(data: bytes, sig: Signal) -> int:
        """
        Extract an integer value from CAN data according to signal layout.

        Supports Intel (little-endian) and Motorola (big-endian) byte orders.
        Handles跨-byte and跨-byte-boundary signals.
        """
        raw = 0
        if sig.byte_order == "Intel":
            raw = DBCParser._extract_intel(data, sig.start_bit, sig.length)
        else:
            raw = DBCParser._extract_motorola(data, sig.start_bit, sig.length)

        if sig.signed and (raw >> (sig.length - 1)) & 1:
            raw -= 1 << sig.length

        return raw

    @staticmethod
    def _extract_intel(data: bytes, start_bit: int, length: int) -> int:
        """
        Intel (little-endian): bits are numbered from LSB.
        start_bit is the LSB of the signal.
        """
        value = 0
        for i in range(length):
            bit_pos = start_bit + i
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            if byte_idx < len(data):
                if (data[byte_idx] >> bit_in_byte) & 1:
                    value |= 1 << i
        return value

    @staticmethod
    def _extract_motorola(data: bytes, start_bit: int, length: int) -> int:
        """
        Motorola (big-endian): start_bit is the MSB of the signal.
        Bits count downward in the byte, then jump to the previous byte.
        """
        value = 0
        for i in range(length):
            bit_pos = start_bit - i
            byte_idx = bit_pos // 8
            bit_in_byte = bit_pos % 8
            if byte_idx < len(data):
                if (data[byte_idx] >> bit_in_byte) & 1:
                    value |= 1 << i
        return value


# ---------------------------------------------------------------------------
# Convenience: built-in battery DBC
# ---------------------------------------------------------------------------

def battery_dbc_content() -> str:
    """
    Return a DBC string describing the standard battery telemetry messages
    used by this framework (CAN 2.0B + J1939).
    """
    return '''VERSION "EV-QA-Framework v1.0"

BS_:

BU_: EV_BMS

BO_ 257 VoltageCurrent: 8 EV_BMS
 SG_ Voltage : 0|16@1+ (0.1,0) [0|1000] "V" EV_BMS
 SG_ Current : 16|16@1- (0.1,0) [-500|500] "A" EV_BMS

BO_ 258 TempSOC: 8 EV_BMS
 SG_ Temperature : 0|8@1+ (1,-40) [-40|150] "degC" EV_BMS
 SG_ SOC : 8|8@1+ (1,0) [0|100] "%" EV_BMS

BO_ 259 SOH: 8 EV_BMS
 SG_ StateOfHealth : 0|8@1+ (1,0) [0|100] "%" EV_BMS

BO_ 65270 BatteryTemp: 8 EV_BMS
 SG_ CellTemp_1 : 0|8@1+ (1,0) [0|150] "degC" EV_BMS
 SG_ CellTemp_2 : 8|8@1+ (1,0) [0|150] "degC" EV_BMS
 SG_ CellTemp_3 : 16|8@1+ (1,0) [0|150] "degC" EV_BMS
 SG_ CellTemp_4 : 24|8@1+ (1,0) [0|150] "degC" EV_BMS

BO_ 65271 BatteryVoltage: 8 EV_BMS
 SG_ PackVoltage : 0|16@1+ (0.01,0) [0|1000] "V" EV_BMS
 SG_ MinCellVoltage : 16|16@1+ (0.001,0) [0|5] "V" EV_BMS
 SG_ MaxCellVoltage : 32|16@1+ (0.001,0) [0|5] "V" EV_BMS

BO_ 65272 BatteryCurrent: 8 EV_BMS
 SG_ PackCurrent : 0|16@1- (0.1,0) [-1000|1000] "A" EV_BMS

BO_ 65273 BatterySOC: 8 EV_BMS
 SG_ SOC : 0|8@1+ (1,0) [0|100] "%" EV_BMS
 SG_ SOH : 8|8@1+ (1,0) [0|100] "%" EV_BMS

CM_ BO_ 257 "Battery voltage and current (CAN 2.0B 0x101)";
CM_ BO_ 258 "Battery temperature and SOC (CAN 2.0B 0x102)";
CM_ BO_ 259 "Battery state of health";
CM_ BO_ 65270 "J1939 PGN 0xFEF6: individual cell temperatures";
CM_ BO_ 65271 "J1939 PGN 0xFEF7: pack and cell voltages";
CM_ BO_ 65272 "J1939 PGN 0xFEF8: pack current";
CM_ BO_ 65273 "J1939 PGN 0xFEF9: SOC and SOH";
CM_ SG_ 257 Voltage "Battery pack voltage (0.1V resolution)";
CM_ SG_ 258 Temperature "Battery temperature in Celsius";
CM_ SG_ 65270 CellTemp_1 "Cell 1 temperature";
'''

def builtin_dbc() -> DBCParser:
    """Return a DBCParser loaded with the built-in battery definition."""
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".dbc", delete=False)
    tmp.write(battery_dbc_content())
    tmp.close()
    parser = DBCParser(tmp.name)
    os.unlink(tmp.name)
    return parser
