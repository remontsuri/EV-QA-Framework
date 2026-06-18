# EV Battery Safety Standards — Research Summary

**Document Version:** 1.0  
**Date:** 2026-06-13  
**Project:** EV-QA-Framework  
**Target File:** `/opt/data/EV-QA-Framework/research/battery_standards.md`

---

## Overview

This document summarizes the five key international and regional safety standards governing EV (lithium-ion) batteries. Each standard addresses different aspects: transportation, cell-level performance, abuse tolerance, regional compliance, and system-level certification.

| Standard | Scope | Jurisdiction | Primary Focus |
|----------|-------|--------------|---------------|
| **UN 38.3** | Transport (air/sea/land) | Global (UN) | Transportation safety classification |
| **IEC 62660** | Cell performance & reliability | International (IEC) | Cell-level cycle life, power, safety |
| **SAE J2464** | Abuse testing | North America (SAE) | Mechanical/electrical/thermal abuse |
| **GB 38031** | Vehicle-level safety | China (SAC) | Mandatory Chinese market access |
| **UL 2580** | System certification | North America (UL) | Pack/system safety for EVs |

---

## 1. UN 38.3 — Lithium Battery Transport Testing

### Full Title
*UN Manual of Tests and Criteria, Part III, Subsection 38.3*

### Jurisdiction & Adoption
- **Global mandate** for air transport (IATA/ICAO), sea (IMDG), road (ADR), rail (RID)
- Legal requirement in 180+ UN member states
- Current revision: **Rev. 7 (2021)**, Amendment 1 (2023)

### Test Matrix (T.1–T.8)
| Test | Code | Description | Pass Criterion |
|------|------|-------------|----------------|
| T.1 | Altitude | 11.6 kPa, 6h, 20±5°C | No mass loss >0.1%, no leakage, no venting, no fire, OCV ≥90% |
| T.2 | Thermal | 75°C (6h) → -40°C (6h), 10 cycles | Same as T.1 |
| T.3 | Vibration | 7–200 Hz, 1g RMS, 3h/axis | Same + no rupture |
| T.4 | Shock | 150g/6ms or 50g/11ms, 3×/axis | Same |
| T.5 | External Short | 100mΩ max, 20°C, 1h after temp stabilizes | Temp ≤170°C, no fire, no rupture (6h obs) |
| T.6 | Impact/Crush | Cell: 9.1kg/61cm impact; Pack: 13kN crush | Temp ≤170°C, no fire (6h obs) |
| T.7 | Overcharge | 2× rated current, 24h (cells) | No fire, no rupture (7 days obs) |
| T.8 | Forced Discharge | 1C to 0V, then reversal | No fire (7 days obs) |

### Key Requirements
- **Test report** required per battery design (not per shipment)
- **38.3.3.2.1**: Summary must include test lab, date, all T.1–T.8 results
- **38.3.3.1**: Re-test triggered by: chemistry change, >20% mass change, design change affecting safety
- **Classification**: Proper shipping names UN 3480 (Li-ion), UN 3481 (in equipment), UN 3171 (vehicles)

### Practical Implications for EV-QA
- Must verify **every cell chemistry** and **pack design** passes full T.1–T.8 before shipment
- Test lab must be ISO 17025 accredited
- Document control: keep test reports 10+ years per IATA
- State of charge for air transport: **≤30% SOC** (UN 38.3.3.1.f)

---

## 2. IEC 62660 — Secondary Li-ion Cells for EV Propulsion

### Standard Family
| Part | Title | Focus |
|------|-------|-------|
| **IEC 62660-1** | Performance testing | Cycle life, power, calendar life |
| **IEC 62660-2** | Reliability & abuse | Thermal, electrical, mechanical abuse |
| **IEC 62660-3** | Safety requirements | Cell-level safety criteria |

*(Note: IEC 62660 is being superseded by **IEC 62619** for industrial/ESS and **ISO 12405** for vehicles, but 62660 remains widely referenced in legacy specs.)*

### IEC 62660-1: Performance Tests
| Test | Condition | Pass Criterion |
|------|-----------|----------------|
| **Rate capability** | C/3, 1C, 2C, 3C charge/discharge @ 25°C | Capacity ≥90% of 0.3C rating at 1C |
| **Cycle life** | 1C/1C @ 25°C, 80% DOD | ≥1000 cycles to 80% retention |
| **High-temp life** | 1C/1C @ 45°C | ≥300 cycles to 80% retention |
| **Low-temp power** | -20°C, pulse 10s @ 50% SOC | Power ≥50% of 25°C baseline |
| **Calendar life** | 100% SOC @ 45°C or 60°C | Capacity fade ≤20% after 1yr equivalent |

### IEC 62660-2: Reliability / Abuse
| Test | Method | Pass |
|------|--------|------|
| External short | 100mΩ, 60min or temp peak | No fire/explosion |
| Overcharge | 1C to 1.5×Vmax or 5h | No fire/explosion |
| Forced discharge | 0.2C to 0V | No fire/explosion |
| Crush | 13kN flat plate | No fire/explosion |
| Impact | 9.1kg/61cm | No fire/explosion |
| Thermal stability | 130°C/10min | No fire/explosion |
| Temperature cycling | -40→85°C, 5 cycles | No leakage, ≤5% cap drop |

### IEC 62660-3: Safety Requirements
- **Cell-level**: No fire, no explosion, no rupture in any abuse test
- **Thermal runaway** onset temperature must be documented
- **Gas composition** analysis recommended (CO, HF, H2, etc.)

### Practical Implications
- **Cell cert level**: Required by OEMs for cell qualification (PPAP Level 3)
- **Test duration**: ~18–24 months for full cycle/calendar life
- **Sampling**: n=30 cells min for statistical significance
- **Data format**: ISO 12405-2 compatible CSV for duty-cycle correlation

---

## 3. SAE J2464 — EV Battery Abuse Testing

### Scope
- **System-level** (pack/module) abuse tolerance
- Focus: **North American** OEM validation (GM, Ford, Stellantis, Tesla reference)
- Covers: **Mechanical, Electrical, Thermal, Environmental** abuse

### Test Categories & Methods

#### A. Mechanical Abuse
| Test | SAE J2464 Ref | Method | Acceptance |
|------|---------------|--------|------------|
| Vibration | §6.1 | 10–2000 Hz, 3.6g RMS, 8h/axis | No leakage, no fire, functional after |
| Shock | §6.2 | 50g/11ms half-sine, 3×/±axis | Same |
| Crush | §6.3 | 100kN flat plate, 30mm penetration | No fire/explosion, venting OK |
| Drop | §6.4 | 1m drop onto concrete (pack) | No fire, structural integrity |

#### B. Electrical Abuse
| Test | Ref | Method | Acceptance |
|------|-----|--------|------------|
| External short | §7.1 | 5mΩ max, 20°C, 6h obs | Temp ≤150°C, no fire |
| Overcharge | §7.2 | 1/3C to 1.2×Vmax or 8h | No fire/explosion |
| Over-discharge | §7.3 | 1C to 0V, hold 24h | No fire |
| Reverse charge | §7.4 | 1C reverse 90min | No fire |

#### C. Thermal Abuse
| Test | Ref | Method | Acceptance |
|------|-----|--------|------------|
| Thermal ramp | §8.1 | 5°C/min to 150°C | Thermal runaway documented, no fire propagation |
| Localized heating | §8.2 | 500W heater on 1 cell | No propagation to adjacent |
| Fire exposure | §8.3 | 800°C flame, 2min | Pack survives 130°C for 60s |

#### D. Environmental
| Test | Ref | Method |
|------|-----|--------|
| Salt spray | §9.1 | ISO 9227, 96h |
| Humidity | §9.2 | 85°C/85% RH, 1000h |
| Immersion | §9.3 | IP67: 1m/30min, IP6K9K high-pressure |

### Key Differences from UN 38.3 / IEC 62660
- **System-level** (not cell-only)
- **Propagation focus**: Does thermal runaway spread?
- **Functional safety**: Post-abuse, pack must communicate with BMS (CAN alive)
- **Test severity**: Generally harsher than UN 38.3 (e.g., 100kN crush vs 13kN)

### Practical Implications
- Required for **FMVSS 305** compliance (electric vehicle crash safety)
- OEMs require **J2464 test report** before SOP (Start of Production)
- Test lab: Typically **SAE J2464 accredited** or ISO 17025 + SAE scope
- **Video documentation** mandatory for thermal/mechanical tests

---

## 4. GB 38031 — Chinese National EV Battery Safety Standard

### Full Title
*GB 38031-2020《电动汽车用动力蓄电池安全要求》*  
**English:** Safety Requirements for Traction Batteries for Electric Vehicles

### Status
- **Mandatory** (GB = Guobiao = National Standard)
- Enforced since **2021-01-01**
- Administered by **SAC/TC 114** (SAMR/CATARC testing)
- **CCC certification required** for China market entry

### Test Structure (3 Categories, 38 Tests)

| Category | Tests | Key Requirements |
|----------|-------|------------------|
| **A. Normal Operation** | 12 | Cycle life (1000@25°C, 500@45°C), rate, temp (-20°C), storage |
| **B. Abuse Safety** | 18 | Overcharge, over-discharge, short, crush, impact, heating, fire, seawater, vibration, shock, drop, temp cycling, low pressure, thermal runaway propagation |
| **C. Environmental** | 8 | Salt spray, humidity, water immersion (IPX7), dust, thermal shock, altitude, EMC |

### Critical Pass/Fail Criteria (GB 38031-2020 §6)

| Test | No Fire | No Explosion | No Leakage | Additional |
|------|---------|--------------|------------|------------|
| Overcharge (1.1C, 1.5×Vmax) | ✓ | ✓ | ✓ | — |
| External short (5mΩ, 10min) | ✓ | ✓ | ✓ | Temp ≤150°C |
| Heating (130°C, 30min) | ✓ | ✓ | ✓ | — |
| Crush (200kN, 50% deformation) | ✓ | ✓ | — | — |
| Needle penetration (5mm, 25mm/s) | ✓ | ✓ | — | **Unique to GB** |
| Seawater immersion (3.5% NaCl, 2h) | ✓ | ✓ | ✓ | **Unique to GB** |
| Thermal runaway propagation | ✓ | ✓ | — | Delay ≥5min between cells |

### Mandatory Documentation
1. **Test report** from CATARC/CNEVCA/CACT-accredited lab
2. **BOM consistency** declaration (cell, BMS, packaging)
3. **Traceability**: Cell lot → module → pack → vehicle VIN
4. **Safety assessment report** (风险评估报告) per GB/T 31485
5. **CCC certificate** (valid 5 years, annual surveillance)

### China-Specific Requirements
- **Needle penetration test** (25mm/s, 5mm Ø) — mandatory for all chemistries
- **Seawater immersion** — simulates coastal flooding
- **Thermal propagation delay** — must demonstrate 5min cell-to-cell delay
- **Data reporting** to **NEV platform** (国家新能源汽车监管平台) real-time BMS upload

### Practical Implications
- **Non-negotiable** for China market — no equivalence acceptance
- Test cycle: **6–9 months** at CATARC Beijing/Shenzhen
- **Cell supplier audit** required (工厂审查) — process control, traceability
- **Annual re-test** (监督抽查) — random packs from production line
- **Chemistry whitelist**: LFP, NCM, NCA approved; others need special review

---

## 5. UL 2580 — Batteries for Use in Electric Vehicles

### Full Title
*UL 2580 Standard for Batteries for Use in Electric Vehicles*

### Jurisdiction
- **USA/Canada** (UL/ULC)
- **Voluntary** but de facto required by all US OEMs (FMVSS 305 linkage)
- Current edition: **3rd Ed. (2023-03-01)**, D2 revision

### Scope
- **Pack/system level** (modules, packs, BMS, thermal management)
- Covers: **Electrical, Mechanical, Thermal, Environmental, Functional Safety**
- Addresses **ISO 26262 ASIL** integration

### Test Categories

| Category | Key Tests | Typical Duration |
|----------|-----------|------------------|
| **Electrical** | Overcharge, over-discharge, short circuit, reverse polarity, isolation resistance, dielectric withstand | 2–4 weeks |
| **Mechanical** | Vibration (ISO 12405 profile), shock, crush (200kN), drop, penetrative impact | 3–4 weeks |
| **Thermal** | Thermal cycling (-40→85°C), thermal shock, thermal runaway propagation, fire exposure (1300°C) | 4–6 weeks |
| **Environmental** | Salt spray, humidity, water immersion (IP67/IP6K9K), dust, altitude, chemical exposure | 4–8 weeks |
| **Functional Safety** | BMS fault injection (ISO 26262), communication loss, sensor faults, watchdog | 2–3 weeks |

### Key Pass Criteria (UL 2580 §37–§54)
| Test | Fire | Explosion | Rupture | Leakage (electrolyte) | Functional After |
|------|------|-----------|---------|-----------------------|------------------|
| Overcharge (1.1C to 1.5×Vmax) | ✗ | ✗ | ✗ | ✗ | — |
| External short (max 5mΩ) | ✗ | ✗ | ✗ | ✗ | ✓ (communication) |
| Crush (200kN) | ✗ | ✗ | ✗ | — | — |
| Thermal runaway (cell trigger) | ✗ | ✗ | — | — | Delay ≥5min |
| Fire exposure (1300°C, 30s) | — | — | — | — | Containment 60min |

### Certification Process
1. **Construction review** — BOM, schematics, BMS code review (IEC 60730/ISO 26262)
2. **Type testing** — Full test suite at UL lab (Fremont, Tokyo, Frankfurt, Shenzhen)
3. **Factory inspection** — Initial + quarterly follow-up
4. **Listing** — UL File Number, authorized to use UL Mark
5. **Surveillance** — Unannounced audits, sample retest

### ISO 26262 Integration (UL 2580 Annex D)
- **ASIL-D** for thermal runaway prevention
- **ASIL-B/C** for BMS monitoring functions
- **Safety goal**: No uncontrolled thermal propagation
- **FMEDA** required for BMS hardware
- **Software**: IEC 61508-3 / ISO 26262-6 compliance

### Practical Implications
- **Timeline**: 12–18 months for first-time certification
- **Cost**: $500k–$2M depending on pack complexity
- **UL Witnessed Testing** option: OEM lab (ISO 17025 + UL approved)
- **Multiple listings**: Cell, module, pack can be separate files
- **Recertification**: Required for any BOM change affecting safety

---

## Cross-Standard Comparison Matrix

| Aspect | UN 38.3 | IEC 62660 | SAE J2464 | GB 38031 | UL 2580 |
|--------|---------|-----------|-----------|----------|---------|
| **Level** | Cell/Pack | Cell | Pack/System | Pack/System | Pack/System |
| **Mandatory** | Transport law | OEM spec | OEM spec | **China law** | US market de facto |
| **Geography** | Global | Intl | N. America | **China only** | N. America |
| **Thermal runaway** | Not explicit | Passive | **Propagation** | **Propagation + delay** | **Propagation + containment** |
| **Needle penetration** | No | Optional (IEC 62660-2) | No | **Mandatory** | Optional |
| **Seawater immersion** | No | No | Yes (env) | **Mandatory** | Yes (IP67) |
| **Cycle life req** | No | **1000 cycles** | No | 1000@25°C / 500@45°C | No explicit |
| **BMS/Functional** | No | No | Post-abuse comms | NEV platform upload | **ISO 26262 ASIL** |
| **Test duration** | ~2 weeks | 18–24 months | 6–8 weeks | 6–9 months | 12–18 months |
| **Cert body** | Accredited lab | Accredited lab | Accredited lab | **CATARC/SAMR** | **UL/ULC** |
| **Re-test trigger** | Design change | Chemistry change | Design change | **Annual surveillance** | BOM change + quarterly |

---

## Recommended EV-QA-Framework Integration

### 1. Test Case Mapping
```python
# research/battery_standards.py
STANDARD_TEST_MAP = {
    "UN38.3": ["T1_altitude", "T2_thermal", "T3_vibration", "T4_shock",
               "T5_short", "T6_crush", "T7_overcharge", "T8_forced_discharge"],
    "IEC62660_1": ["rate_capability", "cycle_life_25C", "cycle_life_45C",
                   "low_temp_power", "calendar_life"],
    "IEC62660_2": ["ext_short", "overcharge", "forced_discharge", "crush",
                   "impact", "thermal_stability", "temp_cycling"],
    "SAE_J2464": ["vibration", "shock", "crush", "drop", "ext_short",
                  "overcharge", "thermal_ramp", "fire_exposure", "salt_spray"],
    "GB_38031": ["cycle_life", "overcharge", "short", "crush", "needle_penetration",
                 "heating", "fire", "seawater", "thermal_propagation", "vibration"],
    "UL_2580": ["overcharge", "short", "crush", "thermal_cycling", "thermal_propagation",
                "fire_exposure", "isolation", "dielectric", "BMS_fault_injection"]
}
```

### 2. Acceptance Criteria Database
Store pass/fail thresholds per standard per chemistry (LFP/NMC/LMO/NCA) in `config/standards_thresholds.yaml`.

### 3. Test Report Generator
Auto-generate compliance matrix showing:
- Which standards each test covers
- Gap analysis (missing tests for target markets)
- Traceability: test → requirement → standard clause

### 4. Market-Specific Test Plans
| Target Market | Required Standards | Minimum Test Scope |
|---------------|-------------------|-------------------|
| **Global (non-China)** | UN 38.3 + IEC 62660 + SAE J2464/UL 2580 | Full cell + pack abuse |
| **China** | **GB 38031 (mandatory)** + UN 38.3 | Needle penetration, seawater, propagation |
| **US Only** | UN 38.3 + UL 2580 (or SAE J2464) | ISO 26262 ASIL-D BMS |
| **EU** | UN 38.3 + IEC 62660 + UN R100 (ECE) | Add UN R100 crash/reverse charge |

### 5. Lab Partner Matrix
| Lab | UN 38.3 | IEC 62660 | SAE J2464 | GB 38031 | UL 2580 |
|-----|---------|-----------|-----------|----------|---------|
| **CATARC (CN)** | ✓ | ✓ | ✓ | **Primary** | ✓ |
| **UL (US/CN/DE/JP)** | ✓ | ✓ | ✓ | ✓ | **Primary** |
| **TÜV Rheinland/SÜD** | ✓ | ✓ | ✓ | ✓ | ✓ |
| **DEKRA** | ✓ | ✓ | ✓ | Limited | ✓ |
| **SGS / Intertek** | ✓ | ✓ | ✓ | Limited | ✓ |
| **KTC / KTR (KR)** | ✓ | ✓ | ✓ | No | ✓ |

---

## Maintenance Notes

- **Update cadence**: Review annually or when standard revisions publish
- **Track revisions**: UN 38.3 (biennial), IEC 62660 (superseded by 62619/ISO 12405), GB 38031 (2025 revision expected), UL 2580 (3rd Ed. 2023)
- **Regulatory watch**: China GB 38031-2025 draft includes sodium-ion, solid-state additions
- **Harmonization**: UN GTR 20 (EV safety) converging SAE J2464 / GB 38031 / UN R100

---

## References

1. UN Manual of Tests and Criteria, Rev. 7, Part III, §38.3 (2021)
2. IEC 62660-1:2018, IEC 62660-2:2018, IEC 62660-3:2022
3. SAE J2464_202106 — Electric Vehicle Battery Abuse Testing
4. GB 38031-2020 — 电动汽车用动力蓄电池安全要求
5. UL 2580, 3rd Edition (March 2023)
6. ISO 12405-1/2/3 — Electrically propelled road vehicles — Test specification for lithium-ion traction battery packs
7. UN R100 — Uniform provisions concerning approval of vehicles with electric power train
8. FMVSS 305 — Electric-powered vehicles: electrolyte spillage and electrical shock protection
9. IEC 62619 — Secondary cells and batteries for industrial applications
10. NEV Platform Technical Specification (China MIIT, 2022)

---

*End of Document*