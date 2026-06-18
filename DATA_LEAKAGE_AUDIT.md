# Data Leakage Audit — EV-QA-Framework ML/Analytics Stack

**Scope**: 8 modules — `analysis.py`, `automl.py`, `battery_scoring.py`, `soh_predictor.py`, `soh_transformer.py`, `fleet_analytics.py`, `digital_twin.py`, `physics_features.py`

**Audit criteria**:
- (a) **fit/predict boundary** — `fit_transform` or `fit+predict` in a single public method
- (b) **In-memory state sharing** — mutable state persisting between train and predict calls
- (c) **Test isolation** — tests using same dataset for train and eval without shuffle/split
- (d) **Target-derived feature leakage** — features computed from or directly using the target variable

---

## Summary Table

| Module | Issue | Line | Severity | Rationale |
|--------|-------|------|----------|-----------|
| **analysis.py** | (a) fit/predict boundary: `analyze_telemetry()` does `fit_transform` + `fit_predict` in one call | 142–152 | **CRITICAL** | First call on any data fits scaler+model AND predicts on that same data. Method name suggests inference but performs training. If called on test data first → fits on test data. |
| **analysis.py** | (b) In-memory state sharing: scaler/model persist in instance; no enforcement of train→predict order | 84, 94, 143–152 | **HIGH** | `self.scaler` and `self.model` are instance attrs. Calling `analyze_telemetry(test_df)` before any train call fits on test data. `AnomalyDetector` subclass (L457+) provides separate `train()`/`detect()` but base class doesn't enforce it. |
| **analysis.py** | (c) Test isolation: tests reuse same DataFrame for fit and predict | N/A (test files) | **MEDIUM** | `test_automl.py` L39–41, L52–53: same `df` used for `fit()`, `predict()`, `evaluate()`. No temporal split or shuffle. |
| **analysis.py** | (d) Target leakage via physics features: `get_physics_features()` may compute features from `capacity`/`soh` | 396–452 | **LOW** | Physics features (IC curve, delta-Q) use `capacity` column which correlates with SOH. If used as features for SOH prediction → leakage. |
| **automl.py** | (a) fit/predict boundary: **CLEAN** — `fit()` does `fit_transform`, `predict()` does `transform` | 50, 103 | — | Proper separation. `fit()` fits scaler + model; `predict()` reuses fitted scaler. |
| **automl.py** | (b) In-memory state sharing: **CLEAN** — state persists correctly between fit→predict | 28, 25–27, 97–104 | — | `self.scaler`, `self.best_model` set in `fit()`, read in `predict()`. No cross-call contamination if API used correctly. |
| **automl.py** | (c) Test isolation: tests use same synthetic data for train and eval | `test_automl.py` L39–41, L52–53 | **MEDIUM** | Single `df = make_telemetry_df()` used for `fit()`, `predict()`, `evaluate()`. No train/test split. |
| **automl.py** | (d) Target-derived features: **CLEAN** — features are `voltage`, `current`, `temperature`; target is `soh` | 41, 48 | — | No target column used as feature. |
| **battery_scoring.py** | (a) fit/predict boundary: inherits from `EVBatteryAnalyzer.analyze_telemetry()` | 83, 263 | **CRITICAL** | `_anomaly_analyzer = EVBatteryAnalyzer()` (L83); `_compute_anomaly()` calls `analyze_telemetry()` (L263) → same fit+predict issue as analysis.py. |
| **battery_scoring.py** | (b) In-memory state sharing: internal `EVBatteryAnalyzer` instance reused across `compute_score()` calls | 83, 263 | **HIGH** | Single `_anomaly_analyzer` instance created at init. Its scaler/model state persists across multiple `compute_score()` invocations on different data. |
| **battery_scoring.py** | (c) Test isolation: **N/A** — not an ML trainer; scoring is deterministic per-call | — | — | Tests verify score ranges, not generalization. |
| **battery_scoring.py** | (d) **Target leakage: `_compute_soh()` returns `df["soh"].iloc[-1]` directly** | 253–254 | **CRITICAL** | If `BatteryScorer` is used in a pipeline where SOH is the prediction target, reading `soh` column as a "feature" (soh_score) IS the target. Composite score then leaks target into training signal. |
| **soh_predictor.py** | (a) fit/predict boundary: **CLEAN** — `train()` fits scaler+model; `predict_next()` uses `transform`+`predict` | ~185, ~240 | — | Proper separation. `prepare_data()` only creates sequences, no fitting. |
| **soh_predictor.py** | (b) In-memory state sharing: **CLEAN** — `self.scaler`, `self._feature_scaler`, `self.model`, `self.is_trained` managed correctly | ~105, ~185, ~240 | — | State set in `train()`, read in `predict_next()`. `is_trained` guard prevents predict before train. |
| **soh_predictor.py** | (c) Test isolation: tests predict on training data subset | `test_soh_predictor.py` L281–303 | **MEDIUM** | `df = make_dataframe(50)`; `predictor.train(df)`; `recent = df.tail(5)` (L301) — predicts on last 5 rows of TRAINING data. No temporal holdout. |
| **soh_predictor.py** | (d) Target-derived features: **CLEAN** — features: voltage, current, temperature; target: soh | ~118, ~137 | — | No SOH used as input feature. |
| **soh_transformer.py** | (a) fit/predict boundary: **CLEAN** — `train()` fits; `predict()` uses fitted scaler+model | ~253, ~330 | — | Same pattern as soh_predictor. |
| **soh_transformer.py** | (b) In-memory state sharing: **CLEAN** — instance attrs for scaler, model, is_trained | ~116, ~138, ~253 | — | Correct lifecycle management. |
| **soh_transformer.py** | (c) Test isolation: tests predict on same data used for training | `test_soh_transformer.py` L351–369 | **MEDIUM** | `df = make_dataframe(20)`; `transformer.train(df)`; `result = transformer.predict(df)` (L354) — evaluates on training data. |
| **soh_transformer.py** | (d) Target-derived features: **CLEAN** — features: voltage, current, temperature; target: soh | ~191, ~260 | — | No target as feature. |
| **fleet_analytics.py** | (a) fit/predict boundary: **N/A** — aggregation class, no ML fit/predict | — | — | Delegates to `BatteryScorer`, `EVBatteryAnalyzer`, `PhysicsFeatureExtractor`. |
| **fleet_analytics.py** | (b) In-memory state sharing: caches `score_battery()` results; stores all battery telemetry in `self._batteries` | ~410 (test), internal | **LOW** | `score_battery()` memoizes results (test L410–415). If underlying telemetry changes, cache is stale. `_batteries` dict holds full DataFrames — memory growth risk. |
| **fleet_analytics.py** | (c) Test isolation: **N/A** — not ML training | — | — | Tests verify aggregation correctness, not generalization. |
| **fleet_analytics.py** | (d) Target leakage: inherits `BatteryScorer` SOH leakage + `EVBatteryAnalyzer` fit issues | 20, 396–452 (analysis.py) | **HIGH** | Uses `BatteryScorer` (which reads `soh` column directly) and `EVBatteryAnalyzer` (fit+predict in one call). Fleet-level scores thus leak target. |
| **digital_twin.py** | (a) fit/predict boundary: **N/A** — physics simulation, no ML | — | — | `predict_soh()` is physics projection, not ML inference. |
| **digital_twin.py** | (b) In-memory state sharing: **BY DESIGN** — `self.state` and `self._history` mutate on `step()` | 92–110 | — | Intentional simulation state. `predict_soh()` explicitly does NOT modify state (test L147–151). |
| **digital_twin.py** | (c) Test isolation: **N/A** | — | — | Not applicable. |
| **digital_twin.py** | (d) Target-derived features: **N/A** | — | — | Physics parameters only. |
| **physics_features.py** | (a) fit/predict boundary: **CLEAN** — stateless feature extraction, no fit/predict | — | — | All methods are pure functions of inputs. |
| **physics_features.py** | (b) In-memory state sharing: **CLEAN** — no persistent instance state | — | — | `PhysicsFeatureExtractor` has no fitted params. |
| **physics_features.py** | (c) Test isolation: **N/A** | — | — | Unit tests for feature correctness, not ML. |
| **physics_features.py** | (d) Target leakage: **CLEAN** — features from raw signals (V, I, T, capacity, time); no target | — | — | `capacity` used for delta-Q/IC curve — if capacity == target proxy, could leak. But typically capacity is measured, SOH is derived. |

---

## Cross-Module Leakage Chains

| Chain | Description | Severity |
|-------|-------------|----------|
| `fleet_analytics` → `battery_scoring` → `analysis` | FleetAnalytics uses BatteryScorer which uses EVBatteryAnalyzer. All three share fit/predict boundary violation and SOH target leakage. | **CRITICAL** |
| `battery_scoring` → `analysis` | BatteryScorer._anomaly_analyzer is an EVBatteryAnalyzer instance. Calling compute_score() multiple times reuses the same analyzer instance, causing state contamination across different batteries. | **HIGH** |
| `automl` tests → `soh_predictor` tests → `soh_transformer` tests | All test files use `make_dataframe()` / `make_telemetry()` with linearly decreasing SOH (`np.linspace(100, 90, n)`). Train and predict on same synthetic trajectory → overfitting not detected. | **MEDIUM** |

---

## Recommendations

1. **analysis.py**: Split `analyze_telemetry()` into explicit `fit(df)` and `predict(df)` methods (like `AnomalyDetector` does). Deprecate the combined method or make it clear it's for "fit+predict on same data" exploratory use only.

2. **battery_scoring.py**: 
   - Remove direct `soh` column read in `_compute_soh()` when used in ML pipelines. Accept SOH as explicit parameter or require pre-computed SOH score.
   - Create fresh `EVBatteryAnalyzer` per `compute_score()` call, or add `reset()` method.

3. **Test suite**: Add proper train/temporal splits in integration tests. Use `train_test_split` with `shuffle=False` for time-series data. Verify generalization on held-out data.

4. **physics_features.py**: Document that `capacity`-derived features (IC curve, delta-Q) should not be used as inputs to SOH prediction models where SOH is derived from capacity.

5. **fleet_analytics.py**: Add cache invalidation for `score_battery()` when telemetry is updated. Consider weak references or TTL for `_batteries` storage.