# Frenetix Motion Planner — Fork Comparison

Comparison of two local copies of the Frenetix Motion Planner:

- **Upstream** — freshly cloned from GitHub, at `/home/yuan/Chatbot/Frenetix-Motion-Planner`
- **Fork** — the in-use modified copy, at `/home/yuan/Chatbot/LLM-MoST/Frenetix-Motion-Planner`

Diff date: 2026-04-23. Generated with `diff -rq`, excluding `.git/`, `venv/`, `logs/`, and `__pycache__/`.

---

## Summary of Divergence

- **6** new files/directories in the fork
- **1** file removed from the upstream tree (replaced)
- **13** files modified

The changes cluster into three themes:

1. **Batch-mode workflow.** A new `main_batch.py` (replacing `main.py`) plus `batch_presets/` and `scenario_batch_list.csv` enable running large scenario batches over a `Scenarios_batch/<name>/Original/<name>.xml` layout, with per-run failure logging.
2. **Cost-function tuning.** `cost.yaml` is rebalanced: many terms that were zero upstream are now weighted (acceleration, jerk, lateral/longitudinal jerk, orientation, path length, lane-center offset, distance to obstacles, responsibility). The previous weights are preserved as `cost_old.yaml`.
3. **Visualization & observability.** `cr_scenario_handler/utils/visualization.py` is heavily rewritten to export all valid/invalid trajectories (with cost maps) as JSON; ego color is changed from orange to blue. `agent_batch.py` now classifies collisions (front/rear/side) and records the colliding object id. `main_multiagent.py` expands the failure CSV header accordingly.

---

## Files Added in Fork

| Path | Size | Purpose |
|---|---:|---|
| `main_batch.py` | 201 lines | Replacement entry point adapted from upstream `main.py`; adds `argparse`, `--input-file`, `--batch` mode, and per-scenario failure logging to `log_failures.csv`. |
| `main_without_ego.py` | 1072 lines | Standalone CommonRoad XML visualizer that renders scenarios and recorded trajectories without running any motion planning. |
| `requirements.txt` | 58 lines | Pip-style pinned requirements (numpy, scipy, matplotlib, torch, commonroad-io, etc.), complementing the existing `pyproject.toml`. |
| `scenario_batch_list.csv` | 100 rows | Active batch scenario list consumed by `main_batch.py`. |
| `batch_presets/` | 7 CSVs | Pre-cut batch lists: `batch_10_scenarios.csv`, `batch_50_scenarios.csv`, `batch_100_scenarios.csv`, `batch_200_scenarios.csv`, `batch_300_scenarios.csv`, `batch_400_scenarios.csv`, `batch_500_scenarios.csv`. |
| `configurations/frenetix_motion_planner/cost_old.yaml` | — | Snapshot of the upstream cost weights kept for reference after the rebalance. |

---

## Files Removed in Fork

- **`main.py`** — replaced by `main_batch.py`. The replacement keeps the same simulation entry shape but adds CLI arguments, the `Scenarios_batch/<name>/Original/<name>.xml` path resolution, and a `start_simulation_single` helper that writes tracebacks into `logs/log_failures.csv` on exception.

---

## Files Modified

### `configurations/frenetix_motion_planner/cost.yaml`

Cost weights rebalanced. Previously-zero terms are now active, and the two dominant upstream terms are dialed down. The old weights are preserved alongside in `cost_old.yaml`.

```yaml
# upstream                        # fork
acceleration:               0.0   acceleration:               0.3
jerk:                       0.0   jerk:                       0.3
lateral_jerk:               0.2   lateral_jerk:               0.5
longitudinal_jerk:          0.2   longitudinal_jerk:          0.5
orientation_offset:         0.0   orientation_offset:         0.1
path_length:                0.0   path_length:                0.2
lane_center_offset:         0.0   lane_center_offset:         0.7
velocity_offset:            1.0   velocity_offset:            0.6
velocity:                   0.    velocity:                   0.2
distance_to_reference_path: 5.0   distance_to_reference_path: 2.0
distance_to_obstacles:      0.0   distance_to_obstacles:      0.8
prediction:                 0.2   prediction:                 0.3
responsibility:             0     responsibility:             0.2
```

The block of commented-out cost-name keys at the bottom of the upstream file is removed. **Impact:** this materially changes planner behavior — not a cosmetic tweak.

### `configurations/simulation/simulation.yaml`

Two trailing blank lines added. No semantic change.

### `configurations/simulation/visualization.yaml`

```yaml
# upstream
save_gif: False
# plot vehicle position with x_y distance in meters. 0=Standard Plot
# fork
save_gif: True
# plot vehicle position with x_y distance in meters. 0=Standard Plot, >0=Dynamic camera following ego
```

Enables GIF output by default; comment clarifies the dynamic camera-follow behavior.

### `cr_scenario_handler/simulation/agent.py`

```python
# upstream
              self.agent_state.goal_checker.last_goal_position):
# fork
# Allow some buffer before triggering max s-position (5 meters tolerance)
              self.agent_state.goal_checker.last_goal_position + 5.0):
```

Gives agents 5 m of slack before the "past goal s-position" termination fires.

### `cr_scenario_handler/simulation/agent_batch.py`

Adds a new `determine_collision_type(agent, scenario, timestep)` function (~90 lines). It:

- Reads ego position/orientation from `agent.record_state_list[-1]`.
- Iterates `scenario.dynamic_obstacles`, filters to those within 10 m.
- Transforms the relative position into ego-local coordinates and classifies the hit as `front_collision`, `rear_collision`, or `side_collision` based on longitudinal/lateral offsets relative to half the ego length (4.508 m) and half-width (1.610 m).
- Returns the nearest candidate's `(collision_type, obstacle_id)`.

The classification is then written into the failure CSV (see `main_multiagent.py` below).

### `cr_scenario_handler/simulation/simulation.py`

```python
# upstream
batch_list.append(AgentBatch(agents, self.global_timestep, self.msg_logger, self.log_path, self.mod_path))
# fork
if agents:  # Only create batch if there are agents
    batch_list.append(AgentBatch(agents, self.global_timestep, self.msg_logger, self.log_path, self.mod_path))
```

Guard against empty-batch `AgentBatch` construction.

### `cr_scenario_handler/utils/configuration_builder.py`

```python
# upstream
config_cli    = OmegaConf.from_cli()
config_merged = OmegaConf.merge(config_default, config_cli)
# fork
# config_cli = OmegaConf.from_cli()
# config_merged = OmegaConf.merge(config_default, config_cli)
# Don't include CLI config pollution
config_merged = config_default
```

**Behavior change:** the fork silently drops all CLI-supplied OmegaConf overrides and always uses `config_default`. Any user or script that relied on passing `key=value` overrides on the command line will no longer see them take effect.

### `cr_scenario_handler/utils/general.py`

Adds a `fix_trajectory_positions(scenario)` helper and calls it from the scenario loader:

```python
def fix_trajectory_positions(scenario):
    """
    The WaleNet prediction network requires positions to be numpy arrays,
    but SUMO-converted scenarios may have positions stored as Python lists.
    """
    for obstacle in scenario.dynamic_obstacles:
        if hasattr(obstacle.prediction, 'trajectory') and obstacle.prediction.trajectory:
            for state in obstacle.prediction.trajectory.state_list:
                if isinstance(state.position, list):
                    state.position = np.array(state.position)
    return scenario
```

Called after the scenario is loaded. Fixes a compatibility issue when feeding SUMO-converted scenarios into WaleNet.

### `cr_scenario_handler/utils/goalcheck.py`

```python
# upstream
def _initialize_last_goal_position(self, planning_horizon_in_seconds=0.5, default_velocity=1.0):
    """Initializes the last goal position."""
# fork
def _initialize_last_goal_position(self, planning_horizon_in_seconds=2.0, default_velocity=1.0):
    """Initializes the last goal position with extended buffer for better goal approach."""
```

Planning horizon for the last-goal-position default extended 0.5 s → 2.0 s.

### `cr_scenario_handler/utils/visualization.py`

Large rewrite — **~331 lines of diff**. Key changes:

- Ego color changed from orange (`#E37222` / `#9C4100`) to blue (`#2255e3` / `#003359`).
- Built-in ego trajectory drawing disabled (`ego_params.trajectory.draw_trajectory = False`) — the fork draws custom trajectories per timestep.
- `obs_params.static_obstacle.show_label` changed `True → False`.
- The optimal trajectory line now carries a `label='Optimal Trajectory'` handle for legend reuse.
- **Removed**: the visible-sensor-area fill and the occlusion-map scatter/legend block.
- **Added**: the sampled-trajectory bundle plotting is replaced with a block that, for each valid and invalid trajectory, extracts `x / y / velocity / acceleration / heading` (sub-sampled every 5th point, rounded to 2 dp) plus the full `costMap`, accumulates them into `all_trajectories_data` and `all_cost_data`, and writes a merged JSON via a helper `save_json_robustly()` that creates parent directories on demand.
- Style tweaks: trajectory lines switched to `linestyle='--'`, `zorder=60`, dashed.

### `frenetix_motion_planner/cost_functions/partial_cost_functions.py`

```python
# upstream
from scipy.integrate import simps
# fork
from scipy.integrate import simpson
```

**⚠ Likely bug in the fork.** The import was changed (presumably for SciPy ≥ 1.14, where `simps` was removed in favor of `simpson`), but the four call sites inside the file still use the old name — `simps(acceleration_sq, ...)`, `simps(jerk_sq, ...)`, `simps(theta, ...)`, `simps(velocity, ...)`. The module will raise `NameError: name 'simps' is not defined` on first use unless a `simps = simpson` alias is added or the call sites are renamed.

### `main_multiagent.py`

```python
# upstream
line = "scenario;agent;timestep;status;message\n"
# fork
line = "scenario;agent;timestep;status;message;result;collision_type;colliding_object_id\n"
```

Failure-log CSV header extended with the three extra columns populated by the new collision-type logic in `agent_batch.py`.

### `.gitignore`

Trailing newline added. No semantic change.

---

## Notable Observations

1. **`partial_cost_functions.py` will crash at runtime.** The `simps → simpson` import change is incomplete — the call sites were not updated. Either alias `simps = simpson` at import time or rename all four calls.
2. **`configuration_builder.py` silently ignores CLI config.** Any downstream workflow that passed `key=value` overrides on the command line will be broken. The comment calls this "CLI config pollution," but it's a breaking behavior change, not a bug fix.
3. **`cost.yaml` is a behavior change, not a config cleanup.** The rebalance activates many previously-dormant cost terms and sharply reduces `distance_to_reference_path` (5.0 → 2.0). Trajectories produced by the fork are not directly comparable to upstream.
4. **Collision classification and trajectory JSON export form an observability layer.** Together with the batch-mode runner and failure CSV, they look purpose-built for offline analysis of large scenario sweeps — plausibly the "LLM-MoST" pipeline driving the fork.

---

## How to Reproduce This Comparison

```bash
diff -rq /home/yuan/Chatbot/Frenetix-Motion-Planner \
         /home/yuan/Chatbot/LLM-MoST/Frenetix-Motion-Planner \
  | grep -v -E '\.git|venv|logs|__pycache__'
```

For a per-file unified diff:

```bash
diff -u /home/yuan/Chatbot/Frenetix-Motion-Planner/<path> \
        /home/yuan/Chatbot/LLM-MoST/Frenetix-Motion-Planner/<path>
```
