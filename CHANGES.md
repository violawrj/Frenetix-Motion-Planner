# Local Changes vs. Upstream Clone

This tree started as a fresh clone of [Frenetix-Motion-Planner](https://github.com/TUM-AVS/Frenetix-Motion-Planner) on 2026-04-23. The changes below were then applied on top by porting selected improvements from the LLM-MoST fork at `/home/yuan/Chatbot/LLM-MoST/Frenetix-Motion-Planner`.

For the full upstream-vs-fork diff that motivated these ports, see [`COMPARISON.md`](COMPARISON.md).

## Summary

| # | File | Type | What changed |
|---|---|---|---|
| 1 | `cr_scenario_handler/utils/visualization.py` | modified | Replaced with fork version: trajectory JSON export, blue ego, dashed trajectory lines |
| 2 | `frenetix_motion_planner/cost_functions/partial_cost_functions.py` | modified | `simps` → `simpson` (SciPy ≥ 1.14 compatibility) — import **and** all call sites |
| 3 | `main_without_ego.py` | **added** | Standalone CommonRoad scenario visualizer (no motion planning) |
| 4 | `main_multiagent.py` | modified | `score_overview.csv` header extended with 3 new columns |
| 5 | `cr_scenario_handler/simulation/agent_batch.py` | modified | New `determine_collision_type()` helper; populates the 3 new CSV columns on collision |

## Details

### 1. `cr_scenario_handler/utils/visualization.py`

Full replacement with the fork's version. Behavioral changes:

- Ego vehicle color: orange (`#E37222`/`#9C4100`) → blue (`#2255e3`/`#003359`)
- `ego_params.trajectory.draw_trajectory = False` — built-in trajectory drawing disabled so the custom per-timestep trajectory plotting takes over
- `obs_params.static_obstacle.show_label = True → False`
- Sampled-trajectory bundle plotting replaced: every valid / invalid trajectory is drawn as a dashed line at `zorder=60`, and its `x / y / velocity / acceleration / heading_radian` (sub-sampled every 5th point, rounded to 2 dp) plus full `costMap` are written to JSON under:
  - `<log_path>/trajectory_cost/trajectory_cost_<timestep>.json`
  - `<log_path>/trajectory/trajectory_<timestep>.json`
  - `<log_path>/cost/cost_<timestep>.json`
- Optimal trajectory line is now labeled `'Optimal Trajectory'` for legend reuse
- **Removed:** the visible-sensor-area fill and the occlusion-map scatter/legend block — occlusion visualization no longer works in this file

### 2. `frenetix_motion_planner/cost_functions/partial_cost_functions.py`

```diff
-from scipy.integrate import simps
+from scipy.integrate import simpson
```

and the four call sites at lines 31, 44, 149, 195 renamed from `simps(...)` to `simpson(...)`.

> Note: the LLM-MoST fork only changed the import and left the call sites untouched, which raises `NameError: name 'simps' is not defined` on first use. This tree fixes the call sites too.

### 3. `main_without_ego.py` (new)

Copied verbatim from the fork. 1072-line standalone script that renders a CommonRoad XML scenario — road network plus recorded trajectories of dynamic obstacles — without running any motion planning. Useful for inspecting raw scenario data or for debugging ingestion.

### 4. `main_multiagent.py`

```diff
-            line = "scenario;agent;timestep;status;message\n"
+            line = "scenario;agent;timestep;status;message;result;collision_type;colliding_object_id\n"
```

The three new columns are populated by the logic added in change #5.

### 5. `cr_scenario_handler/simulation/agent_batch.py`

- Added `import numpy as np`
- Added module-level `determine_collision_type(agent, scenario, timestep)` helper. It transforms the nearest (≤10 m) dynamic obstacle into the ego-local frame at the collision timestep and classifies the hit against half the ego length (4.508 m) / half the width (1.610 m) as one of: `front_collision`, `rear_collision`, `side_collision`. Returns `("boundary_collision", None)` when no nearby dynamic obstacle is found, or `("unknown", None)` on error. Returns `(collision_type, colliding_obstacle_id)` otherwise.
- Hardened `AgentBatch.__init__`: `self.latest_starting_time = max(...) if self.agent_list else 0`, so an empty batch no longer raises.
- In `_update_batch`, when an agent's status is `AgentStatus.COLLISION`, call `determine_collision_type(agent, agent.scenario, agent.current_timestep)` and append `collision_type;colliding_object_id` to the row written to `score_overview.csv`. Non-collision terminations write two empty fields.

## Intentionally *not* ported

The following fork divergences from `COMPARISON.md` were reviewed but deliberately left out of this tree:

- `configurations/frenetix_motion_planner/cost.yaml` rebalance — materially changes planner behavior; left upstream for now
- `cr_scenario_handler/utils/configuration_builder.py` disabling OmegaConf CLI merge — this is a regression (silently drops CLI overrides), not an improvement
- `cr_scenario_handler/utils/general.py` `fix_trajectory_positions()` — only relevant when feeding SUMO-converted scenarios into WaleNet
- `cr_scenario_handler/utils/goalcheck.py` horizon 0.5 s → 2.0 s — tuning change, not a bug fix
- `cr_scenario_handler/simulation/agent.py` +5 m goal-position tolerance — same category
- `cr_scenario_handler/simulation/simulation.py` empty-`agents` guard — edge case not hit in the scenarios we run
- `main_batch.py`, `batch_presets/`, `scenario_batch_list.csv`, `requirements.txt`, `configurations/frenetix_motion_planner/cost_old.yaml` — all part of the fork's batch-mode workflow, which this tree doesn't need

## Verify

```bash
# visualization JSON export should write under <log_path>/{trajectory,cost,trajectory_cost}/
python3 -c "import ast; ast.parse(open('cr_scenario_handler/utils/visualization.py').read())"

# simpson import + call sites
grep -n "simps\|simpson" frenetix_motion_planner/cost_functions/partial_cost_functions.py
# expect: 5 matches, all "simpson"

# collision-type helper and CSV header
grep -n "determine_collision_type" cr_scenario_handler/simulation/agent_batch.py
grep -n "collision_type;colliding_object_id" main_multiagent.py

# standalone visualizer present
ls main_without_ego.py
```
