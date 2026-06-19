# Leading-edge traversability artifact: investigation, fix, and decision log

Branch: `fix/leading-edge-traversability-artifact`

## 1. Symptom

On the deployed Go2 robot the published square `traversability` grid shows a persistent
straight non-traversable line at the outer edge in the robot's direction of motion. As the robot
advances the line is dragged inward and is never erased.

## 2. What was claimed before (and why it was wrong)

A prior analysis (a different tool) attributed the artifact to two things. Both were checked
against the actual code with an adversarial multi-agent review and **neither holds**:

- **"`shift_map_xy` pads the traversability layer (layer 3) with 0.0, so the padded strip shows as
  an obstacle line."** The padding is real, but `get_traversability` masks any cell that is neither
  `is_valid` nor `is_upper_bound` to NaN. A freshly padded strip has both flags 0, so it is
  published as *unknown*, not as a finite obstacle. Layer 3 is also recomputed every fusion cycle.
- **"`traversability_buffer` is not invalidated on shift, so stale border values persist."** True
  that `shift_map_xy` does not invalidate the buffer, but its interior `[3:-3,3:-3]` is fully
  overwritten on every `get_traversability` call, and the 2-cell returned border band is never
  written so it stays NaN. No finite obstacle value can survive there. This is a benign no-op.

## 3. Actual root cause (verified)

The line is produced by the **visibility-cleanup ray loop**, not by padding or the buffer.

1. On forward motion the robot-centered map is rolled (`shift_map_xy`, `elevation_mapping.py:237`)
   and the freshly exposed leading-edge strip is zeroed in all layers (`is_valid=0`,
   `is_upper_bound=0`, `upper_bound=0`).
2. On the next fusion, the visibility-cleanup ray loop stamps the un-measured frontier cells:
   for invalid cells it sets `upper_bound[5]=nz` (a ray-sampled height) and `is_upper_bound[6]=1`
   (`kernels/custom_kernels.py:238-244`, also `262-265`). This creates a **height step** in
   `upper_bound` between the measured ground (`is_valid=1`, true height) and the ray-stamped band.
3. The traversability filter is an edge detector (`exp(-sum|gradients|)`,
   `traversability_filter.py:60-70`) running on the dilated `upper_bound`
   (`elevation_mapping.py:405-417`). A height step becomes a near-zero (obstacle) traversability.
4. Because the ray loop set `is_upper_bound=1`, those cells **pass** the publish mask
   `(is_valid + is_upper_bound) > 0.5` in `get_traversability`, so they are published as a finite
   obstacle (not NaN).
5. It persists because nothing relaxes the stamped `upper_bound`/`is_upper_bound` at the ~10 m
   frontier: re-observation only overwrites directly measured cells; `clear_overlap_map` only
   touches the central 4 m window (`overlap_clear_range_xy=4.0`); and
   `update_upper_bound_with_valid_elevation` runs only at init. `cp.roll` then carries the same
   world cells inward, so the line "comes inward and is never erased".

## 4. Verification

### 4.1 Mechanism, real trained weights (CPU, deterministic)

Loaded the deployed `config/core/weights.dat` and ran the real filter math on synthetic
`upper_bound` fields:

| input | min traversability |
| --- | --- |
| flat ground (no step) | 1.000 (no artifact) |
| 0.10 m step | 0.31 (obstacle, `< 0.5`) |
| 0.20 m step | 0.097 |
| 0.30 m step | 0.030 (hard obstacle) |

So the deployed weights turn any frontier step `>= ~0.1 m` into a published obstacle. Flat ground
produces no line; a height step is required. This is locked in by
`tests/test_leading_edge_traversability.py::test_real_filter_renders_height_step_as_obstacle`.

### 4.2 Real recorded rosbags (no robot, no GPU)

The shipped publisher records `elevation` and `traversability`. Their masks differ:
`elevation` finite `<=> is_valid`; `traversability` finite `<=> is_valid OR is_upper_bound`.
Therefore **`traversability` finite AND `elevation` NaN == a ray-stamped (is_upper_bound-only)
cell**. Analysing `recovered_bags/`:

- The artifact is present in real recorded output: obstacle cells split into directly-measured and
  ray-stamp-only, with the ray-stamp-only obstacles concentrated at the frontier/border.
- Applying the fix (publish mask = `is_valid` only, i.e. keep traversability only where `elevation`
  is finite) removes exactly the ray-stamp-only obstacle cells and preserves 100% of measured
  obstacles (`removed / ray_only = 1.00`).
- Fraction of obstacle cells that are ray-stamp-only is **sensor dependent**: ~25% on the Go2/A2
  lidar (`a2_out`), but ~60-75% on the Velodyne sets (`vlp_out`).

Verification scripts and rendered before/after frames live in `temporary/trav_repro/` (gitignored
scratch): `repro.py` (mechanism), `verify_on_bag.py` (aggregate), `viz_frame.py` / `seq_montage.py`
(figures).

## 5. The fix

`get_traversability` now masks by `is_valid` only by default, so ray-stamped upper-bound-only cells
are published as **unknown (NaN)** instead of a finite obstacle. Controlled by a new parameter:

- `parameter.py`: `traversability_mask_use_upper_bound: bool = False`
- `config/core/core_param.yaml`: `traversability_mask_use_upper_bound: false` (always-loaded core config)
- `scripts/elevation_mapping_node.py`: wired into `set_param_values_from_ros`
- `elevation_mapping_cupy/elevation_mapping.py`: `get_traversability` selects the mask

Setting it `true` restores the exact legacy behavior (`is_valid OR is_upper_bound`).

### Important tradeoff (read before deploying)

This mask is **blunt**: it drops *all* upper-bound-only cells, not only the frontier artifact. A
genuine obstacle detected only via ray-casting/upper-bound (e.g. a wall seen at distance but never
directly measured as ground) becomes `unknown` rather than `obstacle`. On the Go2/A2 this is ~25%
of obstacle cells and is concentrated at the frontier, so it is reasonably targeted; on Velodyne it
removes the majority and would be too aggressive. `terrain_cost` maps unknown to `terrain_cost_unknown`
(default 75), not lethal (100), so unknown is conservative-but-passable. If the navigation stack
relies on upper-bound obstacles, set `traversability_mask_use_upper_bound: true` and instead pursue a
more targeted fix (see Future work).

## 6. Decisions made autonomously

- **Branch base**: branched off the current `feat/terrain-cost-layer` HEAD, because the whole
  analysis (cupy traversability filter, `terrain_cost`, line numbers) matches that working tree.
- **Configurable parameter, default = new behavior**: rather than hard-coding the mask change, added
  a reversible config flag (default `false` = new behavior). This honors the chosen behavior while
  keeping a deployed robot able to roll back from YAML.
- **Verification via local rosbags, not the robot**: SSH to the robot was intentionally not used
  (local-rosbag verification was requested and is non-disruptive). The recorded-bag identity
  (`elevation` finite vs `traversability` finite) closes the "is there really a step / is_upper_bound"
  gap on real data without needing the raw `upper_bound` layer.
- **Clean diff**: the editor's auto-formatter reflowed entire files (imports, docstrings). That churn
  was reverted; only the semantic change is committed.

## 7. Future work (more targeted alternatives)

If keeping upper-bound obstacles matters, prefer one of:

- Suppress the ray-stamp only at the frontier / near `max_ray_length`, so distant ray-stamped steps
  do not enter the filter (fix at source in `custom_kernels.py`).
- Relax stale `is_upper_bound`/`upper_bound` outside the central window via a `max_cell_age` / decay,
  so the frontier line clears as it scrolls inward but real walls persist.
- Feed the traversability filter an `is_valid`-only dilation mask so the frontier step never reaches
  the filter, while leaving the published mask unchanged.

## 8. Tests

- `tests/test_leading_edge_traversability.py`
  - `test_real_filter_renders_height_step_as_obstacle` — pure numpy + real weights, runs anywhere.
  - `test_get_traversability_excludes_upper_bound_only_by_default` — needs cupy (`importorskip`),
    runs on GPU/CI; asserts the default mask drops the ray-stamp-only cell and keeps the measured one.

Note: the package `__init__` hard-imports cupy, so the full test suite only runs where cupy is
installed (robot/CI). The mechanism test was additionally validated standalone on CPU here
(flat=1.000, 0.1 m=0.31, 0.3 m=0.03).
