# Wheely's maze — release policy (M2 final)

Everything the web port (M3) and the demo (M4) need. The TypeScript sim must reproduce **this** behaviour;
the Python sources named below are the reference implementation.

## The policy

- `weights.json` — the shipped actor. Source run: `artifacts/runs/mix_fwd5mem_12x12_pcrit_s0` (memory inputs,
  actor 10 → 12 → 12 → 4 → 2, trained with PPO and a privileged critic; the critic is not shipped).
  Chosen by Bryan 2026-10-05 from single-seed runs (no multi-seed comparison).
- 350 parameters; inference = 4 small matrix multiplies.

### Test set (1,000 held-out maps per category, success %)

| | open | obstacles | scribbles | mixed | mazes | presets | traps |
|---|---|---|---|---|---|---|---|
| **this policy** | 100.0 | 96.8 | 92.9 | 87.6 | 59.7 | 94.0 | 60.7 |
| hand-coded reactive baseline | 100.0 | 95.6 | 89.7 | 84.6 | 68.6 | 94.1 | 38.4 |

Matches or beats the baseline in 6 of 7 categories; mazes are the honest gap (see `maze_analysis.md`,
`maze_failures.png`: it fails winding mazes that need long detours away from B). Every run: `all_runs_test_set.md`.

On the six demo presets from their default A/B (`presets_paths.png`, `gifs/`): reaches B on Open field (6 s),
Posts (9 s), Slalom (16 s), Rooms (13 s), Small maze (36 s); **loops inside The trap** — the intended
"no long-term memory" moment.

## weights.json

```jsonc
{
  "version": 1,
  "arch": [10, 12, 12, 4, 2],          // inputs, hidden..., outputs
  "hidden_activation": "tanh",         // on every hidden layer
  "output": "clip",                    // final layer is linear, then clip to [-1, 1]
  "obs_spec": [...10 names...], "act_spec": ["left wheel", "right wheel"],
  "sim_params": { ... see below ... }, // the sim this policy was trained in — the port must use these
  "W": [[12x10], [12x12], [4x12], [2x4]],   // W[k] is (out, in)
  "b": [[12], [12], [4], [2]],
  "meta": {...}
}
```

Forward pass (also gives every neuron's value for the network panel):

```
h = obs                                   // 10 floats
for k in 0..3: h = W[k] · h + b[k]; if k < 3: h = tanh(h)
action = clip(h, -1, 1)                   // [uL, uR]; show pre-clip h on the output neurons if useful
```

## sim_params (from weights.json)

radius 0.3 · wheel_base 0.6 · v_max 1.2 · reverse_max 0.0 · dt 1/30 · frame_skip 2 (policy at 15 Hz) ·
ray_angles_deg [-60, -30, 0, 30, 60] · ray_range 3.0 · goal_dist_scale 10 · goal_radius 0.35 ·
goal_inputs "sincos_dist" · prev_action_inputs true. Arena border = 4 capsules along the edges, radius 0.05.

## Simulation spec (reference: `mazebot/geometry.py`, `sim.py`, `env.py`)

Conventions: x right, **y down**, heading θ = 0 along +x, positive θ clockwise on screen. float64. Use only
+ − × ÷ and sqrt in geometry (no hypot/atan2) — keeps Python and JS bit-identical except sin/cos.

**Walls** are capsules `[ax, ay, bx, by, rad]` (a zero-length segment is a circular post).

**Policy step (15 Hz):** `[uL, uR] = clip(action)`, then `frame_skip` (2) physics steps; stop early if the centre
is within `goal_radius` of B after any physics step (success).

**Physics step** (`sim.physics_step`):
```
v = max(v_max * (uL + uR) / 2, -reverse_max * v_max)     // reverse_max = 0: no backward motion
w = v_max * (uL - uR) / wheel_base                       // left faster => clockwise
n = max(1, ceil(|v| * dt / (radius / 2)));  h = dt / n
repeat n: tm = th + w*h/2; x += v*cos(tm)*h; y += v*sin(tm)*h; th = wrap(th + w*h); push_out(x, y)
wrap(a): a = fmod(a + pi, 2pi); if a <= 0: a += 2pi; return a - pi     // JS % == C fmod
```
**push_out** (`geometry.push_out`): up to 4 iterations; each finds the capsule with the deepest penetration
`pen = radius + rad - dist(centre, segment)` (first index on ties); if pen ≤ 0 stop; else move the centre along
(centre − closest point)/dist by pen (if dist ≈ 0, along the segment's left normal). Any push ⇒ "contact".

**Rays** (`geometry.cast_rays`): from the robot centre at θ + angle; first hit = min over each capsule's two end
circles and its body (iq's ray-capsule test, see the code for the exact formulas); capped at ray_range.

**Observation (10 floats, this order):**
1. 5 × ray distance / ray_range
2. sin_b = hx·uy − hy·ux,  cos_b = hx·ux + hy·uy  — h = (cos θ, sin θ), u = unit vector from robot to B
   (if distance < 1e-9: sin 0, cos 1). sin > 0 ⇒ B is to the right (clockwise).
3. min(distance to B / 10, 1)
4. previous left wheel, previous right wheel — the **clipped** action from the last policy step; 0, 0 at reset.

## Path check ("no path to B" warning; reference `distance_field.py`)

Grid with cell = radius/2 = 0.15 over the arena; a cell is free iff the clearance at its centre (distance to the
nearest capsule surface) ≥ radius. Reachability: BFS/Dijkstra over free cells, 8-connected, diagonal moves only if
both orthogonal neighbours are free; seed with free cells within 1.5 cells of B; A is reachable if any of the
5×5 cells around A was reached. Gaps narrower than the robot (0.6) correctly count as blocked.

"Lost" timeout used in training/evaluation: 3 × (path distance A→B) / v_max + 10 s.

## Presets (`presets/presets.json`)

Six layouts defined landscape 10 × 6.25 with default start `[x, y, θ]` and goal `[x, y]`; wall radius 0.08–0.45.
Portrait = transpose: swap x/y of every point, width/height, and θ' = π/2 − θ.

## Drawing (training distribution)

Freehand-looking walls in training: RDP-simplified strokes (ε 0.03), stroke radius 0.04–0.2, plus posts 0.15–0.55,
bars, wobbly mazes (wall radius 0.05–0.12, corridors ≥ ~1.1). A demo brush radius of ~0.08–0.12 stays in
distribution; arenas in training were 6–12 units per side (aspect ≤ 1.8).

## Sprite

Site asset `src/assets/mascot/mascot-top.svg` faces −y: draw rotated by θ + π/2; body width ≈ 2.6 × radius.
