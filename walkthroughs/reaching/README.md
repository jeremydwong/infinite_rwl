# Energy and time is all you need — five lectures on predicting reaching movements

Pluto notebooks that build, one step at a time, from "how do I write an optimal control
problem?" to predicting the path, speed profile and duration of human reaches from an
energetic cost plus a cost of time.

| # | notebook | the point |
|---|---|---|
| 1 | `01_optimize_at_all.jl` | InfiniteOpt anatomy (time, functions of time, dynamics as constraints, BCs, objective) on a point mass. Minimum work = highway (bang–coast–bang); minimum jerk = the reach. Flash & Hogan (1985) Fig. 3b digitized; min-jerk derivation by calculus of variations (collapsible). |
| 2 | `02_turning_muscle_on.jl` | Calcium-pumping evidence for a cost of changing force; J per force-rate ⇒ W in force-rate-rate; two-link arm with τ̈ as the control and Wong et al. (2021) energy `c_W∫[θ̇τ]₊ + c_t∫[τ̈]₊`. Bell-shaped speed without asking for it. |
| 3 | `03_curl_fields.jl` | Hand forces into joints with `Jᵀ F`. Energy = work + force-rate with c_t = 0.07. A curl field does no work, yet the energy optimum *rides* it (magnetic-steering arc, R = mv/b, ≈ 10 cm bow; straight +38 %), for every c_t from 0.05 to 0.3; tracking problems (straight, ±5 cm); sensitivity to handle inertia. An honest stress test of the theory. |
| 4 | `04_kistemaker_field.jl` | Kistemaker, Wong & Gribble (2010) field at their 400 ms timing, c_t = 0.07. Work + force-rate optimum is straight (0.5 cm); tracking: straight +0.5 %, ±5 cm ≈ +100 %, their ≈ 16 cm minimal-energy path ×5.7. Work-only (joint work) optimum bows only 2.4 cm at 400 ms. |
| 5 | `05_energy_and_time.jl` | Free duration with `J = E + kT` (k = 25 J/s): duration and peak speed vs distance, scaling argument, and the duality — fixing T is pricing time at k = −dE*/dT. |

## Running

From the repo root:

```sh
julia --project=. -e 'using Pluto; Pluto.run(notebook="walkthroughs/reaching/01_optimize_at_all.jl")'
```

Every figure and animation comes up immediately from `cache/`. Each lecture ends with a **Your turn**
section: sliders + a **Solve** button that runs the optimizer live (a few seconds; the first solve
compiles).

**Static copies** (no Julia needed; figures and the animations work, sliders don't) are in `html/`.

## Editing

The prose lives in ordinary `md"""…"""` cells — click the eye icon in Pluto to show and edit them;
nothing is generated. Shared model code (arm dynamics, cost, fields, tracking, free time, cache helpers,
the animation player) is in `ReachingLectures.jl`; Lecture 1's point-mass model is written inline in
the notebook on purpose, and Lectures 2 and 5 quote the module's source directly (`source_excerpt`), so
the code shown is the code that runs.

## Rebuilding caches and HTML

```sh
julia --project=. walkthroughs/reaching/build_lectures.jl          # all five (≈ 2–4 min from cold caches)
julia --project=. walkthroughs/reaching/build_lectures.jl 3 4      # some
```

This runs the notebooks headless in Pluto, fills any missing cache (everything expensive goes through
`cached("name") do … end`), reports errored cells, and rewrites `html/`. Delete a file in `cache/`
to force that result to be recomputed. Caches are plain `Dict`s of arrays (Serialization), so code edits
don't invalidate them — but they also won't notice a model change; delete them when you change physics.

## Model notes and choices

* Arm: planar, horizontal, Li & Todorov (2004) parameters; shoulder at the origin, x right, y forward.
  Default reach (0, 0.20) → (0, 0.50) m.
* Cost: Wong et al. (2021) Eqs. 1 & 13 with c_W = 4.2 and c_t = 0.085 (their cyclic-reaching fit) in
  Lectures 2 and 5; c_t = 0.07 (discrete, time-constrained reaches) in Lectures 3 and 4. Plus a tiny
  `eps·∫τ̈²` (1e-7) so the L¹ problem is unique. Rest-to-rest with τ = τ̇ = 0 at both ends.
* Tracking problems constrain the path **shape** (sideways offset vs. progress along the line) inside a
  ±2 mm tube and let the optimizer pick the timing. A position-level *equality* at every collocation
  node is high-index (position is four integrations from τ̈) and Ipopt stalls on it.
* Kistemaker field: their Eq. 2 with b = 150 N·s/m², 1 kg handle, T = 0.40 s (their Fig. 1B timing).
  Their minimal-energy path is approximated as an offset ∝ λ⁶(1−λ)³ peaking at 16 cm two-thirds of the way.
* Flash & Hogan data: `data/flash_hogan_1985_fig3b.csv`; the header documents the digitization.
