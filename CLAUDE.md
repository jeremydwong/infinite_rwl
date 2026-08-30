# infinite_rwl — project notes

Reduced-order (point-mass) walking models solved as trajectory-optimization problems
with InfiniteOpt + Ipopt. Legs are push-only force actuators acting along the leg; the
"leg" is a telescoping strut standing in for a real two-segment leg (knee hidden).

Active focus: `infinite_3d_rebula_simpler.jl` — extend the 2D model
(`infinite_rebula_simpler.jl`) to a stable, economical gait in 3D.

Run diagnostics headless with: `julia --project=. <script>` and `ENV["GKSwstype"]="nul"`.

---

## Log

### 2026-08-28 — ollie rider: human-referenced units, scrubbable animation, audit trims

- **Units policy (Jeremy):** all work NONDIMENSIONAL (board mass 1, g 1, ℓ_max 1 — the
  model already was); the HUMAN is the reference for inputs; report in SI where useful.
  `RiderOptions`: `rider_mass` = M/m (25 ≈ 70 kg / 2.8 kg), new `force_limit_bw` = per-leg
  limit in rider body weights (2.0 → 50 board weights, unchanged numbers; `force_limit`
  now an explicit override, NaN = derived via `force_limit_value(opt)`). New
  `RiderScales` (70 kg, ℓ_max 0.9 m, g 9.81) + `dimensional(sc,opt)` → multipliers:
  time 0.303 s, force 27.5 N per board weight, velocity 2.97 m/s.
- **Report (`generate_ollie_rider_svg_report.jl`)**: SMIL animations dropped (chunky —
  ~170 discrete-visibility text nodes re-evaluated every frame + `<object>` embedding).
  Only solve B (lowest board point) is animated: inline SVG + requestAnimationFrame,
  linear interpolation between frames in physical time, scrub slider, play/pause,
  speed, toggles (force arrows, standing-height line deck+ℓ_max = 1.08, leg-reach ℓ_max
  circles, COM trails). A = static snapshots + time series. Parameter and audit tables
  carry SI columns. Audit trimmed to essentials + `energy_residual_total/_rel` +
  `max_constraint_violation` (JuMP primal_feasibility_report, atol 0).
- **Finding:** constraint violation 4e-8, NE/impact/momentum at precision, BUT the energy
  audit is −5.3 ≈ 10% of leg work (−2.4 in load+pop, −2.9 in flight where the legs do
  ~no work): first-order symplectic-Euler discretisation error at 21/31/61 nodes. The
  spec's mesh-refinement check (residual halves at 2× nodes) is still TODO for this model.
- Stage-2 warm start (board-apex → lowest-point) ends NUMERICAL_ERROR; warm and cold B
  now land on the identical solution (lowest-point apex 1.625 = 1.46 m).

### 2026-07-05 — 3D lateral-sway sweep is pathological; cause is pinned cadence

**Goal of the session:** understand whether the 3D model
(`point_mass_walker_3d`) supports the Priority-1 hypothesis that *more lateral sway
should cost more energy*, via the `run_pz_sweep` / `pz_initial` machinery.

**Coordinate / symmetry reminder.** Mirror (`X(0) = −X(tf)`) boundary conditions apply
to the *frontal-plane* lateral states `pz`, `vz` (and to `vy`); the *sagittal* (x–y)
plane is the periodic one (`px` translates by one step, `py(1)=py(0)`, `vx(1)=vx(0)`).
"Sway" = lateral = frontal plane = `pz`. Note: an optimal periodic trajectory is **not**
the same as a stable limit cycle — 3D stability is a separate step-to-step (Poincaré) map
question, not something the optimizer delivers.

**Status of the model.** Baseline `point_mass_walker_3d(pz_initial=0)` solves cleanly:
`LOCALLY_SOLVED`, obj 0.0286, work 0.0191, force-rate(fr2) 0.0094, `t_f`=1.2, energy-balance
error ≈ 0.0067. But the pz sweep blows up:

| `pz_initial` | status | work | fr2 | obj | py(0) |
|---|---|---|---|---|---|
| 0.00 | solved | 0.019 | 0.009 | 0.029 | 0.66 |
| 0.02 | solved | 1.00 | 59.1 | 60.1 | 0.10 (floor) |
| 0.04 | solved | 7.26 | 784 | 792 | 0.59 |
| ≥0.06 | infeasible | — | — | — | — |

**Diagnosis (isolated one lever at a time, via existing kwargs — no code change):**
- **Cadence is the dominant lever, not the physics.** Slowing the step
  (`step_speed` → larger `t_f`) relieves the blow-up: at `t_f`≈2.96 the sweep becomes
  smooth and monotonic (obj 0.12 → 0.27 → 2.29 at pz 0, 2, 4 cm). At `t_f`=1.86,
  2 cm-sway obj drops 60 → 3.6. So *the "more sway costs more" trend is real once the
  step has enough time.* Confirmed.
- **The force-rate weight `c_fr2` is not the culprit.** Cutting it 10× helped only ~8×
  and worsened feasibility; the underlying ∫F̈² stays ~6000. The force spikes are
  physical: with push-only legs and feet only ±0.05 apart while the leg is ~0.9 long,
  only ~5% of leg force points sideways, so redirecting the COM laterally in a short
  fixed step needs large, fast-ramping forces → ∫F̈² explodes.
- **The `py ≥ 0.1` floor is active** in nearly every swayed case (py(0)=0.10). Solutions
  ride a hard bound → distorted physics + poor conditioning.
- **Geometry ceiling:** pushing `pz` past ~half the stance width (0.05) puts the COM
  outside the foot base — near-unphysical for push-only legs. The 6–10 cm sweep points
  are asking the model to fall over.
- **Forward speed drifts across the sweep.** `vx(0)` is free and varied wildly
  (0.545 → 1.70) between pz points, which confounds cost comparisons.

**Dead end tried (don't repeat):** added an opt-in `free_tf` flag that unpins `t_f` and
put the *existing* `cost_time = c_t·t_f` term into the objective. It **collapsed to
`t_f → 0.001`** (mincing, "speed" 651, mostly NUMERICAL_ERROR/infeasible). That term
rewards *finishing fast* — wrong sign. Reverted; file is back to its pre-session state.
**Lesson:** Priority-2A needs a *step-frequency* penalty (penalize high cadence, ∝ 1/t_f),
which pushes toward *longer* steps — the mechanism that historically gave step length
≈ 0.7 L. (One run that happened to reach the slow basin, `t_f`=3.64, solved a swayed
gait at obj 4.6 vs 792 — the target regime exists; it just needs the right cost to steer
there.)

**Next steps (Jeremy's priority order):**

1. **Sanity-check the pz0 sweep before trusting any curve.**
   - Is the `pz_initial=0` baseline itself a real walking gait, or a degenerate low bounce?
     (py(0)=0.66 is low, `vy(0)`≈0.575, lateral force/motion ≈ 0.) Look at the full
     trajectory + `plot_results_3d`.
   - Verify the intended experiment: `pz(0)=pz_initial` and `pz(1)=−pz_initial` while the
     feet stay pinned at z=±0.05 regardless of `pz_initial`. Is that the geometry you want,
     or should the feet move with the sway?
   - Check dynamics-violation and energy balance at *each* pz0, not just pz0=0.
   - Rule out a collocation artifact behind the ∫F̈² spike: re-solve with more supports /
     finer mesh and confirm the blow-up persists.
   - Decide whether forward speed `vx` should be held fixed across the sweep (it currently
     drifts and confounds the comparison).

2. **Redo the sweep and fix py(0) hitting the floor.**
   - The fix is a cost, not a tighter bound: add the bent-knee / posture penalty (2B)
     `c_knee·(∫F_trail·κ(s_trail) + ∫F_lead·κ(s_lead)) dt`, with `κ(s)=0.5·√(1−s²)`
     (two-link geometry; zero at a straight leg s=1, grows as the COM drops). Use
     `κ=(1−s²)` if the √ is numerically stiff near s=1. This lifts the COM off the floor,
     regularizes swayed solutions, and encodes the real knee-torque cost.
   - Prefer a cleaner independent variable: sweep `step_width` with `pz(0)` free (the
     classic Kuo U-shaped cost-vs-step-width), and/or cap `pz_initial` within the stance.

3. **Free some constraints (Priority 2A — the unblocker).**
   - Free `t_f` (initialize slow) and add a *frequency* cost `c_freq·(1/t_f)²`
     (dimensionless with g=1, L≈1). **Not** the `cost_time = c_t·t_f` term — that minces.
   - To control step *length* (not just speed), also free step length: make `px(1)` and
     the lead-foot x-position variables. Sweep `c_freq` so the preferred length lands
     near 0.7 L.
   - Then re-run the pz / step-width sweep and confirm the cost curve is smooth and
     monotonic (Priority 1 demonstrated properly).

**Reproduce the diagnostics:** all via kwargs to `point_mass_walker_3d`, e.g.
`point_mass_walker_3d(pz_initial=0.02, step_speed=0.22)` (slow cadence),
`… c_fr2=0.001` (light force-rate). No source changes needed for the sweeps above.

### 2026-08-27 — Width sweep is flat because the lateral velocity BC is wrong

`export_walking_width_sweep_report.jl` → `simulation_reports/walking_width_sweep_report.html`.
`c_fr2=5e-5` balances work≈fr2 at width 0.10 (0.417 / 0.428); fr2 scales exactly
linearly in `c_fr2` (∫F̈²≈7400 regardless), so force-rate isn't shaping the solution.
Work is ~0.41 for all widths 0–0.30; lateral work ∫|Fx·vx| ≤ 0.004. **Cause:** BCs are
`px(1)=−px(0)` but `vx(1)=vx(0)` — the mirrored gait needs `vx(1)=−vx(0)`. The COM
drifts sideways at constant vx and the reversal is a free impulse at the step flip, so
the step-to-step lateral redirection cost (the human width cost) is absent. With
`vx(1)=−vx(0)` (not committed) width 0 is unchanged but widths ≥0.1 blow up
(work≈42, ITERATION_LIMIT/infeasible) — same pathology as the 2026-07-05 pz sweep.
Lateral work *is* in the objective (leg-length rate uses the full 3D leg·v).

### 2026-08-28 — 3D rider model + kickflip (ollie_rider_3D.jl)

Convention: X lateral, Y forward, Z up (documented at the top of the file). R = Rz(ψ)Rx(θ)Ry(φ);
θ pitch (ollie), φ roll about the long axis (kickflip), Euler's equations in the body frame,
I = (0.07, 0.008, 0.075). Flight foot force = normal push N at body point (σ, s, 0); roll torque
σ·N. Driver/report: `generate_kickflip_3d_report.jl` → `simulation_reports/kickflip_3d_report.html`;
stage data in `simulation_reports/{ollie3d_stage1,kickflip3d_stage2,kickflip3d_stage3}.jls`.
Results: lateral-locked 3D ollie == 2D (1.802 vs 1.804). Kickflip with μ=0 converges: pop-frozen
apex 1.713, all-free apex 1.765 (−2 % vs ollie), roll rate 6.6 rad/unit, square landing.
Convergence recipe: roll-target continuation → flight-only with pop frozen → all-free warm start
with `lateral_ground_locked=true`; if NUMERICAL_ERROR, cold barrier restart (mu_init 0.1, adaptive).
Gotcha: changing `Rider3DOptions` fields invalidates the .jls caches (they store `opt`).
Known gaps: yaw needs `land_yaw_max` (gyroscopic drift lands the board 34° off otherwise); pop
locks roll/yaw without checking wheel-reaction split; post-strike push sharpens with mesh.
- 2026-08-28 later: force cap 3 BW (running peak) in both rider models; `foot_slide_max=1.5` front-foot
  slide limit (needed: without it the foot teleported between pushes; with it the load/pop shows the
  human forward sweep). Solve loose (3.0) then tighten (1.5) — tight-from-guess fails. Results at 3 BW:
  ollie 2.52 (2D 2.46), kickflip 2.47 (−2 %). Foot-force map (top-down, deck frame) in the kickflip report.
- 2026-08-30: force-penalty appendix in kickflip report — pilot sweep shows the ΔF² rate penalty SATURATES
  (w×1000 → peak N unchanged ≈20.5): the post-strike spike is constraint-driven (frozen pop); fix structurally
  (ReBULA-style force states with F, Ḟ continuous across the strike, or rate limit, or Hill F–v). Pluto+PlutoUI+
  IJulia installed; `kickflip_playground.jl` (Pluto): julia --project=. -e 'using Pluto; Pluto.run(notebook="kickflip_playground.jl")'.
