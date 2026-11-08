### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 03000001-5eac-4a11-9b0e-000300000001
md"""
# Lecture 3 — Pushing on the hand: curl fields

*Energy and time is all you need · part 3 of 5*

The workhorse experiment of motor learning: hold the handle of a robot, reach, and the robot
pushes your hand sideways in proportion to its velocity — a **curl field** (Shadmehr & Mussa-Ivaldi 1994).
People are knocked off course at first, and after a few hundred reaches they are back to
nearly straight lines.

We'll build everything on one concrete, typical experiment: **Mattar & Ostry (2007)**, *Modifiability of
generalization in dynamics learning* (J Neurophysiol 98:3321). 160 people, an InMotion2 robot, center-out
reaches of 15 cm (12 cm between target edges), a clockwise curl field of 15 N·s/m, and a hard timing window.
That is what a "standard" curl-field reach looks like: short and not very fast.

This lecture does three things:

1. shows how a force **at the hand** enters an arm model as joint torques, ``\tau_{\text{field}} = J^{\mathsf T} F``;
2. asks what **energy** (work + force rate, Lecture 2) predicts in Mattar & Ostry's task, with time fixed —
   in all eight directions, at other distances, and at other places in the workspace;
3. compares that with what their subjects did.

A curl field never does any work on the hand. You might think energy should therefore ignore it.
It does not — and what it does instead is worth looking at carefully.
"""

# ╔═╡ 03000002-5eac-4a11-9b0e-000300000002
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    using PlutoUI, Plots, Printf, LinearAlgebra
    include(joinpath(@__DIR__, "ReachingLectures.jl"))
    using .ReachingLectures
    default(fontfamily="Helvetica", linewidth=2, framestyle=:axes, grid=false, size=(720, 320),
            legend=:topright, foreground_color_legend=nothing, palette=PAL,
            titlefontsize=11, guidefontsize=10, tickfontsize=8, legendfontsize=8,
            bottom_margin=4Plots.mm, left_margin=3Plots.mm)
    TableOfContents()
end

# ╔═╡ 03000003-5eac-4a11-9b0e-000300000003
md"""
## From a force at the hand to torques at the joints: ``J^{\mathsf T} F``

The hand position is a function of the joint angles, ``x = \phi(\theta)``, and its velocity is
linear in the joint rates through the **Jacobian**:

```math
\dot x = J(\theta)\,\dot\theta, \qquad
J = \begin{bmatrix} -\ell_1 \sin\theta_1 - \ell_2 \sin(\theta_1+\theta_2) & -\ell_2 \sin(\theta_1+\theta_2) \\
                      \ell_1 \cos\theta_1 + \ell_2 \cos(\theta_1+\theta_2) & \ \ \ell_2 \cos(\theta_1+\theta_2) \end{bmatrix}.
```

Now push on the hand with a force ``F``. The power it delivers is the same whether you compute it at
the hand or at the joints — ``F\cdot\dot x = F^{\mathsf T} J\dot\theta = (J^{\mathsf T}F)\cdot\dot\theta`` — so the
torques it produces at the joints are

```math
\tau_{\text{field}} = J(\theta)^{\mathsf T} F .
```

That's all it takes: the equations of motion become ``M\ddot\theta + h = \tau + J^{\mathsf T}F``, and the
model (`solve_reach`) adds exactly that term. Here it is for one posture — the hand at the start
position, pushed 10 N to the right:
"""

# ╔═╡ 03000004-5eac-4a11-9b0e-000300000004
let p = arm_params()
    q = deg2rad.([45.0, 90.0])             # Mattar & Ostry's start posture: shoulder 45°, elbow 90°
    x = hand_position(p, q)
    J = hand_jacobian(p, q)
    F = [10.0, 0.0]                        # 10 N to the right
    τ = J' * F
    md"""
    hand at ($(round(100x[1], digits=1)), $(round(100x[2], digits=1))) cm from the shoulder ·
    ``J^{\mathsf T}F`` = (**$(round(τ[1], digits=2))**, **$(round(τ[2], digits=2))**) N·m at (shoulder, elbow).
    A sideways push at the hand twists *both* joints — the arm has to answer it with both.
    """
end

# ╔═╡ 03000005-5eac-4a11-9b0e-000300000005
md"""
## The curl field

Mattar & Ostry's field, in their notation (x lateral, y forward):

```math
\begin{bmatrix} f_x \\ f_y \end{bmatrix} = \begin{bmatrix} 0 & 15 \\ -15 & 0 \end{bmatrix}\begin{bmatrix} v_x \\ v_y \end{bmatrix}\ \text{N},
\qquad\text{i.e.}\qquad F = B\,\dot x,\quad B = b\begin{bmatrix}0 & 1\\ -1 & 0\end{bmatrix},\quad b = 15\ \text{N·s/m}.
```

The force is always **perpendicular** to the hand's velocity, rotating it clockwise (the counter-clockwise
field is ``-B``). Moving straight ahead at 0.5 m/s you get shoved 7.5 N to the right.

Because ``F \perp \dot x``,

```math
P_{\text{field}} = F\cdot\dot x = \dot x^{\mathsf T} B\,\dot x = 0 \quad (B \text{ is skew-symmetric}).
```

**A curl field never does work on the hand.** Not on any path, not at any instant. So a cost that only
counted mechanical work done on the outside world would not care about the field at all.

But the arm does not pay for work on the outside world; it pays for **joint** work (positive work at each
joint, Lecture 2) and for **changing joint torque**. Resisting a sideways push needs joint torques that
rise and fall as the hand speeds up and slows down — and those cost energy. So what does the energy optimum do?

!!! note "The energy used in this lecture"
    ``E = \sum_i \int \big(c_W[\dot\theta_i\tau_i]_+ + c_t[\ddot\tau_i]_+\big)\,dt`` with ``c_W = 4.2`` and
    **``c_t = 0.07``**. Lecture 2 used ``c_t = 0.085``, the value fitted to *cyclic* reaching (Wong et al. 2021);
    for discrete, time-constrained reaches the coefficient is plausibly somewhat lower, ≈ 0.07. The answer
    turns out not to depend on it (sensitivity table further down). We'll also run **work only**
    (``c_t = 0``) as the comparison.
"""

# ╔═╡ 03000006-5eac-4a11-9b0e-000300000006
c_t3 = 0.07                                            # force-rate-rate coefficient for this lecture

# ╔═╡ 03000030-5eac-4a11-9b0e-000300000030
md"""
## Mattar & Ostry's task, as an optimal control problem

From their Methods:

* **posture** — the start (center of the target circle) is at shoulder 45° (from the frontal plane) and elbow 90°;
* **targets** — 8 targets, 45° apart, on a circle of radius 15 cm (target radius 1.5 cm, so 12 cm edge to edge);
* **timing** — move "within 350 ± 50 ms" (auditory feedback), i.e. at most 400 ms;
* **field** — clockwise curl, ``b`` = 15 N·s/m; reference directions 135° and 315°;
* **curvature measure** — *perpendicular error* (PE): the sideways deviation from the start–end line **at peak
  speed**. We use the same measure, signed so that **PE > 0 is to the right of the direction of travel** —
  which is the way the clockwise field pushes you.

So our problem: start at rest at the center, end at rest at a target 15 cm away, ``T`` = 0.40 s, minimize
energy. For each direction we solve three problems: no field, the field with a **free** path, and the field
with the hand held to a **straight** line (±2 mm tube, timing free). The free problem is started from three
different guesses (straight, and paths bowed ±3 cm); the table reports when they disagree.

The arm is Li & Todorov's (2004), so the hand starts 2 cm left of and 45 cm in front of the shoulder.
No robot inertia for now — we'll add it at the end.
"""

# ╔═╡ 03000031-5eac-4a11-9b0e-000300000031
begin
    mo = (center = hand_position(arm_params(), deg2rad.([45.0, 90.0])),   # shoulder 45°, elbow 90°
          D = 0.15, T = 0.40, b = 15.0, dirs = 0:45:315)                # Mattar & Ostry (2007)
    mo_target(c, angle, D) = c .+ D .* [cosd(angle), sind(angle)]       # angle in degrees from +x (right)
    mo_postures = [(30, 90), (45, 90), (60, 90), (30, 120), (45, 120), (60, 120), (45, 60)]
    mo_reachable(x) = 0.15 < norm(x) < 0.60                              # stay away from the workspace edges
    pe_mm(s) = 1000 * perp_error(s).pe                                   # Mattar & Ostry's PE, mm (+ = right)
end;

# ╔═╡ 03000032-5eac-4a11-9b0e-000300000032
mattar = cached("L3_mattar_b15") do
    E = CostWeights(c_t=c_t3)                          # work + force rate
    W = CostWeights(c_t=0.0, eps=1e-6)                 # work only
    task(st, tg, T; kw...) = ReachTask(; start=st, target=tg, T=T, b_curl=mo.b, kw...)
    # three problems per condition: no field, free path in the field, straight path in the field
    function trio!(d, key, st, tg, T, C; kw...)
        null = solve_reach(task(st, tg, T; kw...), C)
        free = solve_reach_multistart(task(st, tg, T; field=:curl_cw, kw...), C)
        str  = solve_reach(task(st, tg, T; field=:curl_cw, track=bow(0.0), kw...), C; guess=null)
        solve_ok(str) || (str = solve_reach(task(st, tg, T; field=:curl_cw, track=bow(0.0), kw...), C; guess=free))
        d[key*"_null"] = null; d[key*"_free"] = free; d[key*"_straight"] = str
    end
    d = Dict{String,Any}()
    # 1. the experiment: 8 directions, 15 cm, 400 ms
    for a in mo.dirs, (cn, C) in (("E", E), ("W", W))
        trio!(d, "dir_$(a)_$(cn)", mo.center, mo_target(mo.center, a, mo.D), mo.T, C)
    end
    # 2. the same with a 1 kg isotropic handle, and with the real (uncompensated) InMotion2 (appendix)
    for a in mo.dirs
        trio!(d, "dir_$(a)_E_m1", mo.center, mo_target(mo.center, a, mo.D), mo.T, E; m_endpoint=1.0)
        trio!(d, "dir_$(a)_E_robot", mo.center, mo_target(mo.center, a, mo.D), mo.T, E; robot=:inmotion2)
    end
    # 3. reach distance: T fixed at 0.40 s, or T growing like D^0.28 (Lecture 5's energy-and-time optimum)
    for a in (90, 180, 315), D in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30), (cn, C) in (("E", E), ("W", W))
        tg = mo_target(mo.center, a, D); mo_reachable(tg) || continue
        for (tm, T) in (("fix", mo.T), ("scaled", mo.T*(D/mo.D)^0.28))
            trio!(d, "dist_$(a)_$(D)_$(tm)_$(cn)", mo.center, tg, T, C)
        end
    end
    # 4. where in the workspace: other start postures, 15 cm, 400 ms
    for (sh, el) in mo_postures, a in (0, 90, 135, 315), (cn, C) in (("E", E), ("W", W))
        c = hand_position(arm_params(), deg2rad.([sh, el])); tg = mo_target(c, a, mo.D)
        mo_reachable(tg) || continue
        trio!(d, "loc_$(sh)_$(el)_$(a)_$(cn)", c, tg, mo.T, C)
    end
    # 5. movement duration
    for a in (90, 315), T in (0.30, 0.35, 0.50, 0.65), (cn, C) in (("E", E), ("W", W))
        trio!(d, "dur_$(a)_$(T)_$(cn)", mo.center, mo_target(mo.center, a, mo.D), T, C)
    end
    d
end;

# ╔═╡ 03000049-5eac-4a11-9b0e-000300000049
premium(key) = mattar[key*"_straight"]["E"] / mattar[key*"_free"]["E"] - 1;   # extra cost of the best straight path vs the free optimum

# ╔═╡ 03000033-5eac-4a11-9b0e-000300000033
md"""
## What energy predicts: eight 15 cm reaches in 400 ms

Grey: the energy-optimal reaches with no field (close to straight, as Lecture 2 promised). Colour: the
energy-optimal reaches in the clockwise field. Circles are the 1.5 cm targets.
"""

# ╔═╡ 03000034-5eac-4a11-9b0e-000300000034
let
    panels = []
    for (cn, ttl, col) in (("E", "work + force rate (c_t = 0.07)", PAL[1]), ("W", "work only", PAL[2]))
        pl = plot(title=ttl, aspect_ratio=:equal, xlabel="x (m)", ylabel="y (m)", legend=false,
                  xlims=mo.center[1] .+ (-0.21, 0.21), ylims=mo.center[2] .+ (-0.21, 0.21))
        θ = range(0, 2π, length=40)
        for a in mo.dirs
            tg = mo_target(mo.center, a, mo.D)
            plot!(pl, tg[1] .+ 0.015cos.(θ), tg[2] .+ 0.015sin.(θ), c=INK, lw=1, alpha=0.5)
            n = mattar["dir_$(a)_$(cn)_null"]; f = mattar["dir_$(a)_$(cn)_free"]
            plot!(pl, n["hand"][1, :], n["hand"][2, :], c=:gray70, lw=1.5)
            plot!(pl, f["hand"][1, :], f["hand"][2, :], c=col, lw=2)
        end
        scatter!(pl, [mo.center[1]], [mo.center[2]], c=INK, ms=3)
        push!(panels, pl)
    end
    plot(panels..., layout=(1, 2), size=(760, 390))
end

# ╔═╡ 03000035-5eac-4a11-9b0e-000300000035
md"""
Every one of them curves, and they all curve the **same way**: to the *left* of the direction of travel,
i.e. *against* the push. The hand doesn't fight the field — it aims off to the left and lets the field bend
it around to the target. Work alone does the same thing, only more erratically (more on that below).

### Why: the curl field is a free steering wheel

Take away the arm for a moment and think of the hand as a point mass ``m`` with no muscle force at all.
The only force is ``F = B\dot x``: always perpendicular to the velocity, magnitude ``b\,|\dot x|``. That is exactly
the force on a charged particle in a magnetic field, and the motion is the same: **constant speed on a circle**,

```math
m\frac{v^2}{R} = b\,v \quad\Longrightarrow\quad R = \frac{m\,v}{b}.
```

A clockwise field bends every path clockwise — so a path that *starts out to the left* of the target gets
bent back onto it, and the field supplies the sideways force for free. The muscles still have to start and
stop the hand, but they don't have to do the steering. Going straight means the opposite: your muscles must
cancel ``b\,v`` sideways the whole way, with forces that rise and fall with speed, and every one of those
force changes is paid for in the ``c_t`` term.

Put in Mattar & Ostry's numbers. A 15 cm reach in 0.4 s peaks at ≈ 0.7 m/s with no field; in the field the
optimum speeds up to ≈ 0.9 m/s. With ≈ 1.4 kg of effective arm mass at the hand, ``R = mv/b`` ≈ 8 cm — about the
half-distance to the target. A pure field-carried arc would be a half circle, bowing 7.5 cm; the optimum takes a
partial ride instead and bows a few centimetres.
"""

# ╔═╡ 03000036-5eac-4a11-9b0e-000300000036
let
    dirs = collect(mo.dirs)
    pl = plot(title="perpendicular error at peak speed (+ = with the field, to the right)", xlabel="reach direction (deg)",
              ylabel="PE (mm)", xticks=dirs, legend=:outerright, size=(780, 360), ylims=(-90, 40))
    # Mattar & Ostry (2007) Fig. 2, read off by eye: first field trials ≈ +20…25 mm, end of training ≈ +3…5 mm
    band(lo, hi) = Shape([-10, 325, 325, -10], [lo, lo, hi, hi])
    plot!(pl, band(20, 25), c=PAL[4], lw=0, fillalpha=0.35, label="humans, first field trials")
    plot!(pl, band(2.5, 5.5), c=PAL[3], lw=0, fillalpha=0.45, label="humans, end of training")
    hline!(pl, [0], c=INK, lw=1, ls=:dot, label="")
    for (key, lab, col, mk) in (("E", "energy (work + force rate)", PAL[1], :circle), ("W", "work only", PAL[2], :diamond),
                                ("E_m1", "energy, + 1 kg handle", PAL[7], :utriangle), ("E_robot", "energy, + InMotion2 (uncompensated)", PAL[5], :square))
        plot!(pl, dirs, [pe_mm(mattar["dir_$(a)_$(key)_free"]) for a in dirs], c=col, marker=mk, ms=5, label=lab)
    end
    plot!(pl, dirs, [pe_mm(mattar["dir_$(a)_E_null"]) for a in dirs], c=:gray60, ls=:dash, marker=:circle, ms=3, label="energy, no field")
    pl
end

# ╔═╡ 03000037-5eac-4a11-9b0e-000300000037
md"""
### Compare with the people

Mattar & Ostry's subjects (their Fig. 2; the bands above are read off by eye, pooled over their reference
directions): the first reaches in the field are pushed **right** by ≈ 2 cm; by the end of training the PE is
back to a few millimetres, *still slightly to the right* — slightly under-compensated, never over. Catch trials
then swing ≈ 2 cm to the left.

The energy optimum is on the **other side**: 1–6 cm to the left, in every direction. It is not "people are
nearly straight and energy is nearly straight too". Energy, taken literally with time fixed, predicts a curved
reach bowed *against* the field by about as much as the field initially pushed people *with* it, or more —
and people don't do that.

Here is what going straight would cost, by direction:
"""

# ╔═╡ 03000038-5eac-4a11-9b0e-000300000038
let
    rows = ["| direction | no field: PE | **energy**: PE | straight costs | E free / straight / no field (J) | **work only**: PE | straight costs |",
            "|--:|--:|--:|--:|--:|--:|--:|"]
    for a in mo.dirs
        e = mattar["dir_$(a)_E_free"]; es = mattar["dir_$(a)_E_straight"]; en = mattar["dir_$(a)_E_null"]
        w = mattar["dir_$(a)_W_free"]
        flag(s) = length(unique(round.(s["starts"], digits=3))) > 1 ? "†" : ""
        push!(rows, @sprintf("| %d° | %+.1f mm | **%+.0f mm** | %+.0f %% | %.1f / %.1f / %.1f | %+.0f mm%s | %+.0f %% |", a, pe_mm(en), pe_mm(e),
              100premium("dir_$(a)_E"), e["E"], es["E"], en["E"], pe_mm(w), flag(w), 100premium("dir_$(a)_W")))
    end
    Markdown.parse(join(rows, "\n") * "\n\n*Straight = the best path held within ±2 mm of the straight line, timing free. " *
        "† = the three starting guesses reached different local optima (we report the cheapest). " *
        "Every energy solve reached the same optimum from all three guesses.*")
end

# ╔═╡ 03000039-5eac-4a11-9b0e-000300000039
md"""
Things to notice:

* **Straight is always more expensive**, by 15 % to more than 100 %, under both costs.
* **Some reaches (0°, 180°, 225°) are cheaper in the field than without it** (energy column: free < no field).
  The field does no work on the hand, but ``J^{\mathsf T}F\cdot\dot\theta = F\cdot\dot x = 0`` only says the field's
  power at the shoulder and at the elbow **sum** to zero. It can feed positive work into one joint while
  soaking it up at the other — and the cost only charges positive work, joint by joint. The field also
  supplies force *changes* for free. An energy-minimizer in a curl field gets a little free labour.
* **Work only is a badly posed problem here**: it curves, but the size and even the side of the bow jump around
  between directions, and different starting guesses land in different optima (†). Force rate is what makes the
  answer smooth and unique — same lesson as Lecture 2.

Watch one (Mattar & Ostry's 315° reference direction), with the field forces drawn:
"""

# ╔═╡ 03000040-5eac-4a11-9b0e-000300000040
arm_animation([mattar["dir_315_E_null"], mattar["dir_315_E_free"]]; labels=["no field", "clockwise curl field (b = 15)"],
              title="Energy-optimal 15 cm reaches in 400 ms, 315° direction", show_force=true)

# ╔═╡ 03000041-5eac-4a11-9b0e-000300000041
md"""
## Is it the distance? Is it where we are in space?

A fair suspicion: maybe the curvature depends on the reach — longer or shorter, or in another part of the
workspace — and Mattar & Ostry happened to sit somewhere special. Two sweeps.

**Distance.** Same start, three directions, 5 to 30 cm (stopping where the target would leave the workspace).
Either ``T`` stays at 0.40 s (solid), or ``T`` grows like ``D^{0.28}`` — the scaling of the energy-and-time optimum
in Lecture 5 — through 0.40 s at 15 cm (dashed).
"""

# ╔═╡ 03000042-5eac-4a11-9b0e-000300000042
let
    p1 = plot(title="perpendicular error", xlabel="reach distance (cm)", ylabel="PE (mm)", legend=:bottomleft)
    p2 = plot(title="relative: PE / distance", xlabel="reach distance (cm)", ylabel="PE / D", legend=false,
              yformatter=y -> @sprintf("%.0f%%", 100y))
    p3 = plot(title="extra cost of going straight", xlabel="reach distance (cm)", ylabel="", legend=false,
              yformatter=y -> @sprintf("%.0f%%", 100y), ylims=(0, 2))
    for (i, a) in enumerate((90, 180, 315)), (tm, ls) in (("fix", :solid), ("scaled", :dash))
        Ds = [D for D in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30) if haskey(mattar, "dist_$(a)_$(D)_$(tm)_E_free") &&
              solve_ok(mattar["dist_$(a)_$(D)_$(tm)_E_straight"])]
        pe = [pe_mm(mattar["dist_$(a)_$(D)_$(tm)_E_free"]) for D in Ds]
        plot!(p1, 100Ds, pe, c=PAL[i], ls=ls, marker=:circle, ms=4, label=(tm == "fix" ? "$(a)°" : ""))
        plot!(p2, 100Ds, -pe ./ (1000Ds), c=PAL[i], ls=ls, marker=:circle, ms=4)
        plot!(p3, 100Ds, [premium("dist_$(a)_$(D)_$(tm)_E") for D in Ds], c=PAL[i], ls=ls, marker=:circle, ms=4)
    end
    plot(p1, p2, p3, layout=(1, 3), size=(800, 300))
end

# ╔═╡ 03000043-5eac-4a11-9b0e-000300000043
md"""
The bow grows **roughly in proportion to the distance**. At 315° it is 35–40 % of ``D`` at every distance from
5 to 30 cm, with ``T`` fixed or scaled; at 90° it is 15–25 %; at 180° it slowly falls from ≈ 30 % (5 cm) to
15–20 % (30 cm). Going straight costs
≈ 25–40 % extra at 315° at every distance (60–170 % at 180°). So shorter reaches don't rescue "straight": a 5 cm
reach bows 1–2 cm, the same *shape* as a 30 cm one, just smaller. The thing that changes with direction is the
size of the effect, not its existence.

**Where in space.** Seven start postures (shoulder 30–60°, elbow 60–120°), up to four directions each,
15 cm in 400 ms (fewer where the target would leave the workspace). Each panel is centred on its start; the
inset numbers are where that start sits relative to the shoulder.
"""

# ╔═╡ 03000044-5eac-4a11-9b0e-000300000044
let
    panels = []
    for (sh, el) in mo_postures
        c = hand_position(arm_params(), deg2rad.([sh, el]))
        pl = plot(title="shoulder $(sh)°, elbow $(el)°", aspect_ratio=:equal, legend=false, ticks=false, framestyle=:box,
                  xlims=c[1] .+ (-0.19, 0.19), ylims=c[2] .+ (-0.19, 0.19), titlefontsize=9)
        for a in (0, 90, 135, 315)
            key = "loc_$(sh)_$(el)_$(a)_E"; haskey(mattar, key*"_free") || continue
            s = mattar[key*"_free"]; tg = s["task"].target
            plot!(pl, [c[1], tg[1]], [c[2], tg[2]], c=:gray70, lw=1, ls=:dot)
            plot!(pl, s["hand"][1, :], s["hand"][2, :], c=PAL[1], lw=2)
            scatter!(pl, [tg[1]], [tg[2]], c=:white, msc=INK, ms=4)
        end
        scatter!(pl, [c[1]], [c[2]], c=INK, ms=3)
        annotate!(pl, c[1] - 0.18, c[2] - 0.17, text(@sprintf("hand (%.0f, %.0f) cm", 100c[1], 100c[2]), 7, :left, INK))
        push!(panels, pl)
    end
    plot(panels..., layout=(2, 4), size=(800, 420))
end

# ╔═╡ 03000045-5eac-4a11-9b0e-000300000045
let
    rows = ["| start posture (shoulder, elbow) | 0° (right) | 90° (forward) | 135° | 315° |", "|:--|--:|--:|--:|--:|"]
    for (sh, el) in mo_postures
        cells = String[]
        for a in (0, 90, 135, 315)
            key = "loc_$(sh)_$(el)_$(a)_E"
            push!(cells, haskey(mattar, key*"_free") ?
                  @sprintf("%+.0f mm, %+.0f%%", pe_mm(mattar[key*"_free"]), 100premium(key)) : "—")
        end
        push!(rows, "| ($(sh)°, $(el)°)$( (sh, el) == (45, 90) ? " ← Mattar & Ostry" : "") | " * join(cells, " | ") * " |")
    end
    Markdown.parse(join(rows, "\n") * "\n\n*Each entry: energy-optimal PE · extra cost of the best straight path.*")
end

# ╔═╡ 03000046-5eac-4a11-9b0e-000300000046
md"""
Same answer everywhere: a 2–6 cm bow to the left on a 15 cm reach, and straight costs 17 % to 125 % more.
Where you are in space changes the numbers (the arm's inertia and how the field's torque divides between
the joints change with posture), but not the story.

**Duration.** The 350 ± 50 ms window is tight; what if we let it be faster or slower (15 cm, energy)?
"""

# ╔═╡ 03000047-5eac-4a11-9b0e-000300000047
let
    rows = ["| T (s) | 90°: PE | 90°: straight costs | 315°: PE | 315°: straight costs |", "|--:|--:|--:|--:|--:|"]
    for T in (0.30, 0.35, 0.40, 0.50, 0.65)
        k(a) = T == 0.40 ? "dir_$(a)_E" : "dur_$(a)_$(T)_E"
        push!(rows, @sprintf("| %.2f | %+.0f mm | %+.0f %% | %+.0f mm | %+.0f %% |", T, pe_mm(mattar[k(90)*"_free"]), 100premium(k(90)),
              pe_mm(mattar[k(315)*"_free"]), 100premium(k(315))))
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 03000048-5eac-4a11-9b0e-000300000048
md"""
The bow is about the same from 300 to 650 ms (at 90° and 650 ms the optimum even switches sides, to
another field-riding pattern — straight is still +20 %).

### And the robot?

Mattar & Ostry used an InMotion2, and their Methods don't say whether its inertia was compensated (see the
appendix for what that means). The figure above already has two versions: a 1 kg isotropic handle, and the
full **uncompensated** InMotion2 coupled to the hand. Both cut the bow — to roughly 1–3 cm — and both make
going straight relatively cheaper (+6 % to +41 % with the 1 kg handle, +6 % to +77 % with the robot), but neither makes straight optimal, and the bow stays on
the left, against the field. The sign problem doesn't go away.

So in the standard curl-field experiment, energy with time fixed predicts reaches that are clearly curved
*against* the field, and people make reaches that are nearly straight and slightly *with* it. We come back
to what might be missing at the end. First, the same question for a much bigger, faster reach.

## Further and faster: 30 cm in 0.65 s

The same arm reaching 30 cm straight ahead, from (0, 20) cm to (0, 50) cm, in 0.65 s, with a curl field of
13 N·s/m. This is bigger and faster than Mattar & Ostry's reaches (and about the size of Kistemaker et al.'s
reaches, Lecture 4), and the steering-wheel effect shows up at full strength.
"""

# ╔═╡ 03000007-5eac-4a11-9b0e-000300000007
curl = cached("L3_curl_ct0.07") do
    E = CostWeights(c_t=c_t3)                          # energy: work + force-rate-rate
    base(; kw...) = ReachTask(start=[0.0, 0.20], target=[0.0, 0.50], T=0.65; kw...)
    d = Dict{String,Any}()
    d["null"]       = solve_reach(base(), E)
    d["cw"]         = solve_reach(base(field=:curl_cw), E)
    d["ccw"]        = solve_reach(base(field=:curl_ccw), E)
    d["cw_work"]    = solve_reach(base(field=:curl_cw), CostWeights(c_t=0.0, eps=1e-6))
    # tracking problems: the hand must stay within ±2 mm of a given path SHAPE;
    # the optimizer still chooses the timing along it (so each path gets its best shot)
    for fld in (:curl_cw, :curl_ccw), A in (-0.05, 0.0, 0.05)
        d["track_$(fld)_$(A)"] = solve_reach(base(field=fld, track=bow(A)), E)
    end
    # the robot handle's own inertia
    for m in (0.5, 1.0, 2.0)
        d["cw_m$(m)"]       = solve_reach(base(field=:curl_cw, m_endpoint=m), E)
        d["cw_m$(m)_track"] = solve_reach(base(field=:curl_cw, m_endpoint=m, track=bow(0.0)), E)
        d["null_m$(m)"]     = solve_reach(base(m_endpoint=m), E)
    end
    # sensitivity: does the answer depend on the force-rate coefficient?
    for ct in (0.05, 0.07, 0.085, 0.15, 0.3), fld in (:curl_cw, :curl_ccw)
        o = solve_reach(base(field=fld), CostWeights(c_t=ct))
        d["ct_$(ct)_$(fld)"] = o
        d["ct_$(ct)_$(fld)_straight"] = solve_reach(base(field=fld, track=bow(0.0)), CostWeights(c_t=ct); guess=o)
    end
    d
end;

# ╔═╡ 03000008-5eac-4a11-9b0e-000300000008
md"""
Energy with ``c_t`` = 0.07, no field vs the two curl fields:
"""

# ╔═╡ 03000009-5eac-4a11-9b0e-000300000009
let
    sols = [curl["null"], curl["cw"], curl["ccw"]]
    labs = ["no field", "clockwise curl", "counter-clockwise curl"]
    p1 = plot(title="hand paths (energy-optimal)", xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal,
              xlims=(-0.17, 0.12), ylims=(0.17, 0.53), legend=:bottomleft)
    p2 = plot(title="hand speed", xlabel="time (s)", ylabel="m/s", legend=false)
    for (i, s) in enumerate(sols)
        plot!(p1, s["hand"][1, :], s["hand"][2, :], c=PAL[i], label=labs[i])
        plot!(p2, s["t"], s["speed"], c=PAL[i])
    end
    scatter!(p1, [0, 0], [0.2, 0.5], c=INK, ms=4, label="")
    plot(p1, p2, layout=grid(1, 2, widths=[0.42, 0.58]), size=(760, 360))
end

# ╔═╡ 03000010-5eac-4a11-9b0e-000300000010
md"""
It does **not** go straight. In the clockwise field the energy-optimal hand swings about 10 cm to the
**left**, moves *faster* than it would in the null field, and lets the field carry it around. Watch the
force arrows: the field (which is perpendicular to the velocity) is doing the steering.
"""

# ╔═╡ 03000011-5eac-4a11-9b0e-000300000011
arm_animation([curl["null"], curl["cw"]]; labels=["no field", "clockwise curl field"],
              title="Energy-optimal reaches, with and without the clockwise curl field", show_force=true)

# ╔═╡ 03000012-5eac-4a11-9b0e-000300000012
md"""
This is the free steering wheel at full strength. The bow of ≈ 10.5 cm on a 30 cm chord is an arc of radius
≈ 0.16 m; with the peak speed of ≈ 1.5 m/s that implies ``m = bR/v ≈ 1.4`` kg — about the effective mass
of the arm at the hand in this posture. Here a real single arc fits, so the optimum rides it almost all the way.
"""

# ╔═╡ 03000013-5eac-4a11-9b0e-000300000013
md"""
## Tracking problems: what does a straighter path cost?

Now force the issue. Keep the hand within ±2 mm of a prescribed path *shape* — straight, or bowed 5 cm
to either side — and let the optimizer choose the best timing along it (the speed profiles come out
bell-shaped). Every number below is the same energy (``c_W`` = 4.2, ``c_t`` = 0.07), in joules per reach.
"""

# ╔═╡ 03000014-5eac-4a11-9b0e-000300000014
let
    rows = ["| field | free optimum (bow) | straight | bowed 5 cm left | bowed 5 cm right |", "|:--|--:|--:|--:|--:|"]
    for (fld, key, name) in ((:curl_cw, "cw", "clockwise"), (:curl_ccw, "ccw", "counter-clockwise"))
        o = curl[key]
        dev = o["hand"][1, argmax(abs.(o["hand"][1, :]))]
        e(A) = curl["track_$(fld)_$(A)"]["E"]
        # bow(A) with A > 0 bows to the RIGHT of the travel direction (+x when reaching forward)
        push!(rows, @sprintf("| %s | **%.1f** (%+.1f cm) | %.1f | %.1f | %.1f |", name, o["E"], 100dev, e(0.0), e(-0.05), e(0.05)))
    end
    push!(rows, @sprintf("| none | %.1f | | | |", curl["null"]["E"]))
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 03000015-5eac-4a11-9b0e-000300000015
let
    p = plot(title="clockwise curl: hand paths of the tracking problems", xlabel="x (m)", ylabel="y (m)",
             aspect_ratio=:equal, xlims=(-0.15, 0.1), ylims=(0.17, 0.53), legend=:outerright, size=(620, 380))
    for (i, A) in enumerate((-0.05, 0.0, 0.05))
        s = curl["track_curl_cw_$(A)"]
        plot!(p, s["hand"][1, :], s["hand"][2, :], c=PAL[i+3], label=@sprintf("tracked %+.0f cm: E = %.1f J", 100A, s["E"]))
    end
    s = curl["cw"]; plot!(p, s["hand"][1, :], s["hand"][2, :], c=PAL[2], lw=3, label=@sprintf("free optimum: E = %.1f J", s["E"]))
    p
end

# ╔═╡ 03000016-5eac-4a11-9b0e-000300000016
md"""
Read the table carefully, because it is not a one-line story.

* **Fighting the field to go straight is expensive** in the clockwise field: the straight path costs
  roughly 40 % more than the free optimum. A 5 cm bow does not help either way — the optimum is not
  "a bit curved", it is a different strategy (fast, wide, field-steered).
* **The two directions are not mirror images.** The arm is not symmetric (the elbow points right). The
  counter-clockwise optimum is a stranger beast — it hooks left, slows almost to a stop halfway, and
  re-launches (two speed peaks: the green curves above) — and it saves only about a tenth over going straight.

### Does it depend on the force-rate coefficient?

No. Here is the same comparison across a six-fold range of ``c_t``:
"""

# ╔═╡ 03000017-5eac-4a11-9b0e-000300000017
let
    rows = ["| ``c_t`` | clockwise: optimum bow | clockwise: straight costs | counter-clockwise: optimum bow | counter-clockwise: straight costs |",
            "|--:|--:|--:|--:|--:|"]
    for ct in (0.05, 0.07, 0.085, 0.15, 0.3)
        c = [(curl["ct_$(ct)_$(f)"], curl["ct_$(ct)_$(f)_straight"]) for f in (:curl_cw, :curl_ccw)]
        dv(s) = 100*s["hand"][1, argmax(abs.(s["hand"][1, :]))]
        push!(rows, @sprintf("| %s | %+.1f cm | %+.0f %% | %+.1f cm | %+.0f %% |", ct, dv(c[1][1]), 100*(c[1][2]["E"]/c[1][1]["E"] - 1),
                             dv(c[2][1]), 100*(c[2][2]["E"]/c[2][1]["E"] - 1)))
    end
    Markdown.parse(join(rows, "\n") * "\n\n*\"straight costs\" = extra energy of the best straight path relative to the free optimum, same ``c_t``.*")
end

# ╔═╡ 03000018-5eac-4a11-9b0e-000300000018
md"""
In the clockwise field the field-riding optimum (≈ 9–11 cm bow) survives every coefficient, and a larger
``c_t`` only makes going straight *more* expensive — fighting the field is all torque change. (In the
counter-clockwise field the optimum flips sides at high ``c_t``, but straight is never optimal.)

## The handle matters: robot inertia

Real manipulanda add their own inertia at the hand. That changes the arithmetic of the free steering
wheel: a heavier hand turns on a larger radius (``R = mv/b``), so the arc the field can carry it around gets
*flatter* — the bow shrinks — and the free steering is worth less.
"""

# ╔═╡ 03000019-5eac-4a11-9b0e-000300000019
let
    ms = [0.0, 0.5, 1.0, 2.0]
    sol_for(m, suff) = m == 0 ? (suff == "" ? curl["cw"] : suff == "_track" ? curl["track_curl_cw_0.0"] : curl["null"]) :
                   (suff == "null" ? curl["null_m$(m)"] : curl["cw_m$(m)$(suff)"])
    dev = [100*abs(sol_for(m, "")["hand"][1, argmax(abs.(sol_for(m, "")["hand"][1, :]))]) for m in ms]
    Eopt = [sol_for(m, "")["E"] for m in ms]; Estr = [sol_for(m, "_track")["E"] for m in ms]; Enull = [sol_for(m, "null")["E"] for m in ms]
    p1 = plot(ms, dev, marker=:circle, ms=5, c=PAL[1], label="", title="how far the optimum bows", xlabel="handle mass (kg)", ylabel="max sideways deviation (cm)", ylims=(0, 12))
    p2 = plot(ms, Estr ./ Eopt .- 1, marker=:circle, ms=5, c=PAL[2], label="", title="extra cost of going straight", xlabel="handle mass (kg)", ylabel="E straight / E optimum − 1",
              yformatter=y -> @sprintf("%.0f%%", 100y), ylims=(0, 0.5))
    plot(p1, p2, layout=(1, 2), size=(760, 300))
end

# ╔═╡ 03000020-5eac-4a11-9b0e-000300000020
md"""
With a light handle, energy says: ride the field. With a 1–2 kg handle (typical of the robots these
experiments use) the optimal bow drops to a few centimetres — but going straight still costs noticeably more
than the optimum (read the right panel). Inertia tames the prediction; it does not make "straight" optimal.

## So is energy wrong about curl fields?

In Mattar & Ostry's task, and at every distance, direction and posture we tried, the energy optimum with time
fixed bows *against* the field, by a few centimetres on a 15 cm reach. Adapted humans are nearly straight and,
if anything, slightly *with* the field. That is a real mismatch, not a rounding error. Some honest candidates,
which you can test with this model:

* **Inertia of the real robot** — it shrinks the prediction a lot (to 1–3 cm at Mattar & Ostry's size) but never
  flips its sign. See the appendix for the robot's equations of motion.
* **Uncertainty.** The field-steered path only works if the field is exactly what you think it is. In these
  experiments the field is switched off on unannounced catch trials (1 in 10 for Mattar & Ostry); a path that
  relies on the field to reach the target misses badly when it vanishes. A straight path that fights the field
  fails gracefully. Energy-optimal *under uncertainty* is a different problem from energy-optimal.
* **Learning isn't finished.** People approach the field from the uncompensated side (pushed right) and stop
  when the error is small. A learner that reduces error, rather than one that searches for the energy optimum,
  would also end up slightly on the right.
* **Time is not really fixed.** People choose their own speed (Lecture 5). Here the timing window was enforced,
  so this one is less likely.

Keep this example in your pocket. Next lecture we look at a field that was **designed** to make a curved
path cheaper — and there, the energy cost that includes force rate says something very clean.
"""

# ╔═╡ 03000021-5eac-4a11-9b0e-000300000021
md"""
## Your turn

Pick a start posture, a reach, a field and (optionally) a path to track, then press **Solve**. The defaults
are Mattar & Ostry's 315° reach.

shoulder $(@bind sh_user Slider(20:5:70, default=45, show_value=true))° ·
elbow $(@bind el_user Slider(50:5:130, default=90, show_value=true))° ·
direction $(@bind dir_user Slider(0:45:315, default=315, show_value=true))° ·
distance $(@bind D_user Slider(0.05:0.01:0.30, default=0.15, show_value=true)) m

field $(@bind fld_user Select([:curl_cw => "clockwise curl", :curl_ccw => "counter-clockwise curl", :none => "no field"], default=:curl_cw)) ·
``b`` $(@bind b_user Slider(0.0:1.0:30.0, default=15.0, show_value=true)) N·s/m ·
handle mass $(@bind m_user Slider(0.0:0.25:3.0, default=0.0, show_value=true)) kg

duration ``T`` $(@bind T3_user Slider(0.25:0.05:1.2, default=0.40, show_value=true)) s ·
cost $(@bind cost_user Select(["E" => "work + force rate", "W" => "work only"], default="E")) ·
track a path? $(@bind track_user Select([nothing => "no — free optimum", 0.0 => "straight", -0.03 => "bow 3 cm left", 0.03 => "bow 3 cm right", -0.05 => "bow 5 cm left", 0.05 => "bow 5 cm right"], default=nothing))

$(@bind go3 CounterButton("Solve"))
"""

# ╔═╡ 03000022-5eac-4a11-9b0e-000300000022
user3 = go3 == 0 ? mattar["dir_315_E_free"] : let
    c = hand_position(arm_params(), deg2rad.([sh_user, el_user]))
    solve_reach(ReachTask(start=c, target=mo_target(c, dir_user, D_user), T=T3_user, field=fld_user, b_curl=b_user,
                          m_endpoint=m_user, track=(track_user === nothing ? nothing : bow(track_user))),
                cost_user == "E" ? CostWeights(c_t=c_t3) : CostWeights(c_t=0.0, eps=1e-6))
end;

# ╔═╡ 03000023-5eac-4a11-9b0e-000300000023
let s = user3
    st, tg = s["task"].start, s["task"].target
    p1 = plot(s["hand"][1, :], s["hand"][2, :], c=PAL[2], aspect_ratio=:equal, legend=false, xlabel="x (m)", ylabel="y (m)",
              title=@sprintf("PE %+.0f mm (max %+.0f mm)", pe_mm(s), 1000perp_error(s).max))
    plot!(p1, [st[1], tg[1]], [st[2], tg[2]], c=:gray70, ls=:dot)
    p2 = plot(s["t"], s["speed"], c=PAL[2], legend=false, xlabel="time (s)", ylabel="hand speed (m/s)",
              title=@sprintf("%s · E = %.2f J (work %.2f, force rate %.2f)", s["status"], s["E"], s["E_work"], s["E_fr"]))
    plot(p1, p2, layout=grid(1, 2, widths=[0.4, 0.6]), size=(760, 330))
end

# ╔═╡ 03000024-5eac-4a11-9b0e-000300000024
arm_animation([user3]; labels=["your solve"], title="your solve", show_force=true)

# ╔═╡ 03000025-5eac-4a11-9b0e-000300000025
md"""
### Things to try

1. **Turn ``b`` down** from 15 toward 0. Does the bow shrink smoothly, or is there a field strength where the
   optimum stops riding the field? Compare with ``R = mv/b``.
2. **Make it tiny**: 5 cm in 0.3 s. Is the bow still the same fraction of the reach?
3. **Track straight**, then a 3 cm bow to the *right* (with the field, like the people). How much does each cost
   relative to the free optimum?
4. **Work only** (cost menu). The field does no work on the hand; does the work-only optimum ignore it?
   (It doesn't: joint work is not hand work.)
5. **Make the robot heavy** (3 kg). Is going straight now essentially free?

### References

* Mattar AAG, Ostry DJ (2007) Modifiability of generalization in dynamics learning. *J Neurophysiol* 98:3321–3329.
* Shadmehr R, Mussa-Ivaldi FA (1994) Adaptive representation of dynamics during learning of a motor task. *J Neurosci* 14:3208–3224.
* Wong JD, Cluff T, Kuo AD (2021) The energetic basis for smooth human arm movements. *eLife* 10:e68013.
"""

# ╔═╡ Cell order:
# ╟─03000001-5eac-4a11-9b0e-000300000001
# ╟─03000002-5eac-4a11-9b0e-000300000002
# ╟─03000003-5eac-4a11-9b0e-000300000003
# ╠═03000004-5eac-4a11-9b0e-000300000004
# ╟─03000005-5eac-4a11-9b0e-000300000005
# ╠═03000006-5eac-4a11-9b0e-000300000006
# ╟─03000030-5eac-4a11-9b0e-000300000030
# ╠═03000031-5eac-4a11-9b0e-000300000031
# ╠═03000032-5eac-4a11-9b0e-000300000032
# ╟─03000049-5eac-4a11-9b0e-000300000049
# ╟─03000033-5eac-4a11-9b0e-000300000033
# ╟─03000034-5eac-4a11-9b0e-000300000034
# ╟─03000035-5eac-4a11-9b0e-000300000035
# ╟─03000036-5eac-4a11-9b0e-000300000036
# ╟─03000037-5eac-4a11-9b0e-000300000037
# ╟─03000038-5eac-4a11-9b0e-000300000038
# ╟─03000039-5eac-4a11-9b0e-000300000039
# ╟─03000040-5eac-4a11-9b0e-000300000040
# ╟─03000041-5eac-4a11-9b0e-000300000041
# ╟─03000042-5eac-4a11-9b0e-000300000042
# ╟─03000043-5eac-4a11-9b0e-000300000043
# ╟─03000044-5eac-4a11-9b0e-000300000044
# ╟─03000045-5eac-4a11-9b0e-000300000045
# ╟─03000046-5eac-4a11-9b0e-000300000046
# ╟─03000047-5eac-4a11-9b0e-000300000047
# ╟─03000048-5eac-4a11-9b0e-000300000048
# ╠═03000007-5eac-4a11-9b0e-000300000007
# ╟─03000008-5eac-4a11-9b0e-000300000008
# ╟─03000009-5eac-4a11-9b0e-000300000009
# ╟─03000010-5eac-4a11-9b0e-000300000010
# ╟─03000011-5eac-4a11-9b0e-000300000011
# ╟─03000012-5eac-4a11-9b0e-000300000012
# ╟─03000013-5eac-4a11-9b0e-000300000013
# ╟─03000014-5eac-4a11-9b0e-000300000014
# ╟─03000015-5eac-4a11-9b0e-000300000015
# ╟─03000016-5eac-4a11-9b0e-000300000016
# ╟─03000017-5eac-4a11-9b0e-000300000017
# ╟─03000018-5eac-4a11-9b0e-000300000018
# ╟─03000019-5eac-4a11-9b0e-000300000019
# ╟─03000020-5eac-4a11-9b0e-000300000020
# ╟─03000021-5eac-4a11-9b0e-000300000021
# ╟─03000022-5eac-4a11-9b0e-000300000022
# ╟─03000023-5eac-4a11-9b0e-000300000023
# ╟─03000024-5eac-4a11-9b0e-000300000024
# ╟─03000025-5eac-4a11-9b0e-000300000025
