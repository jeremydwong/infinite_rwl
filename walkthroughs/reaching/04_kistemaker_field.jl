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

# ╔═╡ 04000001-5eac-4a11-9b0e-000400000001
md"""
# Lecture 4 — A field built to make curved paths cheap

*Energy and time is all you need · part 4 of 5*

Kistemaker, Wong & Gribble (2010) built a robot force field with one purpose: to make the
**minimum-energy hand path clearly curved**, so that if people minimize energy they would have to
visibly change how they reach. They did not. After hundreds — in one experiment more than 1,600 — reaches
in the field, people kept moving essentially straight, delivering ≈ 5.3 J to the robot per reach when
≈ 2.7 J would have done. Even after practising the curved, cheaper path, they went back to straight.
The paper's title says it plainly: *The central nervous system does not minimize energy cost in arm movements.*

That conclusion rests on what "energy" means. Their minimum-energy path minimized the **mechanical
energy delivered to the robot** (a Hill-type muscle model with the Umberger energetics gave the same
verdict for metabolic energy — a model estimate, not a measurement). Lecture 2
gave us a different energy: work **plus the cost of changing force**. The question for this lecture:

**Under the work + force-rate-rate energy, is the straight path people chose actually the optimum?**
"""

# ╔═╡ 04000002-5eac-4a11-9b0e-000400000002
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

# ╔═╡ 04000003-5eac-4a11-9b0e-000400000003
md"""
## The Kistemaker field

For outward reaches (``y`` forward, ``x`` to the right) the robot applies (their Eq. 2)

```math
F_x = 0, \qquad F_y = b\,(-\dot y + \dot x)\,\lvert y_{\text{target}} - y\rvert, \qquad b = 150\ \text{N·s/m}^2 .
```

* Moving **forward** (``\dot y > 0``) is **resisted**.
* Moving **right** (``\dot x > 0``) produces a **forward, assisting** push.
* Everything scales with the distance still to go, so the field fades to zero at the target.

So the cheap strategy, in terms of work against the robot, is to head out to the right early — where the
resistance is strongest and a rightward velocity cancels it — and come back left late, where the
field is weak. They chose 30 cm reaches; the robot added an isotropic 1 kg of inertia at the handle (we
include it); their energy map (Fig. 1B) uses 400 ms reaches, and so do we.
"""

# ╔═╡ 04000004-5eac-4a11-9b0e-000400000004
md"""
### Reproducing their energy map

Their Figure 1B shows the energy delivered to the robot, ``E_{\text{ext}} = \int F_r\cdot\dot s\,dt``, along a family of
minimum-jerk paths bowed to the left and right. We can compute the same thing directly from the field
for our family of bows (no optimization needed — the path and timing are given):
"""

# ╔═╡ 04000005-5eac-4a11-9b0e-000400000005
"Energy delivered to the robot along a min-jerk path bowed by A (m), duration T (s)."
function external_energy(A, T; start=[0.0, 0.20], target=[0.0, 0.50], n=4001)
    path = curved_path(start, target, A)
    s = range(0, 1, length=n); h = step(s)*T
    X = hcat(path.(s)...)
    W = 0.0
    for k in 2:n-1
        v = (X[:, k+1] .- X[:, k-1]) ./ (2h)                      # hand velocity
        F = field_force(:kistemaker, X[:, k], v; target_y=target[2])  # robot force on the hand
        W += -dot(F, v)*h                                         # the hand does work ON the robot
    end
    W
end

# ╔═╡ 04000006-5eac-4a11-9b0e-000400000006
let
    As = -0.06:0.02:0.18
    p = plot(title="energy delivered to the robot along min-jerk paths", xlabel="x (m)", ylabel="y (m)",
             aspect_ratio=:equal, xlims=(-0.1, 0.22), ylims=(0.17, 0.53), legend=:outerright, size=(700, 380))
    E55 = [external_energy(A, 0.40) for A in As]     # 400 ms, as in their Fig. 1B
    cg = cgrad(["#d6e6f8", "#2a78d6", "#0b2e5c"])                   # one hue, light → dark = more energy
    lo, hi = extrema(E55)
    for (A, E) in zip(As, E55)
        path = curved_path([0.0, 0.20], [0.0, 0.50], A); s = range(0, 1, length=100)
        X = hcat(path.(s)...)
        plot!(p, X[1, :], X[2, :], c=cg[(E - lo)/(hi - lo)], lw=2.5, label=@sprintf("bow %+3.0f cm: %.1f J", 100A, E))
    end
    scatter!(p, [0, 0], [0.2, 0.5], c=INK, ms=4, label="")
    p
end

# ╔═╡ 04000007-5eac-4a11-9b0e-000400000007
md"""
The gradient is the one in their figure: bowing right is cheaper, bowing left is dearer, and the cheapest
paths swing far to the right.
"""

# ╔═╡ 04000008-5eac-4a11-9b0e-000400000008
md"""
## Two energies, two optima

Now optimize. Same arm as before, with the robot's 1 kg at the hand and the Kistemaker field, T = 0.40 s:

* **work only** — ``c_W\int[\dot\theta\tau]_+``: the energy-is-work view (positive *joint* work: the work
  done against the robot **plus** the work of speeding up the arm and the handle);
* **work + force-rate-rate** — with ``c_W`` = 4.2 and **``c_t`` = 0.07**, the value for discrete,
  time-constrained reaches used in Lecture 3 (Wong et al.'s cyclic value 0.085 gives the same ranking:
  straight within 0.5 % of the optimum, ±5 cm about double).
"""

# ╔═╡ 04000009-5eac-4a11-9b0e-000400000009
kist = cached("L4_kistemaker_T0.40") do
    E  = CostWeights(c_t=0.07)                   # work + force-rate-rate
    Wo = CostWeights(c_t=0.0, eps=1e-6)          # work only
    base(; kw...) = ReachTask(start=[0.0, 0.20], target=[0.0, 0.50], T=0.40, m_endpoint=1.0; kw...)
    d = Dict{String,Any}()
    d["null"]      = solve_reach(base(), E)
    d["work_opt"]  = solve_reach(base(field=:kistemaker), Wo)
    d["energy_opt"] = solve_reach(base(field=:kistemaker), E)
    # tracking problems under the full energy: straight, ±5 cm, and the work-only optimum's path
    for A in (-0.05, 0.0, 0.05)
        d["track_$(A)"] = solve_reach(base(field=:kistemaker, track=bow(A)), E; guess=d["energy_opt"])
    end
    fit = fit_offset(d["work_opt"])              # the work-only path as a polynomial shape
    d["track_work_path"] = solve_reach(base(field=:kistemaker, track=fit.offset), E; guess=d["work_opt"])
    d["work_path_fit_rms"] = fit.rms
    # Kistemaker et al.'s minimal-energy path (their Fig. 1B, solid blue): bows ≈ 16 cm right, peaking about
    # two-thirds of the way out. Approximated by offset ∝ λ⁶(1−λ)³ (peak at λ = 2/3).
    kpath = λ -> 0.16*(λ^6*(1 - λ)^3)/((2/3)^6*(1/3)^3)
    d["track_kistemaker_path"] = solve_reach(base(field=:kistemaker, track=kpath), E; guess=d["energy_opt"])
    # and the reverse: what does the energy-optimal (straight) path cost under work only?
    d["track_straight_workcost"] = solve_reach(base(field=:kistemaker, track=bow(0.0)), Wo)
    d
end;

# ╔═╡ 04000010-5eac-4a11-9b0e-000400000010
let
    p1 = plot(title="hand paths in the Kistemaker field", xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal,
              xlims=(-0.12, 0.2), ylims=(0.17, 0.53), legend=:bottomleft)
    p2 = plot(title="hand speed", xlabel="time (s)", ylabel="m/s", legend=false)
    for (i, (k, lab)) in enumerate((("energy_opt", "optimum: work + force rate"), ("work_opt", "optimum: work only"), ("null", "no field")))
        s = kist[k]
        plot!(p1, s["hand"][1, :], s["hand"][2, :], c=PAL[i], label=lab, ls=(k == "null" ? :dash : :solid))
        plot!(p2, s["t"], s["speed"], c=PAL[i], ls=(k == "null" ? :dash : :solid))
    end
    kp = kist["track_kistemaker_path"]
    plot!(p1, kp["hand"][1, :], kp["hand"][2, :], c=INK, ls=:dot, lw=1.5, label="Kistemaker et al.'s path (approx.)")
    scatter!(p1, [0, 0], [0.2, 0.5], c=INK, ms=4, label="")
    plot(p1, p2, layout=grid(1, 2, widths=[0.45, 0.55]), size=(760, 360))
end

# ╔═╡ 04000011-5eac-4a11-9b0e-000400000011
let w = kist["work_opt"], e = kist["energy_opt"]
    bow_w = 100*maximum(w["hand"][1, :])
    dev_e = 100*e["hand"][1, argmax(abs.(e["hand"][1, :]))]
    md"""
    The **work-only** optimum bows right by only **$(round(bow_w, digits=1)) cm**, delivering $(round(w["W_ext"], digits=2)) J
    to the robot (vs $(round(e["W_ext"], digits=2)) J). Why not the ≈ 16 cm of Kistemaker et al.'s minimal-energy path? Theirs
    minimizes only the energy delivered **to the robot**. Our work counts every joule of positive joint work, and in a
    400 ms reach a big share of that goes into speeding up the arm and the 1 kg handle; a wide detour makes the hand
    travel farther and faster, so the arm's own kinetic energy eats the saving.

    The **work + force-rate-rate** optimum goes **straight**: its largest sideways excursion is
    **$(round(dev_e, digits=1)) cm**. Subjects in the experiment deviated by −7 to −8 mm, in the null field and
    the force field alike.
    """
end

# ╔═╡ 04000012-5eac-4a11-9b0e-000400000012
arm_animation([kist["energy_opt"], kist["work_opt"]]; labels=["optimum: work + force rate", "optimum: work only"],
              title="Kistemaker field: the two optima (arrows: robot force on the hand)", show_force=true)

# ╔═╡ 04000013-5eac-4a11-9b0e-000400000013
md"""
## Tracking problems: what do the curved paths really cost?

To make the comparison airtight, solve **tracking problems** under the full energy: the hand must stay
within ±2 mm of a given path shape — straight, bowed 5 cm left or right, the work-only optimum's own path,
or **Kistemaker et al.'s minimal-energy path** (≈ 16 cm right, their Fig. 1B; we approximate its shape) — while
the optimizer picks the cheapest timing and torques it can along that path. Each path gets its best possible shot.
"""

# ╔═╡ 04000014-5eac-4a11-9b0e-000400000014
begin
    rows4 = [("free optimum (work + force rate)", "energy_opt"), ("straight", "track_0.0"),
             ("bowed 5 cm right", "track_0.05"), ("bowed 5 cm left", "track_-0.05"),
             ("work-only optimum's path", "track_work_path"),
             ("Kistemaker et al.'s minimal-energy path (≈ 16 cm)", "track_kistemaker_path")]
    Eref = kist["energy_opt"]["E"]
    tab = ["| path | E, work + force rate (J) | vs. optimum | of which work (J) | of which force rate (J) | energy to robot (J) | max sideways (cm) |",
           "|:--|--:|--:|--:|--:|--:|--:|"]
    for (name, k) in rows4
        s = kist[k]; dev = s["hand"][1, argmax(abs.(s["hand"][1, :]))]
        push!(tab, @sprintf("| %s | **%.1f** | %+.0f %% | %.1f | %.1f | %.2f | %+.1f |", name, s["E"], 100*(s["E"]/Eref - 1),
                            s["E_work"], s["E_fr"], s["W_ext"], 100dev))
    end
    Markdown.parse(join(tab, "\n"))
end

# ╔═╡ 04000015-5eac-4a11-9b0e-000400000015
let
    names = ["optimum", "straight", "5 cm right", "5 cm left", "work-only path", "Kistemaker path"]
    Es = [kist[k]["E"] for (_, k) in rows4]
    Ew = [kist[k]["E_work"] for (_, k) in rows4]; Ef = [kist[k]["E_fr"] for (_, k) in rows4]
    # stacked bars: draw the total in the force-rate colour, then the work part over it from zero
    p = bar(names, Ew .+ Ef, label="force-rate part", c=PAL[2], lw=0, bar_width=0.6, ylabel="energy per reach (J)",
            title="full energy of each path in the Kistemaker field", size=(720, 320), legend=:topleft)
    bar!(p, names, Ew, label="work part", c=PAL[1], lw=0, bar_width=0.6)
    p
end

# ╔═╡ 04000016-5eac-4a11-9b0e-000400000016
md"""
Now the picture is clean (numbers in the table above):

* The **straight** path is within about half a percent of the optimum. Under the work + force-rate energy,
  the straight reach people made *is* the energetic optimum.
* Bowing **5 cm either way roughly doubles the energy**. Bowing right saves a little work against the
  robot, but the extra sideways acceleration and deceleration have to be built and unbuilt with joint
  torque, and that force-rate bill swamps the saving.
* **Kistemaker et al.'s minimal-energy path** costs several times the straight reach. It is the cheapest path
  for the robot's energy meter and among the most expensive for the arm.

In other words, the 2010 result is a strong test of **work** as the energetic objective, and work fails it.
It is not a test of energy once energy includes the cost of changing force — and that energy predicts
exactly what people did.
"""

# ╔═╡ 04000017-5eac-4a11-9b0e-000400000017
let
    s1 = kist["track_straight_workcost"]; s2 = kist["work_opt"]
    md"""
    !!! note "And the other way round"
        Score the straight path with the **work-only** cost and it is worse than the work-only optimum, as it must be:
        $(round(s1["E"], digits=2)) J vs $(round(s2["E"], digits=2)) J. Each cost prefers its own optimum; the
        question is which cost people pay, and the reaches answer it. (The work-only path was imposed through a
        polynomial fit of its shape, rms error $(round(1000*kist["work_path_fit_rms"], digits=2)) mm.)
    """
end

# ╔═╡ 04000018-5eac-4a11-9b0e-000400000018
md"""
## Your turn

Change the field strength, the movement time, the weight on force rate, or the path, and press **Solve**.

field ``b`` $(@bind b4 Slider(0.0:10.0:300.0, default=150.0, show_value=true)) N·s/m² ·
duration ``T`` $(@bind T4 Slider(0.30:0.05:1.0, default=0.40, show_value=true)) s ·
force-rate coefficient ``c_t`` = 0.07 × $(@bind ct4 Select([0.0 => "0 (work only)", 0.1 => "0.1", 1.0 => "1", 3.0 => "3"], default=1.0))

track a path? $(@bind track4 Select([nothing => "no — free optimum", 0.0 => "straight", 0.05 => "bow 5 cm right", -0.05 => "bow 5 cm left", 0.10 => "bow 10 cm right"], default=nothing))

$(@bind go4 CounterButton("Solve"))
"""

# ╔═╡ 04000019-5eac-4a11-9b0e-000400000019
user4 = go4 == 0 ? kist["energy_opt"] :
    solve_reach(ReachTask(start=[0.0, 0.20], target=[0.0, 0.50], T=T4, field=:kistemaker, b_kist=b4, m_endpoint=1.0,
                          track=(track4 === nothing ? nothing : bow(track4))),
                CostWeights(c_t=0.07*ct4, eps=(ct4 == 0 ? 1e-6 : 1e-7)));

# ╔═╡ 04000020-5eac-4a11-9b0e-000400000020
let s = user4
    dev = s["hand"][1, argmax(abs.(s["hand"][1, :]))]
    p1 = plot(s["hand"][1, :], s["hand"][2, :], c=PAL[1], aspect_ratio=:equal, legend=false, xlabel="x (m)", ylabel="y (m)",
              title=@sprintf("max sideways %+.1f cm", 100dev), xlims=(-0.15, 0.2), ylims=(0.17, 0.53))
    p2 = plot(s["t"], s["speed"], c=PAL[1], legend=false, xlabel="time (s)", ylabel="hand speed (m/s)",
              title=@sprintf("%s · E = %.1f J (work %.1f, force rate %.1f) · to robot %.2f J", s["status"], s["E"], s["E_work"], s["E_fr"], s["W_ext"]))
    plot(p1, p2, layout=grid(1, 2, widths=[0.4, 0.6]), size=(760, 330))
end

# ╔═╡ 04000021-5eac-4a11-9b0e-000400000021
arm_animation([user4]; labels=["your solve"], title="your solve", show_force=true)

# ╔═╡ 04000022-5eac-4a11-9b0e-000400000022
md"""
### Things to try

1. **Dial the force-rate cost** from 0 to 1×. Find where the optimum switches from curved to straight.
   How does that compare with ``c_t`` measured by Wong et al. (2021)?
2. **Slow the reach down** (``T`` = 0.8 s). Force-rate costs fall faster with time than work against a
   viscous field does — does the curved strategy come back at long durations?
3. **Strengthen the field** (``b`` = 300). Is there a field strength at which even the full energy says "curve"?
   That would be the experiment to run next.

### References

* Kistemaker DA, Wong JD, Gribble PL (2010) The central nervous system does not minimize energy cost in arm movements. *J Neurophysiol* 104:2985–2994.
* Wong JD, Cluff T, Kuo AD (2021) The energetic basis for smooth human arm movements. *eLife* 10:e68013.
"""

# ╔═╡ Cell order:
# ╟─04000001-5eac-4a11-9b0e-000400000001
# ╟─04000002-5eac-4a11-9b0e-000400000002
# ╟─04000003-5eac-4a11-9b0e-000400000003
# ╟─04000004-5eac-4a11-9b0e-000400000004
# ╠═04000005-5eac-4a11-9b0e-000400000005
# ╟─04000006-5eac-4a11-9b0e-000400000006
# ╟─04000007-5eac-4a11-9b0e-000400000007
# ╟─04000008-5eac-4a11-9b0e-000400000008
# ╠═04000009-5eac-4a11-9b0e-000400000009
# ╟─04000010-5eac-4a11-9b0e-000400000010
# ╟─04000011-5eac-4a11-9b0e-000400000011
# ╟─04000012-5eac-4a11-9b0e-000400000012
# ╟─04000013-5eac-4a11-9b0e-000400000013
# ╟─04000014-5eac-4a11-9b0e-000400000014
# ╟─04000015-5eac-4a11-9b0e-000400000015
# ╟─04000016-5eac-4a11-9b0e-000400000016
# ╟─04000017-5eac-4a11-9b0e-000400000017
# ╟─04000018-5eac-4a11-9b0e-000400000018
# ╟─04000019-5eac-4a11-9b0e-000400000019
# ╟─04000020-5eac-4a11-9b0e-000400000020
# ╟─04000021-5eac-4a11-9b0e-000400000021
# ╟─04000022-5eac-4a11-9b0e-000400000022
