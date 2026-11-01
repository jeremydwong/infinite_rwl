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

This lecture does two things:

1. shows how a force **at the hand** enters an arm model as joint torques, ``\tau_{\text{field}} = J^{\mathsf T} F``;
2. asks what the **energy** objective of Lecture 2 predicts in a curl field, with movement time fixed.

The second one has a twist. A curl field never does any work on the hand. You might think energy
should therefore ignore it. It does not — and what it does instead is worth looking at carefully.
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
    q = ik(p, [0.0, 0.20])                 # joint angles with the hand at (0, 0.20) m
    J = hand_jacobian(p, q)
    F = [10.0, 0.0]                        # 10 N to the right
    τ = J' * F
    md"""
    joint angles ``\theta`` = ($(round(rad2deg(q[1]), digits=1))°, $(round(rad2deg(q[2]), digits=1))°) ·
    ``J^{\mathsf T}F`` = (**$(round(τ[1], digits=2))**, **$(round(τ[2], digits=2))**) N·m at (shoulder, elbow).
    A sideways push at the hand twists *both* joints — the arm has to answer it with both.
    """
end

# ╔═╡ 03000005-5eac-4a11-9b0e-000300000005
md"""
## The curl field

```math
F = B\,\dot x, \qquad B = b\begin{bmatrix}0 & 1\\ -1 & 0\end{bmatrix}\ (\text{clockwise}),\qquad b = 13\ \text{N·s/m}.
```

The force is always **perpendicular** to the hand's velocity, rotating it clockwise (the counter-clockwise
field is ``-B``). Moving straight ahead at 1 m/s you are pushed sideways with 13 N.

Because ``F \perp \dot x``,

```math
P_{\text{field}} = F\cdot\dot x = \dot x^{\mathsf T} B\,\dot x = 0 \quad (B \text{ is skew-symmetric}).
```

**A curl field never does work on the hand.** Not on any path, not at any instant. So a cost that only
counted mechanical work done on the outside world would not care about the field at all.

But the arm does not pay for work on the outside world; it pays for **joint** work (positive work at each
joint, Lecture 2) and for **changing joint torque**. Resisting a 13 N sideways push needs joint torques that
rise and fall as the hand speeds up and slows down — and those cost energy. So what does the energy optimum do?

!!! note "The energy used in this lecture"
    ``E = \sum_i \int \big(c_W[\dot\theta_i\tau_i]_+ + c_t[\ddot\tau_i]_+\big)\,dt`` with ``c_W = 4.2`` and
    **``c_t = 0.07``**. Lecture 2 used ``c_t = 0.085``, the value fitted to *cyclic* reaching (Wong et al. 2021);
    for discrete, time-constrained reaches the coefficient is plausibly somewhat lower, ≈ 0.07. The answer
    turns out not to depend on it — see the sensitivity table further down.
"""

# ╔═╡ 03000006-5eac-4a11-9b0e-000300000006
c_t3 = 0.07                                            # force-rate-rate coefficient for this lecture

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
## What energy predicts

Same arm, same 30 cm reach in 0.65 s, energy with ``c_t`` = 0.07. The only change is the field.
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
### Why: the curl field is a free steering wheel

Take away the arm for a moment and think of the hand as a point mass ``m`` with no muscle force at all.
The only force is ``F = B\dot x``: always perpendicular to the velocity, magnitude ``b\,|\dot x|``. That is exactly
the force on a charged particle in a magnetic field, and the motion is the same: **constant speed on a circle**,

```math
m\frac{v^2}{R} = b\,v \quad\Longrightarrow\quad R = \frac{m\,v}{b}.
```

So a hand launched at the right speed and angle is carried around an arc **to the target for free** — the
field supplies the entire centripetal force, and the muscles only have to start and stop the hand. The
energy-optimal path does a version of this. Its sagitta of ≈ 10.5 cm on a 30 cm chord is an arc of radius
≈ 0.16 m, and with the peak speed of ≈ 1.5 m/s that implies ``m = bR/v ≈ 1.4`` kg — about the effective mass
of the arm at the hand in this posture.

That is a genuinely energetic answer: the field does no work, but it can do **steering**, and steering
with your own muscles costs torque, torque rate, and joint work.
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
wheel: a heavier hand needs a larger radius (``R = mv/b``), so to reach a target 30 cm away it has to
swing wider — and somewhere it stops being worth it.
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

Adapted humans in curl fields move **nearly** straight, with small residual curvature. With a light handle,
the energy objective, taken literally and with time fixed, predicts much more than that. Some honest
candidates for the gap, which you can test with this model:

* **Inertia of the real robot** (above) — the prediction is very sensitive to it.
* **Uncertainty.** The field-steered path only works if the field is exactly what you think it is. In these
  experiments the field is switched off on unannounced catch trials; a path that relies on the field to reach
  the target misses badly when it vanishes. A straight path that fights the field fails gracefully.
  Energy-optimal *under uncertainty* is a different problem from energy-optimal.
* **Time is not really fixed.** People choose their own speed (Lecture 5); the steering trick needs a high
  speed (``v = bR/m``), which a cost of time and effort may not favour.

Keep this example in your pocket. Next lecture we look at a field that was **designed** to make a curved
path cheaper — and there, the energy cost that includes force rate says something very clean.
"""

# ╔═╡ 03000021-5eac-4a11-9b0e-000300000021
md"""
## Your turn

Choose a field, its strength, the handle mass and (optionally) a path to track, then press **Solve**.

field $(@bind fld_user Select([:curl_cw => "clockwise curl", :curl_ccw => "counter-clockwise curl", :none => "no field"], default=:curl_cw)) ·
``b`` $(@bind b_user Slider(0.0:1.0:30.0, default=13.0, show_value=true)) N·s/m ·
handle mass $(@bind m_user Slider(0.0:0.25:3.0, default=0.0, show_value=true)) kg

duration ``T`` $(@bind T3_user Slider(0.4:0.05:1.2, default=0.65, show_value=true)) s ·
track a path? $(@bind track_user Select([nothing => "no — free optimum", 0.0 => "straight", -0.05 => "bow 5 cm left", 0.05 => "bow 5 cm right", -0.10 => "bow 10 cm left", 0.10 => "bow 10 cm right"], default=nothing))

$(@bind go3 CounterButton("Solve"))
"""

# ╔═╡ 03000022-5eac-4a11-9b0e-000300000022
user3 = go3 == 0 ? curl["cw"] :
    solve_reach(ReachTask(start=[0.0, 0.20], target=[0.0, 0.50], T=T3_user, field=fld_user, b_curl=b_user,
                          m_endpoint=m_user, track=(track_user === nothing ? nothing : bow(track_user))), CostWeights(c_t=c_t3));

# ╔═╡ 03000023-5eac-4a11-9b0e-000300000023
let s = user3
    dev = s["hand"][1, argmax(abs.(s["hand"][1, :]))]
    p1 = plot(s["hand"][1, :], s["hand"][2, :], c=PAL[2], aspect_ratio=:equal, legend=false, xlabel="x (m)", ylabel="y (m)",
              title=@sprintf("max deviation %+.1f cm", 100dev), xlims=(-0.17, 0.17), ylims=(0.17, 0.53))
    p2 = plot(s["t"], s["speed"], c=PAL[2], legend=false, xlabel="time (s)", ylabel="hand speed (m/s)",
              title=@sprintf("%s · E = %.2f J (work %.2f, force rate %.2f)", s["status"], s["E"], s["E_work"], s["E_fr"]))
    plot(p1, p2, layout=grid(1, 2, widths=[0.4, 0.6]), size=(760, 330))
end

# ╔═╡ 03000024-5eac-4a11-9b0e-000300000024
arm_animation([user3]; labels=["your solve"], title="your solve", show_force=true)

# ╔═╡ 03000025-5eac-4a11-9b0e-000300000025
md"""
### Things to try

1. **Turn ``b`` down** from 13 toward 0. At what field strength does the optimum stop riding the field?
   Compare with ``R = mv/b``: what radius would the trick need, and is it bigger than the 15 cm half-chord?
2. **Slow down** (``T`` = 1.0 s). Does the bow grow or shrink? Why — what does ``R = mv/b`` say about slow hands?
3. **Work only.** Edit the cell above to use `CostWeights(c_t=0.0, eps=1e-6)`. The field does no work; does
   the work-only optimum ignore it? (It doesn't: joint work is not hand work. Compare `curl["cw_work"]`.)
4. **Make the robot heavy** (3 kg). Is going straight now essentially free?

### References

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
