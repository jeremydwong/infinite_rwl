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

# ╔═╡ 05000001-5eac-4a11-9b0e-000500000001
md"""
# Lecture 5 — Energy and time is all you need

*Energy and time is all you need · part 5 of 5*

Every reach so far had its duration handed to it: 0.65 s, 0.55 s. That is a cheat. Nobody tells
your arm how long to take, and if you minimize energy alone with time free, the answer is to take
forever — a slower reach is always cheaper. Real reaches are fast and remarkably consistent in
duration, and longer reaches take longer.

The last ingredient is a **cost of time** (Shadmehr et al. 2016; Wong et al. 2021):

```math
J = E + k\,T, \qquad E = \sum_i\int_0^T \Big(c_W[\dot\theta_i\tau_i]_+ + c_t[\ddot\tau_i]_+\Big)\,dt .
```

Spend energy to save time, or time to save energy. With ``k`` = 25 J/s (Wong et al. 2021) and the same
``c_W``, ``c_t`` as before, we let the optimizer choose ``T`` as well as everything else — and see whether
energy and time, alone, predict how long a reach takes.
"""

# ╔═╡ 05000002-5eac-4a11-9b0e-000500000002
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

# ╔═╡ 05000003-5eac-4a11-9b0e-000500000003
md"""
## Making time a decision variable

The model has used normalized time all along: ``s \in [0, 1]`` and ``t = T s``, so every derivative
constraint reads ``\frac{d}{ds}(\cdot) = T\,\frac{d}{dt}(\cdot)`` and every integral is ``T\int_0^1 (\cdot)\,ds``.
With ``T`` fixed it is just a number. To free it, ``T`` becomes an ordinary (finite) decision variable with
bounds, and the objective picks up ``kT``. In `solve_reach` that is these lines:
"""

# ╔═╡ 05000004-5eac-4a11-9b0e-000500000004
source_excerpt("free_T = task.T === nothing", "# ---------------- states and control")

# ╔═╡ 05000005-5eac-4a11-9b0e-000500000005
source_excerpt("# ---------------- objective", "# ---------------- initial guess")

# ╔═╡ 05000006-5eac-4a11-9b0e-000500000006
time5 = cached("L5_energy_time") do
    E(k) = CostWeights(k=k)
    start = [0.0, 0.20]
    d = Dict{String,Any}()
    # 1. free duration across reach distances (warm-starting each from the previous)
    Ds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    prev = nothing
    for D in Ds
        prev = solve_reach(ReachTask(start=start, target=[0.0, 0.20 + D], T=nothing), E(25.0); guess=prev)
        d["free_D$(D)"] = prev
    end
    d["Ds"] = Ds
    # 2. fixed-duration energy E*(T) for the 30 cm reach
    Ts = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80, 0.90, 1.00, 1.20]
    d["Ts"] = Ts
    d["Estar"] = [solve_reach(ReachTask(start=start, target=[0.0, 0.50], T=T), CostWeights())["E"] for T in Ts]
    # 3. different prices of time, 30 cm
    ks = [5.0, 10.0, 25.0, 50.0, 100.0]
    d["ks"] = ks
    for k in ks
        d["free_k$(k)"] = solve_reach(ReachTask(start=start, target=[0.0, 0.50], T=nothing), E(k); guess=d["free_D0.3"])
    end
    # 4. free time in the Kistemaker field vs the null field (1 kg handle, as in Lecture 4)
    d["kist_null"]  = solve_reach(ReachTask(start=start, target=[0.0, 0.50], T=nothing, m_endpoint=1.0), E(25.0))
    d["kist_field"] = solve_reach(ReachTask(start=start, target=[0.0, 0.50], T=nothing, m_endpoint=1.0, field=:kistemaker), E(25.0))
    d
end;

# ╔═╡ 05000007-5eac-4a11-9b0e-000500000007
md"""
## Duration falls out of the optimization
"""

# ╔═╡ 05000008-5eac-4a11-9b0e-000500000008
let
    Ds = time5["Ds"]; sols = [time5["free_D$(D)"] for D in Ds]
    cg = cgrad(["#9cc3ee", "#2a78d6", "#0b2e5c"])                      # light → dark = longer reach
    p1 = plot(title="hand speed, duration chosen by the optimizer", xlabel="time (s)", ylabel="m/s", legend=:topleft)
    for (i, (D, s)) in enumerate(zip(Ds, sols))
        plot!(p1, s["t"], s["speed"], c=cg[(i - 1)/(length(Ds) - 1)], label=@sprintf("%2.0f cm → %.2f s", 100D, s["T"]))
    end
    p2 = plot(title="…and all the same shape", xlabel="t / T", ylabel="speed / peak", legend=false)
    for (i, s) in enumerate(sols)
        plot!(p2, s["t"] ./ s["T"], s["speed"] ./ maximum(s["speed"]), c=cg[(i - 1)/(length(Ds) - 1)])
    end
    ss = range(0, 1, length=200); plot!(p2, ss, 16 .* ss.^2 .* (1 .- ss).^2, c=INK, ls=:dot, lw=1.5)
    plot(p1, p2, layout=grid(1, 2, widths=[0.6, 0.4]), size=(780, 330))
end

# ╔═╡ 05000009-5eac-4a11-9b0e-000500000009
let
    Ds = time5["Ds"]; sols = [time5["free_D$(D)"] for D in Ds]
    Ts = [s["T"] for s in sols]; vp = [maximum(s["speed"]) for s in sols]
    # power-law fit T = a D^b on log-log axes
    A = hcat(ones(length(Ds)), log.(Ds)); cT = A \ log.(Ts); cv = A \ log.(vp)
    p1 = scatter(100 .* Ds, Ts, c=PAL[1], ms=6, msw=0, label="optimal duration", xlabel="reach distance (cm)", ylabel="duration (s)",
                 title=@sprintf("duration ∝ D^%.2f", cT[2]), legend=:topleft, ylims=(0, 0.75))
    Dg = range(0.04, 0.36, length=100)
    plot!(p1, 100 .* Dg, exp(cT[1]) .* Dg.^cT[2], c=PAL[1], lw=1, label="power-law fit")
    p2 = scatter(100 .* Ds, vp, c=PAL[2], ms=6, msw=0, label="", xlabel="reach distance (cm)", ylabel="peak speed (m/s)",
                 title=@sprintf("peak speed ∝ D^%.2f", cv[2]), ylims=(0, 1.3))
    plot(p1, p2, layout=(1, 2), size=(760, 300))
end

# ╔═╡ 05000010-5eac-4a11-9b0e-000500000010
md"""
Nobody told the optimizer how long to take. Longer reaches take longer, and they are also faster — peak
speed grows with distance (the "main sequence" of reaching). Every one of them is the same smooth bell.
Wong et al. (2021) compared exactly this kind of prediction with the preferred durations measured by
Reppert et al. (2018) and found them roughly similar (their Fig. 10).

### Where does the scaling come from?

A back-of-the-envelope version for a point mass makes the exponent plain. For a reach of distance ``D``
in time ``T`` the force scales like ``mD/T^2``. Positive work scales like ``m D^2/T^2``. The force-rate-rate
integral ``\int[\ddot F]_+dt`` adds up the climbs of ``\dot F``, which scale like ``mD/T^3``. So

```math
J(T) \approx \frac{\alpha D}{T^3} + \frac{\beta D^2}{T^2} + kT .
```

Setting ``dJ/dT = 0``: when force rate dominates, ``3\alpha D/T^4 = k`` and ``T^* \propto D^{1/4}``; when work
dominates, ``2\beta D^2/T^3 = k`` and ``T^* \propto D^{2/3}``. The arm comes in close to the force-rate end —
and you can see why in the table: along the optimal durations the force-rate part barely grows with
distance (``\propto D^{1-3\cdot 0.28} \approx D^{0.16}``), while the work part grows several-fold.
"""

# ╔═╡ 05000011-5eac-4a11-9b0e-000500000011
let
    rows = ["| D (cm) | T* (s) | peak speed (m/s) | work part (J) | force-rate part (J) | E (J) | time part kT (J) | J (J) |",
            "|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for D in time5["Ds"]
        s = time5["free_D$(D)"]
        push!(rows, @sprintf("| %.0f | %.3f | %.2f | %.2f | %.2f | %.2f | %.2f | %.2f |", 100D, s["T"], maximum(s["speed"]),
                             s["E_work"], s["E_fr"], s["E"], s["E_time"], s["J"]))
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 05000012-5eac-4a11-9b0e-000500000012
md"""
## Consider time *or* consider energy — it is the same thing

Here is the punchline of the series, and it is a piece of optimization theory rather than physiology.

Fix the duration at ``T`` and minimize energy: call the result ``E^*(T)``. It falls as ``T`` grows — slower
is cheaper. Now free the duration and minimize ``E + kT``. The optimum is where

```math
\frac{dE^*}{dT}\Big|_{T^*} = -k ,
```

the point where the energy curve's slope equals the price of time. Read it the other way: **every**
fixed-duration problem we solved in Lectures 2–4 is *exactly* a free-duration problem with a price of time
``k = -dE^*/dT`` — the Lagrange multiplier on the constraint "take ``T`` seconds". Fixing time and pricing time
are two descriptions of the same optimum. So if things are energy and time, you can either consider time,
or consider energy.
"""

# ╔═╡ 05000013-5eac-4a11-9b0e-000500000013
let
    Ts = time5["Ts"]; Es = time5["Estar"]; s = time5["free_D0.3"]; k = 25.0
    Tg = range(0.33, 1.25, length=200)
    p = plot(Ts, Es, c=PAL[1], marker=:circle, ms=4, msw=0, label="E*(T): best energy at fixed T", xlabel="movement duration T (s)",
             ylabel="J per reach", title="30 cm reach: energy, time, and their sum", ylims=(0, 45), size=(720, 360))
    plot!(p, Ts, Es .+ k .* Ts, c=PAL[2], marker=:circle, ms=4, msw=0, label="E*(T) + kT  (k = 25 J/s)")
    plot!(p, Tg, k .* Tg, c=INK, ls=:dot, lw=1.5, label="kT")
    # tangent to E* at the free-time optimum: slope −k
    Ts_ = s["T"]; Eo = s["E"]
    plot!(p, Tg, Eo .- k .* (Tg .- Ts_), c=PAL[3], ls=:dash, lw=1.5, label="slope −k at T*")
    scatter!(p, [Ts_], [Eo], c=PAL[3], ms=7, msw=0, label="")
    scatter!(p, [Ts_], [s["J"]], c=PAL[2], ms=7, msw=0, label=@sprintf("free-time optimum: T* = %.3f s", Ts_))
    vline!(p, [Ts_], c=:gray80, lw=1, label="")
    p
end

# ╔═╡ 05000014-5eac-4a11-9b0e-000500000014
let
    Ts = time5["Ts"]; Es = time5["Estar"]
    # finite-difference slope of E*(T) between neighbouring fixed-time solves
    Tm = (Ts[1:end-1] .+ Ts[2:end]) ./ 2; slope = diff(Es) ./ diff(Ts)
    p1 = plot(Tm, -slope, c=PAL[1], marker=:circle, ms=4, msw=0, xlabel="T (s)", ylabel="−dE*/dT (J/s)",
              title="the price of time implied by fixing T", label="from fixed-T solves", yscale=:log10, legend=:topright)
    ks = time5["ks"]
    scatter!(p1, [time5["free_k$(k)"]["T"] for k in ks], ks, c=PAL[2], ms=6, msw=0, label="free-T solves at k")
    p1
end

# ╔═╡ 05000015-5eac-4a11-9b0e-000500000015
let Ts = time5["Ts"], Es = time5["Estar"]
    md"""
    The line comes from the *fixed*-duration solves (the slope of ``E^*``); the dots are *free*-duration
    solves, each run with a given price of time ``k`` and placed at the duration it chose. They land on the
    line: the free-time optimum at price ``k`` is the fixed-time optimum at the duration whose marginal energy
    is ``k``. The 0.65 s used in Lecture 2 corresponds to ``k`` ≈ $(round(Int, -(Es[8]-Es[6])/(Ts[8]-Ts[6]))) J/s.
    """
end

# ╔═╡ 05000016-5eac-4a11-9b0e-000500000016
md"""
## Free time in the Kistemaker field

One more prediction for free. With time priced instead of fixed, how long should a reach take in the
Kistemaker field compared with the null field (both with the robot's 1 kg)?
"""

# ╔═╡ 05000017-5eac-4a11-9b0e-000500000017
let a = time5["kist_null"], b = time5["kist_field"]
    dev(s) = 100*s["hand"][1, argmax(abs.(s["hand"][1, :]))]
    md"""
    | | chosen duration | peak speed | max sideways | energy E | J = E + kT |
    |:--|--:|--:|--:|--:|--:|
    | null field | $(round(a["T"], digits=3)) s | $(round(maximum(a["speed"]), digits=2)) m/s | $(round(dev(a), digits=1)) cm | $(round(a["E"], digits=1)) J | $(round(a["J"], digits=1)) J |
    | Kistemaker field | $(round(b["T"], digits=3)) s | $(round(maximum(b["speed"]), digits=2)) m/s | $(round(dev(b), digits=1)) cm | $(round(b["E"], digits=1)) J | $(round(b["J"], digits=1)) J |

    With time priced rather than fixed, the model says: in this resistive field, **slow down a lot** — about
    $(round(Int, 100*(b["T"]/a["T"] - 1))) % longer than in the null field — and, now that the reach is slow, accept a
    small rightward bow of $(round(dev(b), digits=1)) cm (slow reaches make the force-rate bill smaller, so a little of the
    work-saving curve comes back; compare Lecture 4, exercise 2). In the experiment the timing was held to a
    300–500 ms window, so this is a prediction, not a comparison — one you could test by removing the window.
    """
end

# ╔═╡ 05000018-5eac-4a11-9b0e-000500000018
arm_animation([time5["free_D$(D)"] for D in (0.10, 0.20, 0.30)];
              labels=["10 cm", "20 cm", "30 cm"], title="Free-time reaches: the optimizer picks the duration",
              targets=[[0.0, 0.2], [0.0, 0.3], [0.0, 0.4], [0.0, 0.5]])

# ╔═╡ 05000019-5eac-4a11-9b0e-000500000019
md"""
## Your turn

Pick a distance and a price of time; the optimizer picks the duration.

distance $(@bind D5 Slider(0.05:0.05:0.35, default=0.30, show_value=true)) m ·
direction $(@bind dir5 Select([0.0 => "straight out", 45.0 => "45° right", -45.0 => "45° left", 90.0 => "sideways right"], default=0.0)) ·
price of time ``k`` $(@bind k5 Select([5.0 => "5 J/s (patient)", 10.0 => "10", 25.0 => "25 (Wong et al.)", 50.0 => "50", 100.0 => "100 (in a hurry)"], default=25.0))

$(@bind go5 CounterButton("Solve"))
"""

# ╔═╡ 05000020-5eac-4a11-9b0e-000500000020
user5 = go5 == 0 ? time5["free_D0.3"] :
    let θ = deg2rad(dir5)
        solve_reach(ReachTask(start=[0.0, 0.20], target=[0.0, 0.20] .+ D5 .* [sin(θ), cos(θ)], T=nothing),
                    CostWeights(k=k5); guess=time5["free_D0.3"])
    end;

# ╔═╡ 05000021-5eac-4a11-9b0e-000500000021
let s = user5
    p1 = plot(s["t"], s["speed"], c=PAL[1], legend=false, xlabel="time (s)", ylabel="hand speed (m/s)",
              title=@sprintf("%s · T* = %.3f s · E = %.1f J, kT = %.1f J", s["status"], s["T"], s["E"], s["E_time"]))
    p2 = plot(s["hand"][1, :], s["hand"][2, :], c=PAL[1], aspect_ratio=:equal, legend=false, xlabel="x (m)", ylabel="y (m)", title="hand path")
    scatter!(p2, [s["task"].start[1], s["task"].target[1]], [s["task"].start[2], s["task"].target[2]], c=INK, ms=4)
    plot(p1, p2, layout=grid(1, 2, widths=[0.62, 0.38]), size=(760, 320))
end

# ╔═╡ 05000022-5eac-4a11-9b0e-000500000022
arm_animation([user5]; labels=["your solve"], title="your solve")

# ╔═╡ 05000023-5eac-4a11-9b0e-000500000023
md"""
### Things to try

1. **Change the price of time** at a fixed distance. How does ``T^*`` scale with ``k``? (The point-mass
   argument predicts ``T^* \propto k^{-1/4}`` when force rate dominates and ``k^{-1/3}`` when work does.)
2. **Reach in different directions.** The arm's inertia is direction-dependent. Does the chosen duration
   depend on direction for the same distance? (People's do, a little.)
3. **Free time in a curl field.** Edit the `solve_reach` call to add `field=:curl_cw`. In Lecture 3 the
   fixed-time optimum rode the field at high speed. What does a price of time do to that strategy?

## What we have built

1. **Optimize at all** — an optimal control problem is five lines of intent: time, functions of time,
   dynamics as constraints, boundary conditions, an objective. Work predicts a highway; jerk predicts a reach.
2. **Turning muscle on costs energy** — calcium cycling makes changing force expensive; a cost in joules per
   force rate is a cost in watts per force-rate-rate, and on an arm it produces the bell.
3. **Curl fields** — ``J^{\mathsf T}F`` puts the robot into the joints; the field does no work, yet energy cares,
   and taken literally it predicts riding the field. A place where the theory is pushed hard.
4. **The Kistemaker field** — the field designed to make a curved path cheaper in *work* leaves the straight
   path optimal in *energy* once force rate is counted. The straight reaches people made were the optimum.
5. **Energy and time** — with a price on time, duration, speed and shape all fall out, and fixing time
   is the same as pricing it.

### References

* Reppert TR, Rigas I, Herzfeld DJ, Sedaghat-Nejad E, Komogortsev O, Shadmehr R (2018) Movement vigor as a traitlike attribute of individuality. *J Neurophysiol* 120:741–757.
* Shadmehr R, Huang HJ, Ahmed AA (2016) A representation of effort in decision-making and motor control. *Curr Biol* 26:1929–1934.
* Wong JD, Cluff T, Kuo AD (2021) The energetic basis for smooth human arm movements. *eLife* 10:e68013.
"""

# ╔═╡ Cell order:
# ╟─05000001-5eac-4a11-9b0e-000500000001
# ╟─05000002-5eac-4a11-9b0e-000500000002
# ╟─05000003-5eac-4a11-9b0e-000500000003
# ╟─05000004-5eac-4a11-9b0e-000500000004
# ╟─05000005-5eac-4a11-9b0e-000500000005
# ╠═05000006-5eac-4a11-9b0e-000500000006
# ╟─05000007-5eac-4a11-9b0e-000500000007
# ╟─05000008-5eac-4a11-9b0e-000500000008
# ╟─05000009-5eac-4a11-9b0e-000500000009
# ╟─05000010-5eac-4a11-9b0e-000500000010
# ╟─05000011-5eac-4a11-9b0e-000500000011
# ╟─05000012-5eac-4a11-9b0e-000500000012
# ╟─05000013-5eac-4a11-9b0e-000500000013
# ╟─05000014-5eac-4a11-9b0e-000500000014
# ╟─05000015-5eac-4a11-9b0e-000500000015
# ╟─05000016-5eac-4a11-9b0e-000500000016
# ╟─05000017-5eac-4a11-9b0e-000500000017
# ╟─05000018-5eac-4a11-9b0e-000500000018
# ╟─05000019-5eac-4a11-9b0e-000500000019
# ╟─05000020-5eac-4a11-9b0e-000500000020
# ╟─05000021-5eac-4a11-9b0e-000500000021
# ╟─05000022-5eac-4a11-9b0e-000500000022
# ╟─05000023-5eac-4a11-9b0e-000500000023
