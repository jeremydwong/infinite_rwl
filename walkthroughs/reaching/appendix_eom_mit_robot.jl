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

# ╔═╡ 09000001-5eac-4a11-9b0e-000900000001
md"""
# Appendix — The robot is part of the arm: equations of motion with the MIT-Manus / InMotion2

*Energy and time is all you need · appendix to Lectures 3 and 4*

In Lecture 3 the energy optimum in a curl field bowed ≈ 10 cm with nothing at the hand, and only
≈ 2–5 cm once we hung 1–2 kg on it. So the robot is not a bystander: whatever it adds at the hand
changes the prediction. That raises two questions this appendix answers with the real robot:

1. **What does the hand actually feel?** The force the experimenter *commands* is not the force at the
   handle; the robot's own links have mass, and moving them takes force.
2. **Does the controller take the robot's inertia into account?** It can — there is a published way to do it
   for exactly this robot — but only if the lab implemented it.

The robot model comes from R. Wilhelms-Tricarico (2004), *Methods for compensation of dynamic robot
properties* (project report on the InMotion2, the commercial MIT-Manus): its Lagrangian, its identified
parameters, and the compensation controller that was actually run on the machine.
"""

# ╔═╡ 09000002-5eac-4a11-9b0e-000900000002
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

# ╔═╡ 09000003-5eac-4a11-9b0e-000900000003
md"""
## The robot's equations of motion

The InMotion2 is a direct-drive parallelogram: two motors at the base, each driving one link through
an **absolute** angle (measured from the table's ``x`` axis), so the handle sits at

```math
x_r(q) = x_{\text{base}} + \ell_1\begin{bmatrix}\cos q_1\\ \sin q_1\end{bmatrix} + \ell_3\begin{bmatrix}\cos q_2\\ \sin q_2\end{bmatrix},
\qquad \ell_1 = 0.406\ \text{m},\ \ell_3 = 0.514\ \text{m}.
```

Its kinetic energy has just three parameters,

```math
L = \tfrac12 c_1\dot q_1^2 + c_2\cos(q_1 - q_2)\,\dot q_1\dot q_2 + \tfrac12 c_3\dot q_2^2 ,
```

and Euler–Lagrange gives (report Eqs. 1, 29–35)

```math
\underbrace{\begin{bmatrix} c_1 & c_2\cos(q_1-q_2)\\ c_2\cos(q_1-q_2) & c_3\end{bmatrix}}_{B(q)}\ddot q
+ \underbrace{c_2\sin(q_1-q_2)\begin{bmatrix}\dot q_2^2\\ -\dot q_1^2\end{bmatrix}}_{c(q,\dot q)}
= \tau_m - J_r^{\mathsf T} h ,
```

where ``\tau_m`` are the motor torques, ``J_r = \partial x_r/\partial q`` and ``h`` is the force the robot exerts on whatever
holds the handle — here, **the force on the hand**. The parameters were identified by regression from
138 s of hand-driven wiggling with a force sensor in the handle:
``c_1 = 0.3912``, ``c_2 = 0.2325``, ``c_3 = 0.2266`` kg m² (they include the ≈ 0.3 kg handle and agree with the
manufacturer's drawings to a few percent).
"""

# ╔═╡ 09000004-5eac-4a11-9b0e-000900000004
robot = inmotion2_params()      # c's from the regression, link lengths from the manufacturer; base: see below

# ╔═╡ 09000005-5eac-4a11-9b0e-000900000005
md"""
**Where is the robot?** The report doesn't say where the subject sits. We put the robot's base across the
table, 1.0 m in front of the shoulder (`robot.base`), with its elbow to the subject's right. That places our
30 cm reach near the middle of the robot's workspace, with the robot's elbow near 90° — where the report
measured its mass matrix. The placement is a guess, so we test others below.

## Writing it in terms of the human arm

The hand holds the handle: ``x_r(q_r) = x_h(\theta)``. Differentiate twice and the robot's joint motion follows from
the hand's:

```math
J_r\dot q_r = \dot x_h, \qquad J_r\ddot q_r + \dot J_r\dot q_r = \ddot x_h .
```

Put that into the robot's equation and solve for the force on the hand. Writing ``\sigma = J_r^{-\mathsf T}\tau_m`` for the
motor torques expressed as a force at the handle (report Eqs. 6–10):

```math
h = \sigma - M_r(x)\,\ddot x_h - G_r(x,\dot x)\,\dot x_h,\qquad
M_r = J_r^{-\mathsf T} B J_r^{-1},\quad G_r\dot x = J_r^{-\mathsf T}\big(c - B J_r^{-1}\dot J_r\dot q_r\big).
```

So the hand feels **whatever the motors command, minus the robot's inertia and Coriolis forces**. The arm's equations
of motion are then the familiar ones with this ``h``:

```math
M(\theta)\ddot\theta + h_{\text{arm}}(\theta,\dot\theta) = \tau + J_h^{\mathsf T}\, h .
```

Substituting ``\ddot x_h = J_h\ddot\theta + \dot J_h\dot\theta`` shows what this does to the *arm*: its mass matrix becomes
``M + J_h^{\mathsf T} M_r J_h``. The robot literally adds to the arm's inertia, by an amount that depends on where both are.

In the optimizer we don't invert anything: the robot's ``q_r, \dot q_r, \ddot q_r`` and the handle force ``h`` are extra
**algebraic** variables at every time node, tied to the hand by the three kinematic relations above and to each other
by the robot's own equation of motion. (They are not differential states, so the "high-index" trouble we hit with path
tracking in Lecture 3 doesn't arise.) Here is that code:
"""

# ╔═╡ 09000006-5eac-4a11-9b0e-000900000006
source_excerpt("function add_robot!", "\"Start values for the robot variables")

# ╔═╡ 09000007-5eac-4a11-9b0e-000900000007
md"""
## How big is the robot, seen from the hand?

The inertia felt at the hand is a 2×2 matrix that depends on position. Below, each ellipse has its axes along
that matrix's eigenvectors, with lengths proportional to the eigenvalues (kg): the effective mass you would feel
pushing the hand in that direction.
"""

# ╔═╡ 09000008-5eac-4a11-9b0e-000900000008
let p = arm_params()
    function ellipse!(plt, c, A, col; s=0.025, lab="")
        E = eigen(Symmetric(A)); θ = range(0, 2π, length=60)
        pts = [c .+ s .* (E.values[1]*E.vectors[:, 1]*cos(t) .+ E.values[2]*E.vectors[:, 2]*sin(t)) for t in θ]
        plot!(plt, first.(pts), last.(pts), c=col, lw=1.5, label=lab)
    end
    plt = plot(aspect_ratio=:equal, xlabel="x (m)", ylabel="y (m)", title="inertia felt at the hand (ellipse axes ∝ kg)",
               xlims=(-0.2, 0.2), ylims=(0.12, 0.6), legend=:outerright, size=(700, 420))
    first_ = true
    for x in -0.1:0.1:0.1, y in 0.2:0.1:0.5
        ellipse!(plt, [x, y], endpoint_inertia(p, [x, y]), PAL[1]; lab=(first_ ? "the arm itself" : ""))
        ellipse!(plt, [x, y], robot_endpoint_inertia(robot, [x, y]), PAL[2]; lab=(first_ ? "InMotion2, uncompensated" : ""))
        ellipse!(plt, [x, y], Matrix(1.0I, 2, 2), INK; lab=(first_ ? "compensated, D = 1 kg" : ""))
        first_ = false
    end
    plot!(plt, [0, 0], [0.2, 0.5], c=:gray80, lw=1, ls=:dash, label="the 30 cm reach")
    plt
end

# ╔═╡ 09000009-5eac-4a11-9b0e-000900000009
let p = arm_params(), x = [0.0, 0.35]
    A = endpoint_inertia(p, x); R = robot_endpoint_inertia(robot, x)
    # (math can't hold $(…) interpolation inside md, so the text is assembled as a string)
    mat(M) = @sprintf("\\begin{bmatrix}%.2f & %.2f\\\\ %.2f & %.2f\\end{bmatrix}", M[1, 1], M[1, 2], M[2, 1], M[2, 2])
    Markdown.parse("At the middle of the reach, (0, 0.35) m: the **arm** feels like ``" * mat(A) * "`` kg, and the " *
        "**uncompensated robot** adds ``" * mat(R) * "`` kg. The robot is *heavier than the arm* as seen from the hand, " *
        "and it is anisotropic. (The report's own estimate near this posture was ``\\begin{bmatrix}1.32 & 0.70\\\\ 0.74 & 1.90\\end{bmatrix}`` " *
        "in the robot's frame — the same size.)")
end

# ╔═╡ 09000010-5eac-4a11-9b0e-000900000010
md"""
## Does the commanded force take the robot into account?

It depends entirely on the controller the lab ran. Three possibilities, all from the report:

**(a) Command the field and nothing else** — ``\tau_m = J_r^{\mathsf T} F_{\text{field}}(\dot x)``. This is the simplest thing to program
(read the velocity, compute ``F = B\dot x``, map to torques). The hand then feels

```math
h = F_{\text{field}} - M_r(x)\ddot x - G_r\dot x ,
```

the field **plus the whole robot**: ≈ 1.4–1.8 kg of anisotropic, position-dependent inertia and the Coriolis forces.

**(b) Cancel the Coriolis term only** (report Eq. 13): ``\sigma = F_{\text{field}} + G_r\dot x``, giving ``h = F_{\text{field}} - M_r\ddot x``.

**(c) Cancel Coriolis and reshape the inertia** to an isotropic mass ``D`` (report Eq. 16):

```math
\sigma = M_r D^{-1}F_{\text{field}} + G_r\dot x - M_r D^{-1}h_{\text{meas}} + h_{\text{meas}} .
```

Put it into ``h = \sigma - M_r\ddot x - G_r\dot x`` and everything robot-specific cancels:

```math
D\,\ddot x = F_{\text{field}} - h \quad\Longleftrightarrow\quad h = F_{\text{field}} - D\,\ddot x .
```

The hand feels the field plus an **isotropic point mass D** — exactly the "1 kg at the handle" we used in Lectures 3–4.
Kistemaker, Wong & Gribble (2010) describe precisely this on their InMotion: *"commanded forces … were adjusted to
compensate for position dependence of the robot arm's inertia and to create an isotropic inertial characteristic with
a mass of 1 kg."*

**What does (c) need?** Notice it never uses acceleration. Instead it uses ``h_{\text{meas}}``, the handle force measured by the
**force/torque sensor**, and the filtered joint **velocities** (for ``G_r`` and ``\dot J_r\dot q_r``) — plus good values of ``c_1, c_2, c_3``
(the report's regression reproduced them within 1.5–2 % across sessions). So it does not need high-precision state; it
needs a force sensor and an identified model. And it has a floor: the report found the robot went unstable for
``D`` below about **0.7 kg**. You can reshape the robot's inertia; you can't make it vanish.

**Which one did a given experiment run?** You can't tell from the forces the paper reports — only from the controller
code (the report's is a `userfn.c` routine) or an explicit sentence in the methods. Shadmehr & Mussa-Ivaldi (1994)
used a different manipulandum entirely. The model below lets us ask how much the answer matters.
"""

# ╔═╡ 09000011-5eac-4a11-9b0e-000900000011
md"""
### Check: the compensation law really is an isotropic mass

We implement controller (c) *literally* in the optimizer — the motor torques computed from the measured handle force,
the velocities and ``c_1, c_2, c_3`` — and compare with the plain "isotropic 1 kg at the hand" model of Lecture 3. They
should be the same optimal control problem written two ways.
"""

# ╔═╡ 09000012-5eac-4a11-9b0e-000900000012
appx = cached("APP_mit_robot") do
    E = CostWeights(c_t=0.07)                       # the Lecture 3 energy
    T = ReachTask                                   # (shorthand)
    d = Dict{String,Any}()
    d["iso1_cw"] = solve_reach(T(field=:curl_cw, m_endpoint=1.0), E)   # Lecture 3's "1 kg at the hand"
    # the robot coupled at the handle, three controllers (and three D's for the compensated one)
    modes = [("none", :none, 1.0), ("coriolis", :coriolis, 1.0),
             ("full_D0.7", :full, 0.7), ("full_D1.0", :full, 1.0), ("full_D1.5", :full, 1.5)]
    for (name, comp, D) in modes
        rt(; kw...) = T(robot=:inmotion2, robot_comp=comp, robot_D=D; kw...)
        d["$(name)_null"] = solve_reach(rt(), E)
        for fld in (:curl_cw, :curl_ccw)
            d["$(name)_$(fld)"] = solve_reach(rt(field=fld), E; guess=d["$(name)_null"])
            d["$(name)_$(fld)_straight"] = solve_reach(rt(field=fld, track=bow(0.0)), E; guess=d["$(name)_null"])
        end
    end
    # no robot at all (Lecture 3's idealization)
    d["ideal_null"] = solve_reach(T(), E)
    for fld in (:curl_cw, :curl_ccw)
        d["ideal_$(fld)"] = solve_reach(T(field=fld), E)
        d["ideal_$(fld)_straight"] = solve_reach(T(field=fld, track=bow(0.0)), E; guess=d["ideal_$(fld)"])
    end
    # where the robot sits (uncompensated): other bases and the other elbow branch
    for (name, base, elbow) in (("base_far", [0.0, 1.1], 1.0), ("base_near", [0.0, 0.9], 1.0),
                                ("base_right", [0.25, 0.95], 1.0), ("elbow_left", [0.0, 1.0], -1.0))
        rp = RobotParams(base=base, elbow=elbow)
        rt(; kw...) = T(robot=:inmotion2, robot_comp=:none, robot_params=rp; kw...)
        g = solve_reach(rt(), E)
        d["$(name)_curl_cw"] = solve_reach(rt(field=:curl_cw), E; guess=g)
        d["$(name)_curl_cw_straight"] = solve_reach(rt(field=:curl_cw, track=bow(0.0)), E; guess=g)
    end
    # Lecture 4's Kistemaker field (400 ms) with the robot uncompensated vs compensated to 1 kg
    for (name, comp) in (("none", :none), ("full_D1.0", :full))
        rt(; kw...) = T(T=0.40, field=:kistemaker, robot=:inmotion2, robot_comp=comp, robot_D=1.0; kw...)
        o = solve_reach(rt(), E)
        d["kist_$(name)"] = o
        for A in (0.0, 0.05, -0.05)
            d["kist_$(name)_track_$(A)"] = solve_reach(rt(track=bow(A)), E; guess=o)
        end
    end
    d
end;

# ╔═╡ 09000013-5eac-4a11-9b0e-000900000013
let a = appx["iso1_cw"], b = appx["full_D1.0_curl_cw"]
    dv(s) = 100*s["hand"][1, argmax(abs.(s["hand"][1, :]))]
    md"""
    | clockwise curl, 0.65 s | energy E (J) | max sideways (cm) | peak speed (m/s) |
    |:--|--:|--:|--:|
    | isotropic 1 kg at the hand (Lecture 3) | $(round(a["E"], digits=4)) | $(round(dv(a), digits=2)) | $(round(maximum(a["speed"]), digits=3)) |
    | full InMotion2 + compensation law (c), D = 1 | $(round(b["E"], digits=4)) | $(round(dv(b), digits=2)) | $(round(maximum(b["speed"]), digits=3)) |

    Identical: the compensation law turns the real robot into the idealized one. (The report's own verification found
    an effective mass matrix within a few percent of ``D``.)
    """
end

# ╔═╡ 09000014-5eac-4a11-9b0e-000900000014
md"""
## What does it do to the prediction?

Same arm, same 30 cm reach in 0.65 s, same energy as Lecture 3 (``c_W`` = 4.2, ``c_t`` = 0.07), clockwise and
counter-clockwise curl fields with ``b`` = 13 N·s/m. For each way of running the robot: the free optimum, and the best
**straight** path (the hand kept within ±2 mm of the line, timing free).
"""

# ╔═╡ 09000015-5eac-4a11-9b0e-000900000015
begin
    modesA = [("ideal", "no robot (Lecture 3 idealization)"),
              ("full_D0.7", "compensated, D = 0.7 kg (the stability floor)"),
              ("full_D1.0", "compensated, D = 1 kg (Kistemaker et al.)"),
              ("full_D1.5", "compensated, D = 1.5 kg (report's field test)"),
              ("coriolis", "Coriolis cancelled only"),
              ("none", "uncompensated: field commanded, whole robot felt")]
    dvA(s) = 100*s["hand"][1, argmax(abs.(s["hand"][1, :]))]
    rowsA = ["| robot | null-field E (J) | CW: optimum bow | CW: straight costs | CCW: optimum bow | CCW: straight costs |",
             "|:--|--:|--:|--:|--:|--:|"]
    for (k, name) in modesA
        cw, cws = appx["$(k)_curl_cw"], appx["$(k)_curl_cw_straight"]
        cc, ccs = appx["$(k)_curl_ccw"], appx["$(k)_curl_ccw_straight"]
        push!(rowsA, @sprintf("| %s | %.1f | %+.1f cm | %+.1f J (%+.0f %%) | %+.1f cm | %+.1f J (%+.0f %%) |", name, appx["$(k)_null"]["E"],
              dvA(cw), cws["E"] - cw["E"], 100*(cws["E"]/cw["E"] - 1), dvA(cc), ccs["E"] - cc["E"], 100*(ccs["E"]/cc["E"] - 1)))
    end
    Markdown.parse(join(rowsA, "\n"))
end

# ╔═╡ 09000016-5eac-4a11-9b0e-000900000016
let
    p1 = plot(title="clockwise curl: energy-optimal hand paths", xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal,
              xlims=(-0.14, 0.06), ylims=(0.17, 0.53), legend=:outerright, size=(700, 400))
    for (i, (k, name)) in enumerate(modesA)
        s = appx["$(k)_curl_cw"]
        plot!(p1, s["hand"][1, :], s["hand"][2, :], c=PAL[i], label=split(name, " (")[1] * (occursin("D =", name) ? "" : ""))
    end
    scatter!(p1, [0, 0], [0.2, 0.5], c=INK, ms=4, label="")
    p1
end

# ╔═╡ 09000017-5eac-4a11-9b0e-000900000017
md"""
The robot shrinks the bow a lot, but it never makes the straight path the optimum:

* With **no robot**, the hand rides the field (≈ 9–10 cm).
* Any **real** robot puts more than a kilogram at the hand, and the bow drops to **2–5 cm** — whether the
  controller compensates (isotropic D) or not. Compensation changes *how* the robot feels (isotropic vs.
  anisotropic, with or without Coriolis), not that it is heavy.
* The **direction** is a prediction too: to the **left** in the clockwise field, and — once the robot is attached —
  to the **right** in the counter-clockwise field (the no-robot counter-clockwise optimum was the odd hook of Lecture 3).
* Going **straight still costs more** in every case, but by very different amounts: ≈ 3–6 J per reach (+30–50 %) in the
  clockwise field, only ≈ 1 J (+4–9 %) in the counter-clockwise field. The uncompensated robot makes straight relatively
  dearer in the clockwise field (its anisotropic inertia and Coriolis forces are themselves costly to fight), not in the
  counter-clockwise one.

So "a 2–5 cm bow" is about what energy predicts for a person holding a real manipulandum in a curl field, with a
clear direction for each field. Whether that is impressive is an empirical question: it is the size and direction of the
curvature to look for in adapted reaches. And in the counter-clockwise field, the energetic penalty for being straight is
small enough that you would not expect people to bother.
"""

# ╔═╡ 09000018-5eac-4a11-9b0e-000900000018
arm_animation([appx["ideal_curl_cw"], appx["none_curl_cw"], appx["full_D1.0_curl_cw"]];
              labels=["no robot", "uncompensated InMotion2", "compensated to 1 kg"],
              title="Clockwise curl field: the robot changes the optimum (grey: the robot's links)",
              xlim=(-0.35, 0.45), ylim=(-0.05, 1.05), show_force=true)

# ╔═╡ 09000019-5eac-4a11-9b0e-000900000019
md"""
## Does it matter where the robot sits?

Uncompensated, the robot's inertia depends on its configuration, so its placement relative to the subject matters.
We don't know the placement in any particular lab, so here are four guesses (clockwise field):
"""

# ╔═╡ 09000020-5eac-4a11-9b0e-000900000020
let
    rows = ["| robot placement (uncompensated) | optimum bow | straight costs |", "|:--|--:|--:|"]
    for (k, name) in (("none", "base 1.0 m ahead, elbow right (default)"), ("base_far", "base 1.1 m ahead"),
                      ("base_near", "base 0.9 m ahead"), ("base_right", "base 0.95 m ahead, 25 cm right"),
                      ("elbow_left", "base 1.0 m ahead, elbow left"))
        o, st = appx["$(k)_curl_cw"], appx["$(k)_curl_cw_straight"]
        push!(rows, @sprintf("| %s | %+.1f cm | %+.1f J (%+.0f %%) |", name, dvA(o), st["E"] - o["E"], 100*(st["E"]/o["E"] - 1)))
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 09000021-5eac-4a11-9b0e-000900000021
md"""
## And the Kistemaker field?

Lecture 4 used the compensated robot (1 kg, as Kistemaker et al. describe). If the robot had been uncompensated
instead, would the straight reach still be the energy optimum?
"""

# ╔═╡ 09000022-5eac-4a11-9b0e-000900000022
let
    rows = ["| Kistemaker field, 400 ms | optimum bow | E optimum (J) | straight | bowed 5 cm right | bowed 5 cm left |", "|:--|--:|--:|--:|--:|--:|"]
    for (k, name) in (("full_D1.0", "compensated to 1 kg"), ("none", "uncompensated"))
        o = appx["kist_$(k)"]; e(A) = appx["kist_$(k)_track_$(A)"]["E"]
        push!(rows, @sprintf("| %s | %+.1f cm | %.1f | %+.0f %% | %+.0f %% | %+.0f %% |", name, dvA(o), o["E"],
              100*(e(0.0)/o["E"] - 1), 100*(e(0.05)/o["E"] - 1), 100*(e(-0.05)/o["E"] - 1)))
    end
    Markdown.parse(join(rows, "\n"))
end

# ╔═╡ 09000023-5eac-4a11-9b0e-000900000023
md"""
Yes. Compensated or not, the energy optimum in the Kistemaker field is straight (under a centimetre), the straight
path is within about 2 % of it, and bowing 5 cm either way roughly doubles the energy. The robot makes every reach
more expensive but does not change Lecture 4's conclusion.
"""

# ╔═╡ 09000024-5eac-4a11-9b0e-000900000024
md"""
## Your turn

Pick the controller, the field and where the robot sits, then **Solve**.

controller $(@bind compU Select([:none => "uncompensated (field only)", :coriolis => "Coriolis cancelled", :full => "full compensation (Eq. 16)"], default=:none)) ·
D $(@bind DU Slider(0.7:0.1:3.0, default=1.0, show_value=true)) kg (full compensation only)

field $(@bind fldU Select([:curl_cw => "clockwise curl", :curl_ccw => "counter-clockwise curl", :none => "no field"], default=:curl_cw)) ·
robot base x $(@bind bxU Slider(-0.3:0.05:0.3, default=0.0, show_value=true)) m, y $(@bind byU Slider(0.85:0.05:1.2, default=1.0, show_value=true)) m ·
robot elbow $(@bind elU Select([1.0 => "right", -1.0 => "left"], default=1.0))

track straight? $(@bind trU CheckBox(default=false))

$(@bind goA CounterButton("Solve"))
"""

# ╔═╡ 09000025-5eac-4a11-9b0e-000900000025
userA = goA == 0 ? appx["none_curl_cw"] :
    solve_reach(ReachTask(field=fldU, robot=:inmotion2, robot_comp=compU, robot_D=DU,
                          robot_params=RobotParams(base=[bxU, byU], elbow=elU), track=(trU ? bow(0.0) : nothing)),
                CostWeights(c_t=0.07));

# ╔═╡ 09000026-5eac-4a11-9b0e-000900000026
let s = userA
    p1 = plot(s["hand"][1, :], s["hand"][2, :], c=PAL[2], aspect_ratio=:equal, legend=false, xlabel="x (m)", ylabel="y (m)",
              title=@sprintf("max sideways %+.1f cm", dvA(s)), xlims=(-0.15, 0.15), ylims=(0.17, 0.53))
    p2 = plot(s["t"], s["speed"], c=PAL[2], legend=false, xlabel="time (s)", ylabel="hand speed (m/s)",
              title=@sprintf("%s · E = %.2f J (work %.2f, force rate %.2f)", s["status"], s["E"], s["E_work"], s["E_fr"]))
    plot(p1, p2, layout=grid(1, 2, widths=[0.38, 0.62]), size=(760, 320))
end

# ╔═╡ 09000027-5eac-4a11-9b0e-000900000027
arm_animation([userA]; labels=["your solve"], title="your solve (grey: the robot)", xlim=(-0.4, 0.5), ylim=(-0.05, 1.1), show_force=true)

# ╔═╡ 09000028-5eac-4a11-9b0e-000900000028
md"""
### Things to try

1. **Full compensation, D from 0.7 to 3 kg.** Plot the optimum's bow against D. Where would a "massless" handle
   be on that curve — and why can't the real robot get there?
2. **Move the base.** With the robot uncompensated, does its placement change the *direction* of the bow, or only its size?
3. **No field.** How much does merely holding the uncompensated robot cost, compared with the compensated one? Does it
   bend the null-field path?

### References

* Wilhelms-Tricarico R (2004) Methods for compensation of dynamic robot properties. Project report (InMotion2).
* Kistemaker DA, Wong JD, Gribble PL (2010) The central nervous system does not minimize energy cost in arm movements. *J Neurophysiol* 104:2985–2994.
* Shadmehr R, Mussa-Ivaldi FA (1994) Adaptive representation of dynamics during learning of a motor task. *J Neurosci* 14:3208–3224.
"""

# ╔═╡ Cell order:
# ╟─09000001-5eac-4a11-9b0e-000900000001
# ╟─09000002-5eac-4a11-9b0e-000900000002
# ╟─09000003-5eac-4a11-9b0e-000900000003
# ╠═09000004-5eac-4a11-9b0e-000900000004
# ╟─09000005-5eac-4a11-9b0e-000900000005
# ╟─09000006-5eac-4a11-9b0e-000900000006
# ╟─09000007-5eac-4a11-9b0e-000900000007
# ╟─09000008-5eac-4a11-9b0e-000900000008
# ╟─09000009-5eac-4a11-9b0e-000900000009
# ╟─09000010-5eac-4a11-9b0e-000900000010
# ╟─09000011-5eac-4a11-9b0e-000900000011
# ╠═09000012-5eac-4a11-9b0e-000900000012
# ╟─09000013-5eac-4a11-9b0e-000900000013
# ╟─09000014-5eac-4a11-9b0e-000900000014
# ╟─09000015-5eac-4a11-9b0e-000900000015
# ╟─09000016-5eac-4a11-9b0e-000900000016
# ╟─09000017-5eac-4a11-9b0e-000900000017
# ╟─09000018-5eac-4a11-9b0e-000900000018
# ╟─09000019-5eac-4a11-9b0e-000900000019
# ╟─09000020-5eac-4a11-9b0e-000900000020
# ╟─09000021-5eac-4a11-9b0e-000900000021
# ╟─09000022-5eac-4a11-9b0e-000900000022
# ╟─09000023-5eac-4a11-9b0e-000900000023
# ╟─09000024-5eac-4a11-9b0e-000900000024
# ╟─09000025-5eac-4a11-9b0e-000900000025
# ╟─09000026-5eac-4a11-9b0e-000900000026
# ╟─09000027-5eac-4a11-9b0e-000900000027
# ╟─09000028-5eac-4a11-9b0e-000900000028
