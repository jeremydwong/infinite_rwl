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

# ╔═╡ 01000001-5eac-4a11-9b0e-000100000001
md"""
# Lecture 1 — Optimize at all

*Energy and time is all you need · part 1 of 5*

Reach for your coffee. Your hand speeds up, slows down, and arrives — one smooth hump of
speed, the same shape almost every time, for almost everyone. Where does that shape come from?

The bet of this whole series is that it comes from **optimization**: of all the ways to get the
hand from here to there, the nervous system picks a cheap one. To take that bet seriously we
first have to be able to *pose* an optimization and *solve* it. So this lecture is about two
things:

1. learning to read (and write) an optimal control problem in **InfiniteOpt**, and
2. noticing that the obvious objective — do as little **work** as possible — gives a movement
   that looks nothing like a reach, while a purely kinematic objective — **minimum jerk** —
   looks almost exactly like one.

That mismatch is the puzzle for the rest of the series.

!!! note "How to use these notebooks"
    Everything below is already solved and cached, so the figures and animations are live the
    moment the page opens. Each lecture ends with a **Your turn** section: move the sliders,
    press **Solve**, and the optimizer runs for real on your machine.
"""

# ╔═╡ 01000002-5eac-4a11-9b0e-000100000002
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    using PlutoUI, Plots, InfiniteOpt, Ipopt, Printf
    include(joinpath(@__DIR__, "ReachingLectures.jl"))
    using .ReachingLectures
    default(fontfamily="Helvetica", linewidth=2, framestyle=:axes, grid=false, size=(720, 320),
            legend=:topright, foreground_color_legend=nothing, palette=PAL,
            titlefontsize=11, guidefontsize=10, tickfontsize=8, legendfontsize=8,
            bottom_margin=4Plots.mm, left_margin=3Plots.mm)
    TableOfContents()
end

# ╔═╡ 01000003-5eac-4a11-9b0e-000100000003
md"""
## The problem

Keep it as simple as possible: a **point mass** ``m`` on a line (a hand with no arm attached).
It starts at rest at ``x = 0`` and must be at rest at ``x = D`` exactly ``T`` seconds later.
We push on it with a force ``F(t)``:

```math
m\,\ddot x = F, \qquad x(0) = 0,\; \dot x(0) = 0, \qquad x(T) = D,\; \dot x(T) = 0 .
```

There are infinitely many force histories that do this. An **objective** picks one. We will try three:

| objective | in words | formula |
|:--|:--|:--|
| minimum **work** | spend as little mechanical energy as you can | ``\int_0^T \lvert F\,\dot x\rvert \, dt`` |
| minimum **force²** | "effort" in the usual squared sense | ``\int_0^T F^2 \, dt`` |
| minimum **jerk** | be as smooth as you can (Flash & Hogan 1985) | ``\int_0^T \dddot x^{\,2} \, dt`` |
"""

# ╔═╡ 01000004-5eac-4a11-9b0e-000100000004
md"""
## Reading an InfiniteOpt problem

InfiniteOpt lets us write the problem almost exactly the way we would on a whiteboard, and then
it does the bookkeeping of turning it into a big finite optimization (here by **orthogonal
collocation**: each function of time becomes a piecewise polynomial on a grid of `n` points,
and the derivative relations hold exactly at the collocation points). Ipopt then solves that.

Read the function below top to bottom. It has five parts and every problem in this series has
the same five parts:

1. an **infinite parameter** — time, here `t ∈ [0, T]`;
2. **decision variables that are functions of time** — `x(t)`, `v(t)`, `F(t)`, …;
3. **dynamics as constraints** — `∂(x, t) == v` says "the derivative of x is v", it is *not* a
   simulation, just a relation the solver must satisfy;
4. **boundary conditions** — `x(0) == 0` evaluates the function at a point;
5. an **objective** — an `integral(…, t)` of something we care about.

Nothing is integrated forward in time. The optimizer adjusts the *entire* trajectory at once
until the physics, the boundary conditions and the optimality conditions all hold together.
"""

# ╔═╡ 01000005-5eac-4a11-9b0e-000100000005
"""
    point_mass(objective; D=0.30, T=0.65, m=1.0, F_max=6.0, n=101)

Move a point mass `m` (kg) a distance `D` (m) in `T` (s), rest to rest.
`objective` is `:work`, `:force2` or `:jerk`. Returns plain arrays (t, x, v, a, F).
"""
function point_mass(objective::Symbol; D=0.30, T=0.65, m=1.0, F_max=6.0, n=101)
    model = InfiniteModel(Ipopt.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "sb", "yes")             # (hides Ipopt's banner)

    # (1) the infinite parameter: time, discretized on n points by collocation
    @infinite_parameter(model, t in [0, T], num_supports=n,
                        derivative_method=OrthogonalCollocation(2))

    # (2) decision variables: functions of time
    @variable(model, x, Infinite(t))                        # position (m)
    @variable(model, v, Infinite(t))                        # velocity (m/s)
    @variable(model, a, Infinite(t))                        # acceleration (m/s²)
    @variable(model, -F_max <= F <= F_max, Infinite(t))     # force (N) — muscles are not infinitely strong

    # (3) dynamics as constraints
    @constraint(model, ∂(x, t) == v)
    @constraint(model, ∂(v, t) == a)
    @constraint(model, F == m*a)                            # Newton

    # (4) boundary conditions: rest at x = 0, rest at x = D
    @constraint(model, x(0) == 0); @constraint(model, v(0) == 0)
    @constraint(model, x(T) == D); @constraint(model, v(T) == 0)

    # (5) the objective
    if objective == :work
        # |F v| is not smooth at 0, so split the power into a positive and a negative part,
        # both ≥ 0, and pay for both: at the optimum one of them is always zero.
        @variable(model, Ppos >= 0, Infinite(t))
        @variable(model, Pneg >= 0, Infinite(t))
        @constraint(model, Ppos - Pneg == F*v)
        # (a whisper of ∫F² makes the answer unique; it changes nothing you can see)
        @objective(model, Min, integral(Ppos + Pneg, t) + 1e-6*integral(F^2, t))
    elseif objective == :force2
        @objective(model, Min, integral(F^2, t))
    elseif objective == :jerk
        # jerk is the derivative of acceleration, so acceleration needs boundary conditions too
        @variable(model, jerk, Infinite(t))
        @constraint(model, ∂(a, t) == jerk)
        @constraint(model, a(0) == 0); @constraint(model, a(T) == 0)
        @objective(model, Min, integral(jerk^2, t))
    end

    optimize!(model)
    ts = supports(t, label=All)                             # all grid + collocation points
    val(z) = value(z, label=All)
    (t=collect(ts), x=val(x), v=val(v), a=val(a), F=val(F),
     objective=objective, D=D, T=T, m=m, status=string(termination_status(model)))
end

# ╔═╡ 01000006-5eac-4a11-9b0e-000100000006
md"""
## Three objectives, three movements

Same mass (1 kg), same distance (30 cm), same time (0.65 s), same boundary conditions.
Only the objective changes. (These are cached; the solve itself takes well under a second.)
"""

# ╔═╡ 01000007-5eac-4a11-9b0e-000100000007
pm = cached("L1_point_mass") do
    Dict(o => point_mass(o) for o in (:work, :force2, :jerk))
end;

# ╔═╡ 01000008-5eac-4a11-9b0e-000100000008
let
    lab = Dict(:work => "minimum work", :force2 => "minimum ∫F²", :jerk => "minimum jerk")
    p1 = plot(title="speed", xlabel="time (s)", ylabel="m/s", legend=:topright)
    p2 = plot(title="force", xlabel="time (s)", ylabel="N", legend=false)
    for (i, o) in enumerate((:work, :force2, :jerk))
        s = pm[o]
        plot!(p1, s.t, s.v, label=lab[o], c=PAL[i])
        plot!(p2, s.t, s.F, c=PAL[i])
    end
    tt = range(0, 0.65, length=200)
    plot!(p1, tt, [min_jerk_profile(τ, 0.30, 0.65).v for τ in tt], c=INK, ls=:dot, lw=1.5, label="min jerk, analytic")
    hline!(p2, [0], c=:gray80, lw=1)
    plot(p1, p2, layout=(1, 2), size=(760, 300))
end

# ╔═╡ 01000009-5eac-4a11-9b0e-000100000009
md"""
Look at the **minimum-work** answer (blue). It slams on the maximum force, gets up to a speed,
**cruises**, and slams on the brakes. The speed profile is a step — a plateau with steep sides.
Make the force limit larger and the sides get steeper; in the limit it is a perfect rectangle
(an impulse, a coast, an impulse). That is the cheapest way, mechanically, to move a mass a fixed
distance in a fixed time: any extra peak speed costs extra kinetic energy that you put in and
then have to take out again.

The **minimum ∫F²** answer (orange) is a parabola of speed, with the force jumping at the
start and the end — it is the classic "effort" objective and it is not bell-shaped either.

**Minimum jerk** (aqua) gives the bell. It sits right on the textbook formula (dotted), which
we derive below.
"""

# ╔═╡ 01000010-5eac-4a11-9b0e-000100000010
point_animation([(t=pm[o].t, x=pm[o].x, v=pm[o].v) for o in (:work, :force2, :jerk)];
                labels=["minimum work", "minimum ∫F²", "minimum jerk"],
                title="Three ways to move 30 cm in 0.65 s (press play; slow it down to 0.1×)")

# ╔═╡ 01000011-5eac-4a11-9b0e-000100000011
md"""
## This is NOT how you ride a bike, or drive on the highway

The minimum-work answer is not crazy. It is exactly how we operate machines when energy
matters and the trip is long:

* **Highway driving**: accelerate onto the on-ramp, set the cruise control, hold a fixed speed,
  brake at the exit. The cruise phase *is* the plateau.
* **Cycling**: get up to speed and hold a cadence. Nobody rides with a bell-shaped speed profile.

A reach never does this. There is no cruise phase, ever — the hand is always either speeding up or
slowing down, and the transitions are smooth. The closest everyday analogue is maybe **driving in
the city**: stop-and-go, rolling smoothly into each stop (possibly for other reasons — comfort,
passengers, the car in front of you).

So: isn't that interesting? The "obvious" energy objective predicts a highway, and the arm does a
city. Either the arm does not care about energy, or the energy that matters is not mechanical work.
Hold that thought — it is the subject of Lecture 2.
"""

# ╔═╡ 01000012-5eac-4a11-9b0e-000100000012
md"""
## What real reaches look like: Flash & Hogan (1985)

Flash and Hogan recorded planar point-to-point reaches and showed that **minimum hand jerk**
predicts them remarkably well: straight paths and single-peaked, bell-shaped speed. Below are the
hand-speed traces from their Figure 3b — measured (their dashed lines) and their minimum-jerk
prediction (their solid lines) — digitized from the journal PDF for this lecture, for two movements.

The digitized numbers live in `data/flash_hogan_1985_fig3b.csv`; the file header documents
exactly how they were extracted (axis calibration, how dashed and solid lines were separated
where they overlap, and the ≈ ±10 ms / ±15 mm/s uncertainty).
"""

# ╔═╡ 01000013-5eac-4a11-9b0e-000100000013
let
    fh = load_flash_hogan()
    p1 = plot(title="Flash & Hogan 1985, Fig. 3b (digitized)", xlabel="time (s)", ylabel="hand speed (m/s)", ylims=(0, 1.35))
    p2 = plot(title="normalized to duration and peak", xlabel="t / duration", ylabel="speed / peak", legend=:topright, ylims=(0, 1.35))
    for (i, mv) in enumerate(("A", "B"))
        meas, pred = fh[mv]["measured"], fh[mv]["predicted"]
        scatter!(p1, meas.t, meas.v, c=INK, ms=3, msw=0, marker=(i == 1 ? :circle : :utriangle),
                 label="movement $mv measured")
        plot!(p1, pred.t, pred.v, c=PAL[i+2], lw=1.5, label="movement $mv predicted (their min jerk)")
        # normalize each MEASURED trace by its own start, end and peak
        t0, t1 = meas.t[1], meas.t[end]; vp = maximum(meas.v)
        scatter!(p2, (meas.t .- t0)./(t1 - t0), meas.v ./ vp, c=INK, ms=3, msw=0,
                 marker=(i == 1 ? :circle : :utriangle), label="movement $mv measured")
    end
    s = range(0, 1, length=200)
    plot!(p2, s, 16 .* s.^2 .* (1 .- s).^2, c=PAL[3], lw=2.5, label="min jerk: 16 s²(1−s)²")
    plot!(p2, s, ones(length(s)) .* (s .> 0.12) .* (s .< 0.88), c=PAL[1], lw=1.5, ls=:dash,
          alpha=0.6, label="what min work would give")
    plot(p1, p2, layout=(1, 2), size=(780, 320), legend=:topright)
end

# ╔═╡ 01000014-5eac-4a11-9b0e-000100000014
md"""
Two things to notice. Once you scale out the duration and the peak, both measured movements fall
on the **same** curve — the shape is stereotyped. And that curve is the minimum-jerk bell, nowhere
near the plateau that minimum work predicts. (Movement B is a little skewed: it peaks early and
decelerates longer, which is common in real reaches and which minimum jerk, being perfectly
symmetric, can't produce.)
"""

# ╔═╡ 01000015-5eac-4a11-9b0e-000100000015
details("Derivation: minimum jerk by the calculus of variations", md"""
We want the function ``x(t)`` that minimizes

```math
J[x] = \frac12 \int_0^T \dddot x(t)^2 \, dt
```

with ``x, \dot x, \ddot x`` fixed at both ends: ``x(0)=0,\ \dot x(0)=\ddot x(0)=0`` and ``x(T)=D,\ \dot x(T)=\ddot x(T)=0``.

**Perturb the answer.** Let ``x \to x + \varepsilon\,\eta(t)``, where ``\eta`` is any smooth
function that does not disturb the boundary conditions, so ``\eta, \dot\eta, \ddot\eta`` vanish at
``0`` and ``T``. At a minimum the first variation must vanish for every such ``\eta``:

```math
\frac{d}{d\varepsilon} J[x + \varepsilon \eta]\Big|_{\varepsilon=0} = \int_0^T \dddot x\, \dddot\eta \, dt = 0 .
```

**Integrate by parts three times** to move all the derivatives off ``\eta``. Each time, the boundary
term (``\dddot x\,\ddot\eta``, then ``x^{(4)}\dot\eta``, then ``x^{(5)}\eta``) vanishes because ``\eta`` and its first two
derivatives are zero at the ends, and each time we pick up a minus sign:

```math
\int_0^T \dddot x\, \dddot\eta \, dt = -\int_0^T x^{(6)}\, \eta \, dt = 0 \quad \text{for every admissible } \eta .
```

**Fundamental lemma** of the calculus of variations: if ``\int f\,\eta = 0`` for every such ``\eta``,
then ``f \equiv 0``. So the optimal trajectory satisfies the Euler–Poisson equation

```math
x^{(6)}(t) = 0 \quad\Longrightarrow\quad x(t) \text{ is a polynomial of degree 5.}
```

**Use the six boundary conditions** for the six coefficients. With ``s = t/T``, the three conditions
at ``s=0`` kill the constant, linear and quadratic terms, leaving ``x = D(a s^3 + b s^4 + c s^5)``;
the three conditions at ``s = 1`` give

```math
a + b + c = 1, \qquad 3a + 4b + 5c = 0, \qquad 6a + 12b + 20c = 0
\quad\Longrightarrow\quad a = 10,\ b = -15,\ c = 6 .
```

```math
\boxed{\,x(t) = D\left(10 s^3 - 15 s^4 + 6 s^5\right),\qquad \dot x(t) = \frac{30D}{T}\, s^2 (1-s)^2\,}
```

The speed is a symmetric bell that peaks at ``s = 1/2`` with ``\dot x_{max} = \tfrac{15}{8}\,D/T = 1.875\,D/T``.

**The same recipe for the other objectives.** Minimizing ``\int \ddot x^{\,2}\,dt`` (i.e. ``\int F^2`` for a point mass)
gives ``x^{(4)} = 0``: a cubic, a parabolic speed profile, and a force that jumps at both ends — the orange curve
above. In general minimizing ``\int (x^{(k)})^2`` gives ``x^{(2k)} = 0``, a polynomial of degree ``2k-1``.
Minimum **work** is different in kind: ``\int |F\dot x|`` is not quadratic and has no smooth stationary
point, so the answer lives on the force limits (bang–coast–bang). That is why the blue curve
is a step and gets steeper as you raise `F_max`.
""")

# ╔═╡ 01000016-5eac-4a11-9b0e-000100000016
md"""
## Your turn

Pick an objective and a task, then press **Solve**. (Until you press it, the plot shows the cached
answer for the default task.) The dotted line is always the analytic minimum-jerk speed for your
``D`` and ``T``.

objective $(@bind obj Select([:work => "minimum work", :force2 => "minimum ∫F²", :jerk => "minimum jerk"], default=:work))

distance ``D`` $(@bind D_user Slider(0.05:0.05:0.6, default=0.30, show_value=true)) m ·
time ``T`` $(@bind T_user Slider(0.2:0.05:1.5, default=0.65, show_value=true)) s ·
force limit $(@bind Fmax_user Slider(2.0:0.5:40.0, default=6.0, show_value=true)) N ·
grid points $(@bind n_user Select([21, 51, 101, 201], default=101))

$(@bind go_user CounterButton("Solve"))
"""

# ╔═╡ 01000017-5eac-4a11-9b0e-000100000017
user_pm = go_user == 0 ? pm[obj] :
    point_mass(obj; D=D_user, T=T_user, F_max=Fmax_user, n=n_user);

# ╔═╡ 01000018-5eac-4a11-9b0e-000100000018
let s = user_pm
    p1 = plot(s.t, s.v, c=PAL[1], label="optimal speed", xlabel="time (s)", ylabel="m/s",
              title=@sprintf("%s · D = %.2f m · T = %.2f s · %s", s.objective, s.D, s.T, s.status))
    tt = range(0, s.T, length=200)
    plot!(p1, tt, [min_jerk_profile(τ, s.D, s.T).v for τ in tt], c=INK, ls=:dot, lw=1.5, label="min jerk, analytic")
    p2 = plot(s.t, s.F, c=PAL[2], label="force", xlabel="time (s)", ylabel="N", legend=false, title="force")
    plot(p1, p2, layout=(1, 2), size=(760, 300))
end

# ╔═╡ 01000019-5eac-4a11-9b0e-000100000019
md"""
### Things to try

1. **Minimum work, raise the force limit** from 6 N toward 40 N. What happens to the ramps? What is
   the limit as ``F_{max} \to \infty``? Why can't the optimizer ever round the corners?
2. **Minimum work, drop the force limit** until the problem becomes infeasible. Can you predict the
   smallest force that can do the job from ``D`` and ``T`` alone? (Hint: bang–bang, no coast.)
3. **Minimum jerk, change the grid** from 101 to 21 points. Does the answer change? Minimum work at
   21 points — does it? Which problem is "easier" for collocation, and why?
4. **Minimum jerk, double ``D``.** The shape doesn't change; only the scale does. What does peak
   speed do? (Compare with ``1.875\,D/T``.)
5. **Edit the code.** Add a small amount of jerk cost to the work objective,
   `integral(Ppos + Pneg, t) + c*integral(jerk^2, t)` (you'll need the jerk lines from the `:jerk`
   branch). How big does `c` need to be before the plateau turns into a bell?

### Where we go next

Minimum jerk *describes* reaches beautifully, but it is a statement about kinematics — it gives
the arm no reason to care. In **Lecture 2** we ask what it would take for smoothness to be about
**energy** after all, and the answer is in the physiology of switching muscle on and off.
"""

# ╔═╡ Cell order:
# ╟─01000001-5eac-4a11-9b0e-000100000001
# ╟─01000002-5eac-4a11-9b0e-000100000002
# ╟─01000003-5eac-4a11-9b0e-000100000003
# ╟─01000004-5eac-4a11-9b0e-000100000004
# ╠═01000005-5eac-4a11-9b0e-000100000005
# ╟─01000006-5eac-4a11-9b0e-000100000006
# ╠═01000007-5eac-4a11-9b0e-000100000007
# ╟─01000008-5eac-4a11-9b0e-000100000008
# ╟─01000009-5eac-4a11-9b0e-000100000009
# ╟─01000010-5eac-4a11-9b0e-000100000010
# ╟─01000011-5eac-4a11-9b0e-000100000011
# ╟─01000012-5eac-4a11-9b0e-000100000012
# ╟─01000013-5eac-4a11-9b0e-000100000013
# ╟─01000014-5eac-4a11-9b0e-000100000014
# ╟─01000015-5eac-4a11-9b0e-000100000015
# ╟─01000016-5eac-4a11-9b0e-000100000016
# ╟─01000017-5eac-4a11-9b0e-000100000017
# ╟─01000018-5eac-4a11-9b0e-000100000018
# ╟─01000019-5eac-4a11-9b0e-000100000019
