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

# ╔═╡ 02000001-5eac-4a11-9b0e-000200000001
md"""
# Lecture 2 — Turning muscle on costs energy

*Energy and time is all you need · part 2 of 5*

Lecture 1 left us with a puzzle. Minimum **jerk** predicts reaches beautifully, and minimum
**work** predicts a highway cruise. Minimum jerk is a kinematic objective: it says *what* the hand
does, and gives the arm no physiological reason to care.

Here is the idea of this lecture. **Suppose smoothness is about energy after all.** What would
muscle have to be like for a jerk-like objective to be an *energetic* one? It would have to be
expensive to **change** force quickly — expensive to switch a muscle on, and to switch it back off.

That turns out to be physiologically true, and when we put it in an arm model the bell comes out.
"""

# ╔═╡ 02000002-5eac-4a11-9b0e-000200000002
begin
    import Pkg
    Pkg.activate(joinpath(@__DIR__, "..", ".."))
    using PlutoUI, Plots, Printf
    include(joinpath(@__DIR__, "ReachingLectures.jl"))
    using .ReachingLectures
    default(fontfamily="Helvetica", linewidth=2, framestyle=:axes, grid=false, size=(720, 320),
            legend=:topright, foreground_color_legend=nothing, palette=PAL,
            titlefontsize=11, guidefontsize=10, tickfontsize=8, legendfontsize=8,
            bottom_margin=4Plots.mm, left_margin=3Plots.mm)
    TableOfContents()
end

# ╔═╡ 02000003-5eac-4a11-9b0e-000200000003
md"""
## Why would changing force cost energy? Calcium.

A muscle fibre is switched on by releasing calcium from the sarcoplasmic reticulum (SR) and
switched off by **pumping it back** (SERCA pumps, one ATP per two Ca²⁺). Crossbridges also need
ATP, but the pumping is a separate bill — and it is paid per *activation and relaxation*, not per
unit of work. A few lines of evidence that this bill is big:

* **The pumps are a large slice of the budget.** In skinned fibres, SR Ca²⁺ pumping takes a constant
  **25–40 %** of the ATP used in isometric contraction, in slow and fast muscle alike
  (Rome & Klimov 2000); in human fibres the SR share is considerably larger in fast than in slow
  fibres (Szentesi et al. 2001). The "activation" heat that goes with it is a classic component of
  muscle energetics (Homsher & Kean 1978; Barclay 2015).
* **Short contractions cost more than long ones.** Stimulate human quadriceps with the same total
  contraction time and the same number of pulses, but chopped into 0.8 s instead of 3.2 s contractions,
  and the ATP cost goes **up**; Bergström & Hultman (1988) estimate that activation and relaxation make up
  **≈ 37 %** of the ATP used in a 1 s contraction. Hogan, Ingham & Kurdak (1998) found the same direction
  in situ: at matched force-time, shorter, more frequent contractions cost more metabolic energy and
  fatigue faster.
* **Fast muscles are built around it.** The superfast toadfish swimbladder muscle can switch on and
  off at 200 Hz only by investing in fast calcium handling and paying for it in force (Rome et al. 1999;
  Rome & Lindstedt 1998).
* **In humans, the rate of force change has a measurable cost.** Swinging the leg at fixed amplitude but
  higher frequency costs disproportionately more, beyond the mechanical work (Doke, Donelan & Kuo 2005;
  Doke & Kuo 2007). Isolated cyclic force production costs more with faster force development — at high
  frequencies the force-rate cost was comparable to the cost of work (van der Zee & Kuo 2021). And in
  cyclic reaching at a **fixed** mechanical power, metabolic cost rose with force rate as
  ``f^{5/2}`` (Wong, Cluff & Kuo 2021).

So: changing force is expensive, in addition to doing work.
"""

# ╔═╡ 02000004-5eac-4a11-9b0e-000200000004
md"""
## From "joules per force-rate" to "watts in force-rate-rate"

Wong et al. (2021) wrote the energy of one movement as

```math
E = c_W\,W^{+} + c_f\,\dot F ,
```

mechanical (positive) work ``W^+`` at a fixed cost per joule, plus a cost proportional to the
**amplitude of the force rate** ``\dot F`` — so many joules per (N/s) of force rate.

Now ask what that means *moment to moment*. Every time the force rate climbs, from zero up to some
peak, you pay in proportion to the climb. Add up all the climbs and you have the positive part of
the **force-rate-rate**:

```math
\sum \text{(climbs of } \dot F) = \int_0^T \big[\ddot F(t)\big]_+ \, dt ,
\qquad [z]_+ = \max(z, 0) .
```

A cost in **J per unit force rate** is a cost **rate** — **watts** — in ``[\ddot F]_+``. That gives the
objective we use for the rest of this series, summed over the joints of the arm (Wong et al. 2021,
Eqs. 1 and 13):

```math
\boxed{\;E = \sum_{i} \int_0^T \Big( c_W\,[\dot\theta_i \tau_i]_+ \;+\; c_t\,[\ddot\tau_i]_+ \Big)\, dt\;}
```

with ``c_W = 4.2`` (muscle is ≈ 25 % efficient at positive work; negative work is folded in) and
``c_t = 8.5\times10^{-2}`` fitted to the cyclic-reaching experiment — **not** fitted to the reaches
we are about to predict. Joint torque stands in for muscle force.
"""

# ╔═╡ 02000005-5eac-4a11-9b0e-000200000005
md"""
## The arm

A planar two-link arm in the horizontal plane (supported, so no gravity): shoulder at the origin,
upper arm and forearm+hand, joint torques ``\tau_1, \tau_2``. Its equations of motion are

```math
M(\theta)\,\ddot\theta + h(\theta, \dot\theta) = \tau + J(\theta)^{\mathsf T} F_{\text{hand}} ,
```

with the mass matrix ``M``, the velocity-product (Coriolis/centripetal) torques ``h``, the hand
Jacobian ``J`` and — from Lecture 3 on — a force ``F_{\text{hand}}`` applied at the hand.
Segment parameters are the standard set from Li & Todorov (2004): upper arm 0.30 m and 1.4 kg,
forearm+hand 0.33 m and 1.0 kg.

To pay for ``[\ddot\tau]_+`` the optimizer has to *have* ``\ddot\tau``, so the torques are states and
their second derivative is the control:

```math
\theta \xrightarrow{\;d/dt\;} \dot\theta \xrightarrow{\;d/dt\;} \ddot\theta ,
\qquad \tau \xrightarrow{\;d/dt\;} \dot\tau \xrightarrow{\;d/dt\;} u = \ddot\tau .
```

The task: hand from (0, 0.20) m to (0, 0.50) m — 30 cm straight out from the shoulder — in 0.65 s
(the standard movement Wong et al. compared with Harris & Wolpert 1998), from rest to rest, with zero
torque and torque rate at both ends.

Here is the problem as it is actually written in `ReachingLectures.jl` (this is the source, read
from the file — not a copy). It is the Lecture 1 recipe again: parameter, variables, dynamics,
boundary conditions, objective. Time is normalized, ``t = T s``, so that ``T`` can become a decision
variable in Lecture 5.
"""

# ╔═╡ 02000006-5eac-4a11-9b0e-000200000006
source_excerpt("# ---------------- states and control", "# ---------------- initial guess")

# ╔═╡ 02000007-5eac-4a11-9b0e-000200000007
md"""
## Four costs on the same arm

Same arm, same task. We compare

1. **work only** — ``c_W\int[\dot\theta\tau]_+``, the Lecture 1 objective on an arm;
2. **force-rate-rate only** — ``c_t\int[\ddot\tau]_+``;
3. **work + force-rate-rate** — the full energy ``E`` above (Wong et al. 2021);
4. **quadratic force-rate-rate** — ``\int\ddot\tau^2``, a smoothness cost with no energetic
   meaning, for contrast.

(Each is a few seconds of Ipopt; cached.)
"""

# ╔═╡ 02000008-5eac-4a11-9b0e-000200000008
arm2 = cached("L2_arm_costs") do
    task = ReachTask(start=[0.0, 0.20], target=[0.0, 0.50], T=0.65)
    Dict("work"     => solve_reach(task, CostWeights(c_t=0.0, eps=1e-6)),
         "frr"      => solve_reach(task, CostWeights(c_W=0.0)),
         "energy"   => solve_reach(task, CostWeights()),
         "frr_quad" => solve_reach(task, CostWeights(c_W=0.0, c_t=0.0, c_frr2=1e-4)))
end;

# ╔═╡ 02000009-5eac-4a11-9b0e-000200000009
begin
    keys2 = ["work", "frr", "energy", "frr_quad"]
    labels2 = Dict("work" => "work only", "frr" => "force-rate-rate only",
                   "energy" => "work + force-rate-rate (energy)", "frr_quad" => "quadratic ∫τ̈²")
    md""
end

# ╔═╡ 02000010-5eac-4a11-9b0e-000200000010
let
    p1 = plot(title="hand speed", xlabel="time (s)", ylabel="m/s", legend=:topright, ylims=(0, 1.25))
    for (i, k) in enumerate(keys2)
        s = arm2[k]; plot!(p1, s["t"], s["speed"], c=PAL[i], label=labels2[k])
    end
    tt = range(0, 0.65, length=200)
    plot!(p1, tt, [min_jerk_profile(τ, 0.30, 0.65).v for τ in tt], c=INK, ls=:dot, lw=1.5, label="min jerk (analytic)")
    p2 = plot(title="hand paths", xlabel="x (m)", ylabel="y (m)", aspect_ratio=:equal, legend=false,
              xlims=(-0.08, 0.08), ylims=(0.18, 0.52), xticks=[-0.05, 0, 0.05])
    for (i, k) in enumerate(keys2)
        s = arm2[k]; plot!(p2, s["hand"][1, :], s["hand"][2, :], c=PAL[i])
    end
    plot(p1, p2, layout=grid(1, 2, widths=[0.72, 0.28]), size=(780, 330))
end

# ╔═╡ 02000011-5eac-4a11-9b0e-000200000011
md"""
**Work only** (blue) gives the arm version of the Lecture 1 highway: a fast launch, a long
cruise, a hard stop. Add the cost of changing force and the plateau is gone: **force-rate-rate**
alone (orange) and the **full energy** (aqua) both give a single bell, and the full energy sits right on
the minimum-jerk curve. All of them move the hand along a nearly straight line, without being asked to.

The quadratic ``\int\ddot\tau^2`` (yellow) is *also* a bell, but a peakier one. That is not an accident,
and the next cell says why.
"""

# ╔═╡ 02000012-5eac-4a11-9b0e-000200000012
details("Of course it looks like jerk — the point-mass algebra", md"""
For a point mass ``F = m\ddot x``, so every derivative of force is a derivative of motion:

```math
\dot F = m\,\dddot x \;(\text{jerk}), \qquad \ddot F = m\,x^{(4)} \;(\text{snap}).
```

* Minimizing ``\int \dot F^2`` **is** minimizing jerk: the quintic of Lecture 1, peak speed ``1.875\,D/T``.
* Minimizing ``\int \ddot F^2`` is minimizing snap: Euler–Poisson gives ``x^{(8)} = 0``, a 7th-order polynomial,
  speed ``\propto s^3(1-s)^3``, peak ``2.19\,D/T`` — the yellow curve, a slightly peakier bell.
* Minimizing ``\int [\ddot F]_+`` is an ``L^1`` problem: the optimizer makes ``\ddot F`` either zero or as
  concentrated as possible, so ``\dot F`` is close to piecewise constant and the force is close to piecewise linear.
  Integrate a piecewise-linear force twice and you get a speed profile made of cubic pieces that lands
  very close to the minimum-jerk bell — and the arm's inertia (with the positive-work term) rounds it further.

So any cost on how fast force changes produces a smooth, single-peaked speed profile, and *which*
derivative you penalize sets how peaky it is. Minimum jerk is the kinematic shadow of an energetic cost
on changing force. The difference is that this cost has units of joules, a coefficient measured in a
different experiment, and a mechanism (calcium) behind it.
""")

# ╔═╡ 02000013-5eac-4a11-9b0e-000200000013
let
    fh = load_flash_hogan()
    p = plot(title="shape: model vs. Flash & Hogan (1985) measured reaches", xlabel="t / duration",
             ylabel="speed / peak", legend=:topright, ylims=(0, 1.3), size=(720, 330))
    for (i, mv) in enumerate(("A", "B"))
        m = fh[mv]["measured"]; t0, t1 = m.t[1], m.t[end]
        scatter!(p, (m.t .- t0) ./ (t1 - t0), m.v ./ maximum(m.v), c=INK, ms=3, msw=0,
                 marker=(i == 1 ? :circle : :utriangle), label="Flash & Hogan movement $mv")
    end
    for (i, k) in enumerate(keys2)
        s = arm2[k]; plot!(p, s["t"] ./ s["T"], s["speed"] ./ maximum(s["speed"]), c=PAL[i], label=labels2[k])
    end
    p
end

# ╔═╡ 02000014-5eac-4a11-9b0e-000200000014
md"""
## Watch it

Press play (0.25× is real time slowed four-fold). Same start, same target, same duration; only the cost differs.
"""

# ╔═╡ 02000015-5eac-4a11-9b0e-000200000015
arm_animation([arm2[k] for k in keys2]; labels=[labels2[k] for k in keys2],
              title="30 cm reach in 0.65 s under four costs")

# ╔═╡ 02000016-5eac-4a11-9b0e-000200000016
md"""
## Where the energy goes

The full-energy solution splits its bill between work and force rate. Note what the work-only
solution does: it is the cheapest in *work*, as it must be, but it would be expensive in force rate
(its torques change abruptly) — and the reverse for the force-rate-rate-only solution.
"""

# ╔═╡ 02000017-5eac-4a11-9b0e-000200000017
let
    rows = ["| cost optimized | positive work W⁺ (J) | ``c_W W^+`` (J) | ``\\int[\\ddot\\tau]_+`` (N·m/s) | ``c_t \\int[\\ddot\\tau]_+`` (J) | total ``E`` (J) | peak speed (m/s) |",
            "|:--|--:|--:|--:|--:|--:|--:|"]
    for k in keys2
        s = arm2[k]
        Wp = s["W_pos"]; Fp = s["FRR_pos"]
        E = 4.2*Wp + 8.5e-2*Fp       # evaluate every solution with the SAME full energy
        push!(rows, @sprintf("| %s | %.3f | %.2f | %.1f | %.2f | **%.2f** | %.2f |", labels2[k], Wp, 4.2Wp, Fp, 8.5e-2*Fp, E, maximum(s["speed"])))
    end
    Markdown.parse(join(rows, "\n") * "\n\n*Every row is scored with the same full energy ``E = 4.2\\,W^+ + 0.085\\int[\\ddot\\tau]_+``, whatever it was optimized for.*")
end

# ╔═╡ 02000018-5eac-4a11-9b0e-000200000018
let s = arm2["energy"]
    p1 = plot(s["t"], s["tau"]', label=["shoulder" "elbow"], title="torque (energy-optimal)", xlabel="time (s)", ylabel="N·m", c=[PAL[1] PAL[2]])
    p2 = plot(s["t"], s["taud"]', label=["shoulder" "elbow"], title="torque rate", xlabel="time (s)", ylabel="N·m/s", c=[PAL[1] PAL[2]], legend=false)
    p3 = plot(s["t"], s["power"]', label=["shoulder" "elbow"], title="joint power", xlabel="time (s)", ylabel="W", c=[PAL[1] PAL[2]], legend=false)
    plot(p1, p2, p3, layout=(1, 3), size=(780, 260))
end

# ╔═╡ 02000019-5eac-4a11-9b0e-000200000019
md"""
Look at the torque rate (middle): it is **piecewise constant**, so the torque (left) is
**piecewise linear**. That is the ``L^1`` cost at work — the optimizer changes ``\dot\tau`` in a few
decisive jumps (``\ddot\tau`` is a handful of impulses) instead of continuously, because with
``[\ddot\tau]_+`` it pays for the total *climb* in torque rate, not for how it is spread out. The torque
has kinks, but the arm integrates it twice through its inertia, so the motion (and the hand speed) is smooth.

## Your turn

Scale the two energy coefficients, change the duration or the target, and press **Solve**
(a few seconds; the first solve in a session takes longer while Julia compiles). Until you
press it you see the cached full-energy answer.

work coefficient ``c_W`` $(@bind cW_user Select([0.0 => "0 (no work cost)", 1.0 => "1", 4.2 => "4.2 (muscle)", 10.0 => "10"], default=4.2)) ·
force-rate coefficient ``c_t`` × $(@bind ct_mult Select([0.0 => "0 (none)", 0.1 => "0.1", 1.0 => "1 (Wong et al.)", 10.0 => "10"], default=1.0))

duration ``T`` $(@bind T2_user Slider(0.3:0.05:1.2, default=0.65, show_value=true)) s ·
target x $(@bind tx2 Slider(-0.20:0.05:0.20, default=0.0, show_value=true)) m ·
target y $(@bind ty2 Slider(0.35:0.05:0.55, default=0.50, show_value=true)) m

$(@bind go2 CounterButton("Solve"))
"""

# ╔═╡ 02000020-5eac-4a11-9b0e-000200000020
user2 = go2 == 0 ? arm2["energy"] :
    solve_reach(ReachTask(start=[0.0, 0.20], target=[tx2, ty2], T=T2_user),
                CostWeights(c_W=cW_user, c_t=8.5e-2*ct_mult, eps=(cW_user > 0 && ct_mult == 0 ? 1e-6 : 1e-7)));

# ╔═╡ 02000021-5eac-4a11-9b0e-000200000021
let s = user2
    D = sqrt(sum(abs2, s["task"].target .- s["task"].start))
    p1 = plot(s["t"], s["speed"], c=PAL[3], label="optimal", xlabel="time (s)", ylabel="hand speed (m/s)",
              title=@sprintf("%s · E = %.2f J (work %.2f, force rate %.2f)", s["status"], s["E"], s["E_work"], s["E_fr"]))
    tt = range(0, s["T"], length=200)
    plot!(p1, tt, [min_jerk_profile(τ, D, s["T"]).v for τ in tt], c=INK, ls=:dot, lw=1.5, label="min jerk")
    p2 = plot(s["hand"][1, :], s["hand"][2, :], c=PAL[3], aspect_ratio=:equal, legend=false, title="hand path",
              xlabel="x (m)", ylabel="y (m)")
    scatter!(p2, [s["task"].start[1], s["task"].target[1]], [s["task"].start[2], s["task"].target[2]], c=INK, ms=4)
    plot(p1, p2, layout=grid(1, 2, widths=[0.68, 0.32]), size=(780, 320))
end

# ╔═╡ 02000022-5eac-4a11-9b0e-000200000022
arm_animation([user2]; labels=["your solve"], title="your solve")

# ╔═╡ 02000023-5eac-4a11-9b0e-000200000023
md"""
### Things to try

1. **Turn the force-rate cost off** (``c_t`` × 0) with the work cost on. You get the highway back.
   Now try ``c_t`` × 0.1: how much force-rate cost does it take before the plateau is gone?
2. **Turn the work cost off** and keep force rate. Does the bell survive? Is the path still straight?
3. **Move the target sideways** (x = ±0.2). Minimum jerk says the path is straight in *hand* space no matter
   what. Does the energetic optimum agree? Where does it bend, and why might the arm's mass matrix care?
4. **Slow down** (``T`` = 1.2 s). Which part of the energy shrinks faster, work or force rate? Predict
   first with the point mass: torque scales like ``mD/T^2``, so positive work goes like ``m D^2/T^2``, while
   ``\int[\ddot F]_+`` adds up climbs in ``\dot F \sim mD/T^3``. Force rate should fall off faster — and with a
   *different* power of ``D``. Keep that in mind for Lecture 5.

### Where we go next

So far the world has been empty. In **Lecture 3** we push on the hand with a robot — a curl field — and map
the hand forces into joint torques with ``J^{\mathsf T} F``. A curl field never does any work. Does energy
care about it at all?

### References

* Barclay CJ (2015) Energetics of contraction. *Compr Physiol* 5:961–995.
* Bergström M, Hultman E (1988) Energy cost and fatigue during intermittent electrical stimulation of human skeletal muscle. *J Appl Physiol* 65:1500–1505.
* Doke J, Donelan JM, Kuo AD (2005) Mechanics and energetics of swinging the human leg. *J Exp Biol* 208:439–445.
* Doke J, Kuo AD (2007) Energetic cost of producing cyclic muscle force, rather than work, to swing the human leg. *J Exp Biol* 210:2390–2398.
* Flash T, Hogan N (1985) The coordination of arm movements. *J Neurosci* 5:1688–1703.
* Harris CM, Wolpert DM (1998) Signal-dependent noise determines motor planning. *Nature* 394:780–784.
* Hogan MC, Ingham E, Kurdak SS (1998) Contraction duration affects metabolic energy cost and fatigue in skeletal muscle. *Am J Physiol Endocrinol Metab* 274:E397–E402.
* Homsher E, Kean CJ (1978) Skeletal muscle energetics and metabolism. *Annu Rev Physiol* 40:93–131.
* Li W, Todorov E (2004) Iterative linear quadratic regulator design for nonlinear biological movement systems. *ICINCO*.
* Rome LC, Lindstedt SL (1998) The quest for speed: muscles built for high-frequency contractions. *News Physiol Sci* 13:261–268.
* Rome LC, Cook C, Syme DA, et al. (1999) Trading force for speed: why superfast crossbridge kinetics leads to superlow forces. *PNAS* 96:5826–5831.
* Rome LC, Klimov AA (2000) Superfast contractions without superfast energetics: ATP usage by SR-Ca²⁺ pumps and crossbridges in toadfish swimbladder muscle. *J Physiol* 526:279–286.
* Szentesi P, Zaremba R, van Mechelen W, Stienen GJM (2001) ATP utilization for calcium uptake and force production in different types of human skeletal muscle fibres. *J Physiol* 531:393–403.
* van der Zee TJ, Kuo AD (2021) The high energetic cost of rapid force development in muscle. *J Exp Biol* 224:jeb233965.
* Wong JD, Cluff T, Kuo AD (2021) The energetic basis for smooth human arm movements. *eLife* 10:e68013.
"""

# ╔═╡ Cell order:
# ╟─02000001-5eac-4a11-9b0e-000200000001
# ╟─02000002-5eac-4a11-9b0e-000200000002
# ╟─02000003-5eac-4a11-9b0e-000200000003
# ╟─02000004-5eac-4a11-9b0e-000200000004
# ╟─02000005-5eac-4a11-9b0e-000200000005
# ╟─02000006-5eac-4a11-9b0e-000200000006
# ╟─02000007-5eac-4a11-9b0e-000200000007
# ╠═02000008-5eac-4a11-9b0e-000200000008
# ╟─02000009-5eac-4a11-9b0e-000200000009
# ╟─02000010-5eac-4a11-9b0e-000200000010
# ╟─02000011-5eac-4a11-9b0e-000200000011
# ╟─02000012-5eac-4a11-9b0e-000200000012
# ╟─02000013-5eac-4a11-9b0e-000200000013
# ╟─02000014-5eac-4a11-9b0e-000200000014
# ╟─02000015-5eac-4a11-9b0e-000200000015
# ╟─02000016-5eac-4a11-9b0e-000200000016
# ╟─02000017-5eac-4a11-9b0e-000200000017
# ╟─02000018-5eac-4a11-9b0e-000200000018
# ╟─02000019-5eac-4a11-9b0e-000200000019
# ╟─02000020-5eac-4a11-9b0e-000200000020
# ╟─02000021-5eac-4a11-9b0e-000200000021
# ╟─02000022-5eac-4a11-9b0e-000200000022
# ╟─02000023-5eac-4a11-9b0e-000200000023
