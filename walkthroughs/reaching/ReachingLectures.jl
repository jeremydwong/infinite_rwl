# =============================================================================
# ReachingLectures.jl — shared model code for the five reaching lectures
#                       (walkthroughs/reaching/0*_*.jl, Pluto notebooks).
#
# What lives here (and what does NOT):
#   * Lecture 1 writes its point-mass InfiniteOpt problems INLINE in the notebook —
#     that lecture is about learning to read an optimization. Everything from
#     Lecture 2 on uses the planar two-link arm below, so it lives in one place.
#   * The arm model, the energy cost, endpoint force fields, hand-path tracking,
#     free movement time, the cache helpers and the browser animation.
#
# COORDINATES (SI units throughout — the human is the reference):
#   shoulder at the origin, x lateral (to the subject's RIGHT), y forward,
#   arm moves in the horizontal plane (supported on an air sled / manipulandum,
#   so no gravity). q1 = shoulder angle from +x, q2 = elbow angle (relative,
#   q2 > 0 = elbow bent with the elbow out to the right: a right arm).
#
# THE OPTIMAL CONTROL PROBLEM (one function, `solve_reach`, many switches):
#   states    q, q̇ (joint angles, rates), τ, τ̇ (joint torques and their rates)
#   control   u = τ̈ (torque "rate-rate"), so torque is a smooth C¹ signal
#   dynamics  M(q) q̈ + h(q, q̇) = τ + Jᵀ F_hand
#   cost      E = c_W Σ∫[q̇ᵢτᵢ]₊ dt  +  c_t Σ∫[τ̈ᵢ]₊ dt   (+ k·T if T is free)
#             — Wong, Cluff & Kuo (2021) eLife, Eqs. 1, 12, 13.
#   time      normalized: s ∈ [0, 1], t = T·s, so d/dt = (1/T) d/ds and T can be
#             a decision variable (Lecture 5).
#
# [x]₊ = max(x, 0) is not smooth, so we use the standard slack trick:
#   introduce p ≥ 0 with p ≥ x and minimize p — at the optimum p = max(x, 0).
# =============================================================================
module ReachingLectures

using LinearAlgebra, Printf, Serialization, Markdown
using InfiniteOpt, JuMP, Ipopt

export ArmParams, arm_params, hand_position, elbow_position, hand_jacobian, hand_velocity,
       mass_matrix, coriolis, ik, inverse_dynamics,
       min_jerk, min_jerk_profile, ReachTask, CostWeights, solve_reach, hand_path,
       curved_path, bow, fit_offset, field_force, RobotParams, inmotion2_params, robot_ik, robot_position,
       robot_elbow, robot_jacobian, robot_mass_matrix, robot_coriolis, robot_endpoint_inertia, robot_state, endpoint_inertia, cached, cache_path, interp1, load_flash_hogan,
       arm_animation, point_animation, energy_breakdown, perp_error, solve_reach_multistart, solve_ok, CACHE_DIR, DATA_DIR, PAL, INK, source_excerpt

const HERE = @__DIR__
const CACHE_DIR = joinpath(HERE, "cache")
const DATA_DIR = joinpath(HERE, "data")

# One fixed categorical order for every figure and animation in the series (validated
# colorblind-safe default palette); measured data are drawn in neutral ink, not a hue.
const PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
const INK = "#52514e"

"""
    source_excerpt(from, to)

The lines of THIS file from the line containing `from` up to (not including) the line
containing `to`, as a Julia code block — so the notebooks quote the code that actually
runs, not a copy that can drift.
"""
function source_excerpt(from::AbstractString, to::AbstractString)
    lines = readlines(@__FILE__)
    i = findfirst(l -> occursin(from, l), lines); j = findnext(l -> occursin(to, l), lines, i + 1)
    Markdown.parse("```julia\n" * join(lines[i:j-1], "\n") * "\n```")
end

# ============================================================================ arm
"""
Planar two-link arm. `I1`, `I2` are moments of inertia about the PROXIMAL joint
(shoulder for the upper arm, elbow for the forearm+hand), which keeps the mass
matrix in its compact textbook form. Default numbers: Li & Todorov (2004).
"""
Base.@kwdef struct ArmParams
    m1::Float64 = 1.4     # upper arm mass (kg)
    m2::Float64 = 1.0     # forearm + hand (kg)
    l1::Float64 = 0.30    # upper arm length (m)
    l2::Float64 = 0.33    # forearm + hand length, elbow → hand (m)
    s1::Float64 = 0.11    # shoulder → upper-arm COM (m)
    s2::Float64 = 0.16    # elbow → forearm COM (m)
    I1::Float64 = 0.025   # upper arm, about the shoulder (kg m²)
    I2::Float64 = 0.045   # forearm + hand, about the elbow (kg m²)
end
arm_params(; kw...) = ArmParams(; kw...)

# --- kinematics (generic: works on Float64 AND on InfiniteOpt variables) ---
elbow_position(p, q) = [p.l1*cos(q[1]), p.l1*sin(q[1])]
hand_position(p, q) = [p.l1*cos(q[1]) + p.l2*cos(q[1] + q[2]),
                       p.l1*sin(q[1]) + p.l2*sin(q[1] + q[2])]

"Hand Jacobian J(q): hand velocity = J(q) q̇. Its transpose maps hand forces to joint torques."
function hand_jacobian(p, q)
    s1, c1 = sin(q[1]), cos(q[1]); s12, c12 = sin(q[1] + q[2]), cos(q[1] + q[2])
    [-p.l1*s1 - p.l2*s12   -p.l2*s12;
      p.l1*c1 + p.l2*c12    p.l2*c12]
end
hand_velocity(p, q, qd) = hand_jacobian(p, q) * qd

"J̇(q, q̇)·q̇ — the velocity-product part of hand acceleration, a = J q̈ + J̇ q̇."
function jdot_qdot(p, q, qd)
    w1 = qd[1]; w12 = qd[1] + qd[2]
    [-p.l1*cos(q[1])*w1^2 - p.l2*cos(q[1] + q[2])*w12^2,
     -p.l1*sin(q[1])*w1^2 - p.l2*sin(q[1] + q[2])*w12^2]
end

# --- dynamics: M(q) q̈ + h(q, q̇) = τ + Jᵀ F ---
function mass_matrix(p, q)
    a1 = p.I1 + p.I2 + p.m2*p.l1^2      # constant part of M11
    a2 = p.m2*p.l1*p.s2                 # coupling: forearm COM seen from the shoulder
    a3 = p.I2
    c2 = cos(q[2])
    [a1 + 2a2*c2   a3 + a2*c2;
     a3 + a2*c2    a3]
end
"Coriolis/centripetal torques h(q, q̇)."
function coriolis(p, q, qd)
    a2 = p.m2*p.l1*p.s2; s2 = sin(q[2])
    [-a2*s2*(2qd[1]*qd[2] + qd[2]^2),
      a2*s2*qd[1]^2]
end

"Inverse dynamics (numbers only): τ = M q̈ + h − Jᵀ F_hand."
inverse_dynamics(p, q, qd, qdd, F=zeros(2)) = mass_matrix(p, q)*qdd .+ coriolis(p, q, qd) .- hand_jacobian(p, q)'*F

"Inverse kinematics for the right-arm (elbow-right) branch."
function ik(p, x)
    r2 = x[1]^2 + x[2]^2
    c2 = clamp((r2 - p.l1^2 - p.l2^2)/(2p.l1*p.l2), -1, 1)
    q2 = acos(c2)
    q1 = atan(x[2], x[1]) - atan(p.l2*sin(q2), p.l1 + p.l2*cos(q2))
    [q1, q2]
end

# ============================================================== force fields
"""
    field_force(kind, x, v; b_curl=13, b_kist=150, target_y)

Force the robot applies to the hand (N) at hand position `x` and velocity `v`.

* `:none`
* `:curl_cw`, `:curl_ccw` — velocity-dependent curl field F = B v with B = b·[0 1; −1 0]
  (clockwise) or its negative. F ⟂ v, so the field does NO work on the hand.
* `:kistemaker` — Kistemaker, Wong & Gribble (2010) Eq. 2, outward movements:
  F_x = 0, F_y = b(−ẏ + ẋ)·|y_target − y|.  Moving forward is resisted, moving right
  is assisted, and the field vanishes at the target. We use (y_target − y) without the
  absolute value: inside the workspace y < y_target, so it is identical and smooth.
"""
function field_force(kind::Symbol, x, v; b_curl=13.0, b_kist=150.0, target_y=0.0)
    if kind === :none
        return [0.0, 0.0]
    elseif kind === :curl_cw
        return [b_curl*v[2], -b_curl*v[1]]
    elseif kind === :curl_ccw
        return [-b_curl*v[2], b_curl*v[1]]
    elseif kind === :kistemaker
        return [0.0, b_kist*(-v[2] + v[1])*(target_y - x[2])]
    else
        error("unknown field $kind")
    end
end

# ============================================================== the robot (InMotion2 / MIT-Manus)
# Wilhelms-Tricarico R (2004) "Methods for compensation of dynamic robot properties", project report.
# Parallelogram, direct-drive, 2 DOF. Both joint angles are ABSOLUTE (measured from the +x axis),
# links of length l1 (driven by motor 1) and l3 (driven by motor 2):
#     handle = base + l1 (cos q1, sin q1) + l3 (cos q2, sin q2)
# Lagrangian L = ½c1 q̇1² + c2 cos(q1−q2) q̇1 q̇2 + ½c3 q̇2²  ⇒
#     B(q) q̈ + c(q, q̇) = τ_motor − J_rᵀ h,
#     B = [c1  c2 cos(q1−q2); c2 cos(q1−q2)  c3],  c = c2 sin(q1−q2) [q̇2²; −q̇1²],
# where h is the force the robot applies to whatever holds the handle (the hand).
# (The report measures angles clockwise; the equations have the same form either way.)

"""
Robot parameters. Defaults: the regression estimates of Wilhelms-Tricarico (2004, §4.2,
3-parameter column; they include the ≈0.3 kg handle) and the link lengths from the
manufacturer's table. `base` is the robot's base in OUR coordinates (shoulder at the
origin) — the report gives no subject placement, so this is an assumption: the robot sits
across the table, in front of the subject. `elbow = +1/−1` picks the configuration branch.
"""
Base.@kwdef struct RobotParams
    c1::Float64 = 0.3912     # kg m²
    c2::Float64 = 0.2325     # kg m²
    c3::Float64 = 0.2266     # kg m²
    l1::Float64 = 0.4064     # m (16")
    l3::Float64 = 0.51435    # m (20.25")
    base::Vector{Float64} = [0.0, 1.0]
    elbow::Float64 = 1.0
end
inmotion2_params(; kw...) = RobotParams(; kw...)

robot_position(r, qr) = [r.base[1] + r.l1*cos(qr[1]) + r.l3*cos(qr[2]),
                         r.base[2] + r.l1*sin(qr[1]) + r.l3*sin(qr[2])]
robot_elbow(r, qr) = [r.base[1] + r.l1*cos(qr[1]), r.base[2] + r.l1*sin(qr[1])]
robot_jacobian(r, qr) = [-r.l1*sin(qr[1])  -r.l3*sin(qr[2]);
                          r.l1*cos(qr[1])   r.l3*cos(qr[2])]
"J̇_r q̇_r: the velocity-product part of the handle acceleration."
robot_jdot_qdot(r, qr, qrd) = [-r.l1*cos(qr[1])*qrd[1]^2 - r.l3*cos(qr[2])*qrd[2]^2,
                               -r.l1*sin(qr[1])*qrd[1]^2 - r.l3*sin(qr[2])*qrd[2]^2]
robot_mass_matrix(r, qr) = [r.c1                   r.c2*cos(qr[1] - qr[2]);
                            r.c2*cos(qr[1] - qr[2])  r.c3]
robot_coriolis(r, qr, qrd) = [ r.c2*sin(qr[1] - qr[2])*qrd[2]^2,
                              -r.c2*sin(qr[1] - qr[2])*qrd[1]^2]

"Robot joint angles for a handle position (numbers only; branch set by `r.elbow`)."
function robot_ik(r, x)
    d = x .- r.base; dist2 = d[1]^2 + d[2]^2
    cδ = clamp((dist2 - r.l1^2 - r.l3^2)/(2r.l1*r.l3), -1, 1)
    δ = r.elbow*acos(cδ)                       # q2 − q1
    q1 = atan(d[2], d[1]) - atan(r.l3*sin(δ), r.l1 + r.l3*cos(δ))
    [q1, q1 + δ]
end

"Robot joint state (q_r, q̇_r, q̈_r) for a handle position, velocity and acceleration (numbers)."
function robot_state(r, x, v, a)
    qr = robot_ik(r, x); Jr = robot_jacobian(r, qr)
    qrd = Jr \ v
    qrdd = Jr \ (a .- robot_jdot_qdot(r, qr, qrd))
    (qr=qr, qrd=qrd, qrdd=qrdd)
end

"The robot's inertia as felt at the handle, M_r = J_r⁻ᵀ B J_r⁻¹ (kg)."
function robot_endpoint_inertia(r, x)
    qr = robot_ik(r, x); Ji = inv(robot_jacobian(r, qr))
    Ji'*robot_mass_matrix(r, qr)*Ji
end

"The human arm's inertia as felt at the hand, Λ = (J M⁻¹ Jᵀ)⁻¹ (kg)."
function endpoint_inertia(p, x)
    q = ik(p, x); J = hand_jacobian(p, q)
    inv(J*inv(mass_matrix(p, q))*J')
end

# ============================================================== min jerk
"Minimum-jerk position fraction: 10s³ − 15s⁴ + 6s⁵ (s = t/T ∈ [0,1])."
min_jerk(s) = 10s^3 - 15s^4 + 6s^5
"(position, velocity, acceleration, jerk) of a min-jerk move of distance D in time T, at time t."
function min_jerk_profile(t, D, T)
    s = clamp(t/T, 0, 1)
    (x = D*min_jerk(s),
     v = D/T*(30s^2 - 60s^3 + 30s^4),
     a = D/T^2*(60s - 180s^2 + 120s^3),
     j = D/T^3*(60 - 360s + 360s^2))
end

# ============================================================== hand paths
# A "path" here is a SHAPE: the sideways offset of the hand from the start→target line,
# as a function of how far along the line it is (λ = 0 at the start, 1 at the target).
# Positive offsets bow to the RIGHT of the direction of travel. The timing along the
# path is left to the optimizer (Lectures 3–4 tracking problems).

"""
    bow(A)

Symmetric bow of peak A (m) at mid-path: offset(λ) = A·64 λ³(1−λ)³. Zero offset, slope
and curvature at both ends, so the hand leaves and arrives along the straight line.
"""
bow(A) = λ -> A*64*λ^3*(1 - λ)^3

"""
    fit_offset(sol; degree=7)

Fit a polynomial offset(λ) = λ(1−λ)·Σₖ cₖ λᵏ to a solution's hand path (least squares),
so any optimal path can be re-imposed as a tracking constraint. Returns (offset, rms error).
"""
function fit_offset(sol; degree=7)
    st, tg = sol["task"].start, sol["task"].target
    d = tg .- st; D = norm(d); e = d ./ D; nrm = [e[2], -e[1]]
    H = sol["hand"]
    λ = [dot(H[:, k] .- st, e)/D for k in 1:size(H, 2)]
    o = [dot(H[:, k] .- st, nrm) for k in 1:size(H, 2)]
    B = hcat([λ .* (1 .- λ) .* λ.^k for k in 0:degree]...)
    c = B \ o
    off = l -> l*(1 - l)*sum(c[k+1]*l^k for k in 0:degree)
    (offset=off, rms=sqrt(sum(abs2, B*c .- o)/length(o)))
end

"Hand path for an offset function, with a minimum-jerk time course along it (used for guesses and drawing)."
function curved_path(start, target, offset)
    d = target .- start; D = norm(d); e = d ./ D
    n = [e[2], -e[1]]                      # right-hand normal of the travel direction
    s -> (λ = min_jerk(s); start .+ e .* (D*λ) .+ n .* offset(λ))
end
curved_path(start, target, A::Real) = curved_path(start, target, bow(A))

# ============================================================== the problem
"""
    ReachTask(; start, target, T, ...)

What the arm has to do.
* `start`, `target` — hand positions (m), shoulder at the origin
* `T` — movement time (s); `nothing` = free (then add a time cost `k` in `CostWeights`)
* `field` — `:none`, `:curl_cw`, `:curl_ccw`, `:kistemaker` (see `field_force`)
* `m_endpoint` — extra point mass at the hand (kg), e.g. a manipulandum handle
* `track` — `nothing`, or an offset function λ ↦ sideways offset (m) (see `bow`, `fit_offset`):
  the hand is kept within ±`track_tol` of that path SHAPE; the optimizer still chooses the timing along it
"""
Base.@kwdef struct ReachTask
    start::Vector{Float64} = [0.0, 0.20]
    target::Vector{Float64} = [0.0, 0.50]
    T::Union{Nothing,Float64} = 0.65
    T_bounds::Tuple{Float64,Float64} = (0.15, 3.0)
    field::Symbol = :none
    b_curl::Float64 = 13.0
    b_kist::Float64 = 150.0
    m_endpoint::Float64 = 0.0
    track::Any = nothing
    track_tol::Float64 = 0.002   # tube half-width for tracking (m)
    # the robot (Lecture appendix): :none = ideal (only `field` and an isotropic `m_endpoint`), or
    # :inmotion2 = the full robot dynamics coupled at the handle, with controller `robot_comp`:
    #   :none     — motors command the field only, τ_m = J_rᵀ F_field (robot inertia + Coriolis felt)
    #   :coriolis — Coriolis cancelled, inertia not (report Eq. 13)
    #   :full     — Coriolis + inertia compensation to an isotropic mass `robot_D` (report Eq. 16)
    robot::Symbol = :none
    robot_comp::Symbol = :none
    robot_D::Float64 = 1.0
    robot_params::RobotParams = RobotParams()
end

"""
    CostWeights(; c_W=4.2, c_t=0.085, k=0, c_frr2=0, eps=1e-7)

E = c_W Σ∫[q̇τ]₊ dt + c_t Σ∫[τ̈]₊ dt + c_frr2 Σ∫τ̈² dt + k·T.

* `c_W = 4.2` — metabolic cost per unit positive work (25 % efficiency, Margaria 1963;
  Wong et al. 2021 lump negative work into it, so positive work counts ×4.2).
* `c_t = 8.5e-2` — force-rate coefficient, fitted to cyclic reaching (Wong et al. 2021, Fig. 6B).
* `k` — cost of time, J/s (25 J/s in Wong et al. 2021, Fig. 10).
* `c_frr2` — optional QUADRATIC force-rate-rate cost (∫τ̈², a "torque-change-change" smoothness cost).
* `eps` — a tiny ∫τ̈² so the L1 problem has a unique answer. Keep it small; we report it.
"""
Base.@kwdef struct CostWeights
    c_W::Float64 = 4.2
    c_t::Float64 = 8.5e-2
    k::Float64 = 0.0
    c_frr2::Float64 = 0.0
    eps::Float64 = 1e-7
end

"Linear interpolation on a sorted grid (used for warm starts and resampling)."
function interp1(xs::AbstractVector, ys::AbstractVector, x::Real)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    w = (x - xs[i])/(xs[i+1] - xs[i])
    (1 - w)*ys[i] + w*ys[i+1]
end

"""
Initial guess: minimum-jerk straight-line hand path (or the tracked path), joint angles
by IK, torques by inverse dynamics, torque rates by finite differences.
Returns functions of normalized time s.
"""
function straight_guess(p, task, T)
    path = curved_path(task.start, task.target, task.track === nothing ? bow(0.0) : task.track)
    ss = range(0, 1, length=401); h = step(ss)
    Q = hcat([ik(p, path(s)) for s in ss]...)
    d(X) = hcat([(X[:, min(k+1, end)] .- X[:, max(k-1, 1)]) ./ ((min(k+1, size(X,2)) - max(k-1, 1))*h) for k in 1:size(X, 2)]...)
    QD = d(Q) ./ T; QDD = d(QD) ./ T
    TAU = hcat([inverse_dynamics(p, Q[:, k], QD[:, k], QDD[:, k]) for k in 1:length(ss)]...)
    TAUD = d(TAU) ./ T; U = d(TAUD) ./ T
    f(X, i) = s -> interp1(ss, X[i, :], s)
    (q=[f(Q, i) for i in 1:2], qd=[f(QD, i) for i in 1:2], qdd=[f(QDD, i) for i in 1:2],
     tau=[f(TAU, i) for i in 1:2], taud=[f(TAUD, i) for i in 1:2], u=[f(U, i) for i in 1:2])
end

"Warm start from a previous solution (any T): resample its arrays on normalized time."
function solution_guess(sol)
    ss = sol["t"] ./ sol["T"]
    f(key, i) = s -> interp1(ss, sol[key][i, :], s)
    (q=[f("q", i) for i in 1:2], qd=[f("qd", i) for i in 1:2], qdd=[f("qdd", i) for i in 1:2],
     tau=[f("tau", i) for i in 1:2], taud=[f("taud", i) for i in 1:2], u=[f("u", i) for i in 1:2])
end

"""
    solve_reach(task, cost; p=arm_params(), n=61, guess=nothing, verbose=false)

Solve the reaching optimal-control problem. Returns a NamedTuple of plain arrays
(cache-friendly): times, joint/hand trajectories, torques, cost breakdown, status.
`guess` may be a previous solution (warm start); otherwise a min-jerk straight line.
"""
function solve_reach(task::ReachTask, cost::CostWeights; p=arm_params(), n=61, guess=nothing,
                     verbose=false, max_iter=3000, tol=1e-8)
    model = InfiniteModel(Ipopt.Optimizer)
    set_optimizer_attribute(model, "print_level", verbose ? 5 : 0)
    set_optimizer_attribute(model, "sb", "yes")          # no Ipopt banner
    set_optimizer_attribute(model, "max_iter", max_iter)
    set_optimizer_attribute(model, "tol", tol)

    # normalized time s ∈ [0, 1]; t = T s
    @infinite_parameter(model, s in [0, 1], num_supports=n, derivative_method=OrthogonalCollocation(2))
    free_T = task.T === nothing
    if free_T
        T0 = guess === nothing ? 0.65 : guess["T"]
        @variable(model, task.T_bounds[1] <= T <= task.T_bounds[2], start=T0)
    else
        T = task.T
    end

    # ---------------- states and control (all functions of s)
    @variable(model, q[1:2], Infinite(s))       # joint angles (rad)
    @variable(model, qd[1:2], Infinite(s))      # joint angular velocities (rad/s)
    @variable(model, qdd[1:2], Infinite(s))     # joint accelerations (rad/s²) — algebraic
    @variable(model, τ[1:2], Infinite(s))       # joint torques (N m)
    @variable(model, τd[1:2], Infinite(s))      # torque rates (N m/s)
    @variable(model, u[1:2], Infinite(s))       # CONTROL: torque rate-rate τ̈ (N m/s²)
    @variable(model, wpos[1:2] >= 0, Infinite(s))  # slack: positive joint power [q̇τ]₊
    @variable(model, upos[1:2] >= 0, Infinite(s))  # slack: positive force-rate-rate [τ̈]₊

    # ---------------- kinematic chain of integrators, d/ds = T d/dt
    @constraint(model, [i=1:2], ∂(q[i], s) == T*qd[i])
    @constraint(model, [i=1:2], ∂(qd[i], s) == T*qdd[i])
    @constraint(model, [i=1:2], ∂(τ[i], s) == T*τd[i])
    @constraint(model, [i=1:2], ∂(τd[i], s) == T*u[i])

    # ---------------- equations of motion: M q̈ + h = τ + Jᵀ F_hand
    xh = hand_position(p, q); J = hand_jacobian(p, q)
    vh = J*qd
    F = field_force(task.field, xh, vh; b_curl=task.b_curl, b_kist=task.b_kist, target_y=task.target[2])
    if task.m_endpoint > 0          # a point mass at the hand pushes back with −m a_hand
        ah = J*qdd .+ jdot_qdot(p, q, qd)
        F = F .- task.m_endpoint .* ah
    end
    if task.robot === :inmotion2
        F = add_robot!(model, s, task, p, q, qd, qdd, xh, vh, J)   # force of the robot on the hand
    end
    M = mass_matrix(p, q); h = coriolis(p, q, qd); JtF = J'*F
    @constraint(model, [i=1:2], M[i, 1]*qdd[1] + M[i, 2]*qdd[2] + h[i] == τ[i] + JtF[i])

    # ---------------- positive-part slacks: wpos ≥ q̇τ, upos ≥ τ̈ (and both ≥ 0)
    @constraint(model, [i=1:2], wpos[i] >= qd[i]*τ[i])
    @constraint(model, [i=1:2], upos[i] >= u[i])

    # ---------------- boundary conditions: rest → rest, zero torque and torque rate at both ends
    q0 = ik(p, task.start); qT = ik(p, task.target)
    @constraint(model, [i=1:2], q[i](0) == q0[i]);  @constraint(model, [i=1:2], q[i](1) == qT[i])
    @constraint(model, [i=1:2], qd[i](0) == 0);     @constraint(model, [i=1:2], qd[i](1) == 0)
    @constraint(model, [i=1:2], τ[i](0) == 0);      @constraint(model, [i=1:2], τ[i](1) == 0)
    @constraint(model, [i=1:2], τd[i](0) == 0);     @constraint(model, [i=1:2], τd[i](1) == 0)

    # ---------------- optional: keep the hand on a prescribed path SHAPE (tracking problems)
    if task.track !== nothing
        # progress along the start→target line, λ, and the sideways offset of the hand.
        # The offset function is plain polynomial algebra, so it works on JuMP expressions.
        d = task.target .- task.start; D = norm(d); e = d ./ D; nrm = [e[2], -e[1]]
        rel = xh .- task.start
        λ = (rel[1]*e[1] + rel[2]*e[2])/D
        err = rel[1]*nrm[1] + rel[2]*nrm[2] - task.track(λ)
        # A TUBE, not an equality: "offset == path" at every collocation node is a
        # high-index constraint (position is four integrations away from the control τ̈),
        # and collocation + Ipopt stall on it. A ±track_tol tube is honest and solves.
        @constraint(model, err <= task.track_tol)
        @constraint(model, err >= -task.track_tol)
    end

    # ---------------- objective (integrals over s, times T to make them integrals over t)
    @expression(model, work_int, integral(wpos[1] + wpos[2], s))     # ∫[q̇τ]₊ ds
    @expression(model, frr_int, integral(upos[1] + upos[2], s))      # ∫[τ̈]₊ ds
    @expression(model, u2_int, integral(u[1]^2 + u[2]^2, s))         # ∫τ̈² ds
    @objective(model, Min, T*(cost.c_W*work_int + cost.c_t*frr_int + (cost.c_frr2 + cost.eps)*u2_int) + cost.k*T)

    # ---------------- initial guess
    G = guess === nothing ? straight_guess(p, task, free_T ? 0.65 : T) : solution_guess(guess)
    for i in 1:2
        set_start_value_function(q[i], G.q[i]);     set_start_value_function(qd[i], G.qd[i])
        set_start_value_function(qdd[i], G.qdd[i]); set_start_value_function(τ[i], G.tau[i])
        set_start_value_function(τd[i], G.taud[i]); set_start_value_function(u[i], G.u[i])
        set_start_value_function(wpos[i], sv -> max(0.0, G.qd[i](sv)*G.tau[i](sv)))
        set_start_value_function(upos[i], sv -> max(0.0, G.u[i](sv)))
    end
    task.robot === :inmotion2 && robot_guess!(model, task, p, G)

    tsolve = @elapsed optimize!(model)
    status = string(termination_status(model))

    # ---------------- unpack on ALL supports (public + internal collocation nodes)
    ss = supports(s, label=All)
    val(x) = value(x, label=All)
    Tv = free_T ? value(T) : T
    M2(v) = Matrix(hcat(val(v[1]), val(v[2]))')
    Q, QD, QDD, TAU, TAUD, U = M2(q), M2(qd), M2(qdd), M2(τ), M2(τd), M2(u)
    extra = Dict{String,Any}()
    if task.robot === :inmotion2
        extra["F_robot"] = M2(model[:Fh]); extra["qr"] = M2(model[:qr])
    end
    pack_solution(p, task, cost, Tv, collect(ss), Q, QD, QDD, TAU, TAUD, U;
                  status=status, objective=objective_value(model), solve_time=tsolve, n=n, extra=extra)
end

"""
Couple the robot to the hand (handle = hand). Robot joint angles, rates and accelerations
are ALGEBRAIC variables tied to the hand by the kinematic map at every node (no extra
differential states, so no index problem); the handle force Fh closes the robot's own
equation of motion,  B q̈_r + c = τ_m − J_rᵀ Fh,  for the chosen controller τ_m.
Returns Fh, the force of the robot on the hand, for the arm's equations of motion.
"""
function add_robot!(model, s, task, p, q, qd, qdd, xh, vh, J)
    r = task.robot_params
    @variable(model, qr[1:2], Infinite(s))       # robot joint angles (absolute)
    @variable(model, qrd[1:2], Infinite(s))      # … rates
    @variable(model, qrdd[1:2], Infinite(s))     # … accelerations
    @variable(model, Fh[1:2], Infinite(s))       # force of the robot on the hand (N) = report's h
    ah = J*qdd .+ jdot_qdot(p, q, qd)            # hand acceleration
    Jr = robot_jacobian(r, qr); jdr = robot_jdot_qdot(r, qr, qrd)
    xr = robot_position(r, qr)
    @constraint(model, [i=1:2], xr[i] == xh[i])                                  # handle = hand
    @constraint(model, [i=1:2], Jr[i, 1]*qrd[1] + Jr[i, 2]*qrd[2] == vh[i])       # velocities
    @constraint(model, [i=1:2], Jr[i, 1]*qrdd[1] + Jr[i, 2]*qrdd[2] + jdr[i] == ah[i])  # accelerations
    # the field the experimenter commands, computed from the handle's position and velocity
    Ff = field_force(task.field, xh, vh; b_curl=task.b_curl, b_kist=task.b_kist, target_y=task.target[2])
    B = robot_mass_matrix(r, qr); c = robot_coriolis(r, qr, qrd); JrT = Jr'
    if task.robot_comp === :none                   # τ_m = J_rᵀ F_field
        τm = JrT*Ff
    elseif task.robot_comp === :coriolis           # τ_m = J_rᵀ F_field + c − B J_r⁻¹ J̇_r q̇_r  (report Eq. 13)
        @variable(model, z[1:2], Infinite(s))      # z = J_r⁻¹ J̇_r q̇_r
        @constraint(model, [i=1:2], Jr[i, 1]*z[1] + Jr[i, 2]*z[2] == jdr[i])
        τm = JrT*Ff .+ c .- B*z
    elseif task.robot_comp === :full               # report Eq. 54: τ_m = B J_r⁻¹[D⁻¹(F_field − h) − J̇_r q̇_r] + c + J_rᵀ h
        @variable(model, z[1:2], Infinite(s))      # z = J_r⁻¹[D⁻¹(F_field − h) − J̇_r q̇_r]; h is the MEASURED handle force
        w = (Ff .- Fh) ./ task.robot_D .- jdr
        @constraint(model, [i=1:2], Jr[i, 1]*z[1] + Jr[i, 2]*z[2] == w[i])
        τm = B*z .+ c .+ JrT*Fh
    else
        error("unknown robot_comp $(task.robot_comp)")
    end
    JrTF = JrT*Fh
    @constraint(model, [i=1:2], B[i, 1]*qrdd[1] + B[i, 2]*qrdd[2] + c[i] == τm[i] - JrTF[i])
    collect(Fh)
end

"Start values for the robot variables, from the arm's initial guess."
function robot_guess!(model, task, p, G)
    r = task.robot_params
    function st(sv)
        q = [G.q[1](sv), G.q[2](sv)]; qd = [G.qd[1](sv), G.qd[2](sv)]; qdd = [G.qdd[1](sv), G.qdd[2](sv)]
        x = hand_position(p, q); J = hand_jacobian(p, q)
        robot_state(r, x, J*qd, J*qdd .+ jdot_qdot(p, q, qd))
    end
    for i in 1:2
        set_start_value_function(model[:qr][i], sv -> st(sv).qr[i])
        set_start_value_function(model[:qrd][i], sv -> st(sv).qrd[i])
        set_start_value_function(model[:qrdd][i], sv -> st(sv).qrdd[i])
    end
end

"Post-process: hand kinematics, fields, power, and the energy breakdown (plain arrays only)."
function pack_solution(p, task, cost, T, ss, Q, QD, QDD, TAU, TAUD, U; status="", objective=NaN, solve_time=NaN, n=0,
                       extra=Dict{String,Any}())
    t = T .* ss; nk = length(t)
    X = hcat([hand_position(p, Q[:, k]) for k in 1:nk]...)
    E = hcat([elbow_position(p, Q[:, k]) for k in 1:nk]...)
    V = hcat([hand_velocity(p, Q[:, k], QD[:, k]) for k in 1:nk]...)
    A = hcat([hand_jacobian(p, Q[:, k])*QDD[:, k] .+ jdot_qdot(p, Q[:, k], QD[:, k]) for k in 1:nk]...)
    # force of the robot on the hand: the commanded field (ideal robot), or the full coupled robot's
    Fh = haskey(extra, "F_robot") ? extra["F_robot"] :
         hcat([field_force(task.field, X[:, k], V[:, k]; b_curl=task.b_curl, b_kist=task.b_kist, target_y=task.target[2]) for k in 1:nk]...)
    if task.m_endpoint > 0 && !haskey(extra, "F_robot")   # include the handle's inertial push in the force on the hand
        Fh = Fh .- task.m_endpoint .* A
    end
    P = QD .* TAU                                  # joint power (W)
    eb = energy_breakdown(t, P, U, cost)
    Wext = trapz(t, -sum(Fh .* V, dims=1)[:])      # work the hand does ON the robot (Kistemaker's E_ext)
    Dict{String,Any}(
        "status" => status, "objective" => objective, "solve_time" => solve_time, "n" => n,
        "T" => T, "t" => t, "q" => Q, "qd" => QD, "qdd" => QDD, "tau" => TAU, "taud" => TAUD, "u" => U,
        "hand" => X, "elbow" => E, "hand_v" => V, "hand_a" => A, "speed" => vec(sqrt.(sum(V.^2, dims=1))),
        "F_field" => Fh, "power" => P, "W_ext" => Wext,
        "E_work" => eb.E_work, "E_fr" => eb.E_fr, "E" => eb.E, "E_time" => cost.k*T, "J" => eb.E + cost.k*T,
        "W_pos" => eb.W_pos, "FRR_pos" => eb.FRR_pos,
        "task" => (start=task.start, target=task.target, T=task.T, field=task.field, b_curl=task.b_curl,
                   b_kist=task.b_kist, m_endpoint=task.m_endpoint, tracked=task.track !== nothing,
                   robot=task.robot, robot_comp=task.robot_comp, robot_D=task.robot_D,
                   robot_base=task.robot_params.base),
        "cost" => (c_W=cost.c_W, c_t=cost.c_t, k=cost.k, c_frr2=cost.c_frr2, eps=cost.eps),
        extra...)
end

trapz(t, y) = sum((t[k+1] - t[k])*(y[k+1] + y[k])/2 for k in 1:length(t)-1)

"""
Energy, recomputed from the trajectories (not read off the slacks): positive joint work
W⁺ = Σ∫[q̇τ]₊ dt, positive force-rate-rate Σ∫[τ̈]₊ dt, and E = c_W W⁺ + c_t Σ∫[τ̈]₊.
"""
function energy_breakdown(t, P, U, cost)
    Wp = sum(trapz(t, max.(P[i, :], 0)) for i in 1:size(P, 1))
    Fp = sum(trapz(t, max.(U[i, :], 0)) for i in 1:size(U, 1))
    (W_pos=Wp, FRR_pos=Fp, E_work=cost.c_W*Wp, E_fr=cost.c_t*Fp, E=cost.c_W*Wp + cost.c_t*Fp)
end

# ============================================================== curvature & robust solves
"""
    perp_error(sol) -> (pe, max)

Signed perpendicular deviation of the hand from the start→target line (m), POSITIVE to the RIGHT of
the direction of travel. `pe` is taken at peak hand speed — Mattar & Ostry's (2007) "perpendicular
error" (PE); `max` is the largest deviation anywhere on the path. In a clockwise curl field the
field pushes the hand to the right, so an uncompensated reach has PE > 0.
"""
function perp_error(sol)
    st, tg = sol["task"].start, sol["task"].target
    e = normalize(tg .- st); nr = [e[2], -e[1]]
    H = sol["hand"]
    o = [dot(H[:, j] .- st, nr) for j in 1:size(H, 2)]
    (pe=o[argmax(sol["speed"])], max=o[argmax(abs.(o))])
end

solve_ok(sol) = sol["status"] in ("LOCALLY_SOLVED", "ALMOST_LOCALLY_SOLVED")

"""
    solve_reach_multistart(task, cost; bows=(-0.03, 0.03), kw...)

Free optimum from several starting guesses: the default straight-line guess, plus warm starts
from solutions that track paths bowed by each `bows` (m). Returns the cheapest converged solution
with `"starts"` = the energies every start reached (identical numbers = a robust optimum).
"""
function solve_reach_multistart(task::ReachTask, cost::CostWeights; bows=(-0.03, 0.03), kw...)
    cands = [solve_reach(task, cost; kw...)]
    for A in bows
        tr = ReachTask(start=task.start, target=task.target, T=task.T, T_bounds=task.T_bounds, field=task.field,
                       b_curl=task.b_curl, b_kist=task.b_kist, m_endpoint=task.m_endpoint, track=bow(A),
                       robot=task.robot, robot_comp=task.robot_comp, robot_D=task.robot_D, robot_params=task.robot_params)
        g = solve_reach(tr, cost; kw...)
        solve_ok(g) && push!(cands, solve_reach(task, cost; guess=g, kw...))
    end
    good = filter(solve_ok, cands)
    best = isempty(good) ? cands[1] : good[argmin([c["E"] for c in good])]
    best["starts"] = [c["E"] for c in good]
    best
end

# ============================================================== caching
cache_path(name) = joinpath(CACHE_DIR, name * ".jls")

"""
    cached(f, name; refresh=false)

Load `name` from walkthroughs/reaching/cache/ if it is there, otherwise run `f()` and
save the result. Results are plain Dicts/arrays, so the caches survive code edits.
Delete the file (or pass `refresh=true`) to recompute.
"""
function cached(f, name::AbstractString; refresh::Bool=false)
    path = cache_path(name)
    if !refresh && isfile(path)
        return deserialize(path)
    end
    res = f()
    mkpath(CACHE_DIR)
    serialize(path, res)
    res
end

# ============================================================== data
"""
Digitized speed profiles from Flash & Hogan (1985) J Neurosci 5:1688, Fig. 3b
(movements A: targets 3→6, and B: targets 1→4; measured = dashed, predicted = solid).
Returns Dict movement => Dict trace => (t, v) with t in s and v in m/s.
"""
function load_flash_hogan()
    lines = readlines(joinpath(DATA_DIR, "flash_hogan_1985_fig3b.csv"))
    rows = [split(l, ',') for l in lines if !startswith(l, "#") && !startswith(l, "movement")]
    out = Dict{String,Dict{String,Any}}()
    for r in rows
        m, tr, t, v = r[1], r[2], parse(Float64, r[3]), parse(Float64, r[4])/1000
        d = get!(out, m, Dict{String,Any}())
        tv = get!(d, tr, (t=Float64[], v=Float64[]))
        push!(tv.t, t); push!(tv.v, v)
    end
    for d in values(out), (k, tv) in d
        o = sortperm(tv.t); d[k] = (t=tv.t[o], v=tv.v[o])
    end
    out
end

# ============================================================== browser animation
# A small, dependency-free SVG + JavaScript player. It works inside Pluto AND in the
# static HTML export (no Julia needed to play it). Each "series" is a set of points
# per frame (shoulder, elbow, hand for the arm; one point for a point mass) drawn as a
# polyline, plus the trace of its last point and a speed curve with a moving cursor.

_json(x::Real) = isfinite(x) ? @sprintf("%.5g", x) : "null"
_json(x::AbstractString) = "\"" * replace(x, "\"" => "\\\"") * "\""
_json(x::AbstractVector) = "[" * join(_json.(x), ",") * "]"
_json(x::AbstractMatrix) = "[" * join([_json(x[:, k]) for k in 1:size(x, 2)], ",") * "]"   # column-major frames
_json(x::Symbol) = _json(string(x))
_json(d::Dict) = "{" * join([_json(string(k)) * ":" * _json(v) for (k, v) in d], ",") * "}"
_json(x::Nothing) = "null"
_json(x::Bool) = x ? "true" : "false"

"""
    arm_animation(sols; labels, colors, title, targets, xlim, ylim, show_force=false)

Scrubbable animation of one or more arm solutions (Dicts from `solve_reach`) on a common
time axis: the arm in the workspace (left) and the hand speed (right). With
`show_force=true` an arrow at each hand shows the force the robot applies to it.
"""
function arm_animation(sols::AbstractVector; labels=["solution $i" for i in 1:length(sols)],
                       colors=PAL, title="", targets=nothing, xlim=(-0.22, 0.36), ylim=(-0.05, 0.58),
                       width=660, show_force=false)
    series = Dict{String,Any}[]
    for (i, sl) in enumerate(sols)
        nk = length(sl["t"])
        pts = [vcat(0.0, 0.0, sl["elbow"][:, k], sl["hand"][:, k]) for k in 1:nk]   # x0,y0,x1,y1,x2,y2
        d = Dict{String,Any}("label" => labels[i], "color" => colors[mod1(i, length(colors))],
                             "t" => sl["t"], "pts" => hcat(pts...), "speed" => sl["speed"])
        show_force && (d["F"] = sl["F_field"])
        if haskey(sl, "qr")       # draw the robot too (grey): base → robot elbow → handle
            r = RobotParams(base=sl["task"].robot_base)
            d["pts2"] = hcat([vcat(r.base, robot_elbow(r, sl["qr"][:, k]), robot_position(r, sl["qr"][:, k])) for k in 1:nk]...)
        end
        push!(series, d)
    end
    tg = targets === nothing ? [sols[1]["task"].start, sols[1]["task"].target] : targets
    _player_html(series, title, tg, xlim, ylim, width; drawlinks=true, side=true)
end

"""
    point_animation(runs; labels, colors, title)

Same player for 1-D point-mass runs: each run is (t, x, v) and is drawn as a "car"
moving left → right on its own lane, with the speed traces underneath.
"""
function point_animation(runs::AbstractVector; labels=["run $i" for i in 1:length(runs)],
                         colors=PAL, title="", width=660)
    series = Dict{String,Any}[]
    nr = length(runs); D = maximum(maximum(r.x) for r in runs)
    for (i, r) in enumerate(runs)
        lane = (nr - i + 0.5)/nr*0.3
        pts = hcat([[r.x[k], lane] for k in 1:length(r.t)]...)
        push!(series, Dict{String,Any}("label" => labels[i], "color" => colors[mod1(i, length(colors))],
                                       "t" => r.t, "pts" => pts, "speed" => abs.(r.v)))
    end
    _player_html(series, title, [[0.0, -0.02], [D, -0.02]], (-0.05D, 1.05D), (-0.04, 0.32), width;
                 drawlinks=false, side=false)
end

function _player_html(series, title, targets, xlim, ylim, width; drawlinks=true, side=true)
    id = "anim" * string(rand(UInt32), base=16)
    data = "[" * join(_json.(series), ",") * "]"
    if side                       # arm left, speed right
        W1 = round(Int, 0.44width); W2 = width - W1 - 8; H1 = 360; H2 = 360
    else                          # lanes on top, speed underneath
        W1 = width; W2 = width; H1 = 150; H2 = 170
    end
    HTML("""
<div id="$id" style="font-family: system-ui, sans-serif; font-size: 13px; max-width: $(width)px;">
  <div style="font-weight:600; margin-bottom:4px;">$title</div>
  <div style="display:flex; flex-wrap:wrap; gap:8px;">
    <svg class="ws" width="$W1" height="$H1" style="background:#fcfcfb; border:1px solid #ddd; border-radius:4px;"></svg>
    <svg class="sp" width="$W2" height="$H2" style="background:#fcfcfb; border:1px solid #ddd; border-radius:4px;"></svg>
  </div>
  <div style="display:flex; gap:10px; align-items:center; margin-top:6px;">
    <button class="pl">▶ play</button>
    <input class="sc" type="range" min="0" max="1000" value="0" style="flex:1;">
    <span class="tt" style="width:70px; text-align:right; font-variant-numeric: tabular-nums;">0.000 s</span>
    <select class="spd"><option value="0.1">0.1×</option><option value="0.25" selected>0.25×</option><option value="0.5">0.5×</option><option value="1">1×</option></select>
  </div>
  <div class="lg" style="display:flex; flex-wrap:wrap; gap:12px; margin-top:4px; color:#52514e;"></div>
</div>
<script>
(function(){
  const root = document.getElementById("$id");
  const S = $data;
  const targets = $(_json(hcat(targets...)));
  const xl = [$(xlim[1]), $(xlim[2])], yl = [$(ylim[1]), $(ylim[2])];
  const ws = root.querySelector(".ws"), sp = root.querySelector(".sp");
  const W1 = $W1, H1 = $H1, W2 = $W2, H2 = $H2, ns = "http://www.w3.org/2000/svg";
  // workspace: equal scaling in x and y, centred
  // (the arm view uses equal x/y scaling; the 1-D "lanes" view stretches x to the full width)
  const sc = Math.min((W1-20)/(xl[1]-xl[0]), (H1-20)/(yl[1]-yl[0]));
  const scx = $(side ? "sc" : "(W1-30)/(xl[1]-xl[0])"), scy = $(side ? "sc" : "(H1-20)/(yl[1]-yl[0])");
  const xm = (xl[0]+xl[1])/2, ym = (yl[0]+yl[1])/2;
  const X = x => W1/2 + (x - xm)*scx, Y = y => H1/2 - (y - ym)*scy;
  const Tmax = Math.max(...S.map(s => s.t[s.t.length-1]));
  const vmax = Math.max(...S.map(s => Math.max(...s.speed)))*1.08 || 1;
  const PX = t => 42 + t/Tmax*(W2-56), PY = v => H2 - 26 - v/vmax*(H2-44);
  const Fmax = Math.max(1e-9, ...S.filter(s => s.F).map(s => Math.max(...s.F.map(f => Math.hypot(f[0], f[1])))));
  const Fsc = 0.08/Fmax;     // longest force arrow = 8 cm on the drawing (arm view only)
  function el(tag, attrs, parent){ const e = document.createElementNS(ns, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; }
  // targets and shoulder
  targets.forEach(p => el("circle", {cx:X(p[0]), cy:Y(p[1]), r:8, fill:"none", stroke:"#888", "stroke-width":1.5}, ws));
  if ($(drawlinks ? "true" : "false")){
    el("circle", {cx:X(0), cy:Y(0), r:6, fill:"#52514e"}, ws);
    const sb = el("line", {x1:W1-12-0.1*sc, y1:H1-12, x2:W1-12, y2:H1-12, stroke:"#52514e", "stroke-width":2}, ws);
    const st = el("text", {x:W1-12-0.05*sc, y:H1-16, "font-size":11, fill:"#52514e", "text-anchor":"middle"}, ws); st.textContent = "10 cm";
  }
  // speed axes
  el("line", {x1:42, y1:H2-26, x2:W2-10, y2:H2-26, stroke:"#bbb"}, sp);
  el("line", {x1:42, y1:14, x2:42, y2:H2-26, stroke:"#bbb"}, sp);
  const lab = (txt, x, y, anchor) => { const e = el("text", {x:x, y:y, "font-size":11, fill:"#52514e", "text-anchor":anchor||"start"}, sp); e.textContent = txt; };
  lab("hand speed (m/s)", 46, 12); lab(vmax.toFixed(2), 38, PY(vmax)+4, "end"); lab("0", 38, H2-22, "end");
  lab("time (s)", (42+W2-10)/2, H2-3, "middle"); lab(Tmax.toFixed(2), PX(Tmax), H2-14, "middle"); lab("0", 42, H2-14, "middle");
  const items = S.map(s => {
    const k = s.pts[0].length;
    const tr = s.pts.map(p => X(p[k-2]) + "," + Y(p[k-1])).join(" ");
    el("polyline", {points: tr, fill:"none", stroke:s.color, "stroke-width":1.5, opacity:0.55}, ws);
    el("polyline", {points: s.t.map((t,i) => PX(t) + "," + PY(s.speed[i])).join(" "), fill:"none", stroke:s.color, "stroke-width":2}, sp);
    const link2 = s.pts2 ? el("polyline", {points:"", fill:"none", stroke:"#9a9890", "stroke-width":4, "stroke-linejoin":"round", opacity:0.55}, ws) : null;
    const link = el("polyline", {points:"", fill:"none", stroke:s.color, "stroke-width":$(drawlinks ? 6 : 0), "stroke-linecap":"round", "stroke-linejoin":"round", opacity:0.8}, ws);
    const arrow = s.F ? el("line", {stroke:s.color, "stroke-width":2.5, "marker-end":"url(#ah$id)"}, ws) : null;
    const dot = el("circle", {r:$(drawlinks ? 5 : 9), fill:s.color, stroke:"#fcfcfb", "stroke-width":2}, ws);
    const cur = el("circle", {r:4.5, fill:s.color, stroke:"#fcfcfb", "stroke-width":2}, sp);
    const lg = document.createElement("span"); lg.innerHTML = '<span style="display:inline-block;width:14px;height:3px;background:'+s.color+';vertical-align:middle;margin-right:5px;"></span>' + s.label;
    root.querySelector(".lg").appendChild(lg);
    return {s, link, link2, dot, cur, k, arrow};
  });
  if (S.some(s => s.F)){
    const defs = el("defs", {}, ws); const m = el("marker", {id:"ah$id", markerWidth:6, markerHeight:6, refX:5, refY:3, orient:"auto"}, defs);
    el("path", {d:"M0,0 L6,3 L0,6 z", fill:"#52514e"}, m);
    const t = el("text", {x:8, y:16, "font-size":11, fill:"#52514e"}, ws); t.textContent = "arrows: force of the robot on the hand (" + Fmax.toFixed(1) + " N max)";
  }
  const tline = el("line", {x1:0, y1:14, x2:0, y2:H2-26, stroke:"#52514e", "stroke-width":1}, sp);
  function at(s, t){ const ts = s.t, n = ts.length; if (t <= ts[0]) return 0; if (t >= ts[n-1]) return n-1;
    let lo = 0, hi = n-1; while (hi - lo > 1){ const m = (lo+hi)>>1; if (ts[m] <= t) lo = m; else hi = m; } return lo + (t-ts[lo])/(ts[hi]-ts[lo]); }
  function lerp(a, i){ const i0 = Math.floor(i), i1 = Math.min(i0+1, a.length-1), w = i - i0; return a[i0]*(1-w) + a[i1]*w; }
  function draw(t){
    items.forEach(it => {
      const i = at(it.s, t), i0 = Math.floor(i), i1 = Math.min(i0+1, it.s.t.length-1), w = i - i0;
      const p = it.s.pts[i0].map((v, j) => v*(1-w) + it.s.pts[i1][j]*w);
      let pts = []; for (let j = 0; j < it.k; j += 2) pts.push(X(p[j]) + "," + Y(p[j+1]));
      it.link.setAttribute("points", pts.join(" "));
      if (it.link2){ const r = it.s.pts2[i0].map((v, j) => v*(1-w) + it.s.pts2[i1][j]*w);
        it.link2.setAttribute("points", [0,2,4].map(j => X(r[j]) + "," + Y(r[j+1])).join(" ")); }
      const hx = p[it.k-2], hy = p[it.k-1];
      it.dot.setAttribute("cx", X(hx)); it.dot.setAttribute("cy", Y(hy));
      if (it.arrow){ const f0 = it.s.F[i0], f1 = it.s.F[i1]; const fx = f0[0]*(1-w)+f1[0]*w, fy = f0[1]*(1-w)+f1[1]*w;
        it.arrow.setAttribute("x1", X(hx)); it.arrow.setAttribute("y1", Y(hy));
        it.arrow.setAttribute("x2", X(hx + fx*Fsc)); it.arrow.setAttribute("y2", Y(hy + fy*Fsc));
        it.arrow.setAttribute("opacity", Math.hypot(fx, fy) > 0.02*Fmax ? 1 : 0); }
      const v = lerp(it.s.speed, i); const tt = Math.min(t, it.s.t[it.s.t.length-1]);
      it.cur.setAttribute("cx", PX(tt)); it.cur.setAttribute("cy", PY(v));
    });
    tline.setAttribute("x1", PX(t)); tline.setAttribute("x2", PX(t));
    root.querySelector(".tt").textContent = t.toFixed(3) + " s";
  }
  let t = 0, playing = false, last = null;
  const btn = root.querySelector(".pl"), scr = root.querySelector(".sc"), spd = root.querySelector(".spd");
  btn.onclick = () => { playing = !playing; btn.textContent = playing ? "❚❚ pause" : "▶ play"; last = null; };
  scr.oninput = () => { t = scr.value/1000*Tmax; draw(t); };
  function step(now){
    if (!document.body.contains(root)) return;          // the cell re-ran: stop this loop
    if (playing){ if (last !== null){ t += (now-last)/1000*parseFloat(spd.value); if (t > Tmax*1.15) t = 0; }
      last = now; draw(Math.min(t, Tmax)); scr.value = Math.min(t, Tmax)/Tmax*1000; }
    requestAnimationFrame(step);
  }
  draw(0); requestAnimationFrame(step);
})();
</script>
""")
end

end # module
