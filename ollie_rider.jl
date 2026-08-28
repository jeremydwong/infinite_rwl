module OllieRider
# Rider + board ollie. Three phases with free durations, A-reduced form for the pop:
#   phase 0 "load"  : board flat on both wheels (θ=0, y fixed), slides on ice in x
#   phase 1 "pop"   : rear wheel pivots on the ground, reduced coords (x_b, θ)
#   impact          : Newton tail strike (board only; legs are not rigid)
#   phase 2 "flight": full board (x_b, y_b, θ), level landing on both wheels
# Rider = point mass M at (x_r, y_r) connected to the board by two push-only
# telescoping legs: back leg to the tail tip, front leg to a free deck point s(t).
# Leg force F_i >= 0 acts along the leg (foot -> rider) on the rider and opposite on
# the board; complementarity F_i*(ℓ_i - ℓ_max) <= 0 lets a leg carry force only when
# shorter than ℓ_max (=1).  See reports/ollie_rider_mass_spec.md.

using JuMP, Ipopt, LinearAlgebra
include("ollie_reduced_check.jl")
using .OllieReducedCheck

export RiderOptions, RiderScales, force_limit_value, board_mass_kg, dimensional,
       solve_rider_ollie, rider_data, audit_rider_ollie, leg_extension_rates

# ---------------------------------------------------------------------------
# UNITS. The model is NONDIMENSIONAL: board mass = 1, g = 1, length unit = ℓ_max
# (the fully extended leg). All optimisation is done in these units. The HUMAN is
# the reference for the inputs below (rider_mass = M/m, force limit in rider body
# weights); `RiderScales` + `dimensional(...)` convert results to SI for reporting.
# ---------------------------------------------------------------------------
Base.@kwdef struct RiderOptions
    rider_mass::Float64 = 34.0        # rider / board mass ratio (170 lb / 5 lb = 77.1 kg / 2.27 kg; see report Appendix 1)
    leg_max::Float64 = 1.0
    leg_min::Float64 = 0.25
    force_limit_bw::Float64 = 2.0     # per-leg force limit in RIDER BODY WEIGHTS (M g)
    force_limit::Float64 = NaN        # explicit override in board-weight units; NaN = derived
    n_load::Int = 21
    n_support::Int = 31
    n_flight::Int = 61
    load_time_bounds::Tuple{Float64,Float64} = (0.05, 2.0)
    support_time_bounds::Tuple{Float64,Float64} = (0.05, 2.0)
    flight_time_bounds::Tuple{Float64,Float64} = (0.10, 4.0)
    front_point_bounds::Tuple{Float64,Float64} = (0.0, 0.40)
    front_point0::Float64 = 0.29      # initial front foot: over the front wheel
    rider_clearance::Float64 = 0.15   # rider COM at least this far above the deck line
    vx0::Float64 = 0.0                # common initial horizontal velocity (ice)
    force_rate_weight::Float64 = 1e-4
    objective::Symbol = :board_apex   # :board_apex | :rider_apex | :tail_apex | :lowest_point
    # leg-force direction: :normal = perpendicular to the deck (foot pushes into the deck,
    # rider gets the reaction); :leg = along the leg (foot -> rider).
    force_dir::Symbol = :leg          # ground phases (load + pop)
    flight_force_dir::Symbol = :friction # flight: :friction (normal into deck + tangential <= mu*normal) | :cone | :normal | :leg
    mu::Float64 = 0.0                    # foot-deck friction coefficient for :friction (0 = no dragging)
    force_flip::Union{Nothing,Float64} = nothing   # diagnostic: require th2[mid] <= this
    force_theta_mid::Union{Nothing,Float64} = nothing   # diagnostic: require th2[mid] == this
    warm_from_board_apex::Bool = true   # for :tail_apex, first maximize board apex, then switch
    feet_on_board::Bool = true          # keep both legs within reach (ℓ <= leg_max) throughout flight
    rider_above_feet::Bool = true       # rider COM world-y at least `foot_margin` above both foot points, all phases
    land_matched_velocity::Bool = false # touchdown: rider vx == board vx (ride away together; with vx0=0 both must be 0 by momentum)
    land_com_over_trucks::Bool = false  # touchdown: rider COM x between the rear and front wheel contacts (COP ahead of rear truck)
    foot_margin::Float64 = 0.2
    # Hill-type force–velocity limit on each leg actuator (Alexander 1990/1992), opt-in.
    # Leg extension rate v = dℓ/dt (nondim: length ℓ_max=0.9 m, time √(0.9/9.81)=0.303 s,
    # so 1 velocity unit ≈ 2.97 m/s).  F·(v_max + v/k) <= F_max·(v_max − v), v <= v_max.
    force_velocity::Bool = false
    fv_vmax::Float64 = 1.2            # max unloaded leg extension speed (≈ 3.6 m/s)
    fv_k::Float64 = 0.25              # Hill curvature a/F0 = b/v_max = 1/G, Alexander's G ≈ 4
    fv_eccentric_gain::Float64 = 1.0  # force cap while the leg is compressing (v < 0), × F_max
    print_level::Int = 0
    max_iter::Int = 6000
    T0_guess::Float64 = 0.3
    T1_guess::Float64 = 0.35
    T2_guess::Float64 = 1.0
    tol::Float64 = 1e-6
    mu_strategy::String = "monotone"
end

"Per-leg force limit in solver (board-weight) units: force_limit_bw · M · g unless overridden."
force_limit_value(opt::RiderOptions; g=1.0) = isnan(opt.force_limit) ? opt.force_limit_bw*opt.rider_mass*g : opt.force_limit

"""
Reference scales that turn nondimensional results into SI. Board mass follows from the
rider/board ratio in `RiderOptions` (70 kg / 25 = 2.8 kg, a real deck+trucks+wheels).
"""
Base.@kwdef struct RiderScales
    rider_mass_kg::Float64 = 77.1     # 170 lb
    leg_length_m::Float64 = 0.90      # ℓ_max = 1
    g_si::Float64 = 9.81
end
board_mass_kg(sc::RiderScales, opt::RiderOptions) = sc.rider_mass_kg/opt.rider_mass
"""
    dimensional(sc, opt) -> NamedTuple of multipliers

`length` (m per unit), `time` (s per unit, √(L/g)), `velocity` (m/s), `force` (N per
board weight), `force_bw` (rider body weights per board weight), `mass_board_kg`.
"""
function dimensional(sc::RiderScales, opt::RiderOptions)
    L = sc.leg_length_m; T = sqrt(L/sc.g_si); mb = board_mass_kg(sc, opt)
    (; length=L, time=T, velocity=L/T, force=mb*sc.g_si, force_bw=1/opt.rider_mass,
       mass_board_kg=mb, mass_rider_kg=sc.rider_mass_kg)
end

_X(th, r) = cos(th)*r[1] - sin(th)*r[2]
_Y(th, r) = sin(th)*r[1] + cos(th)*r[2]

function solve_rider_ollie(; p=ReducedBoardParams(), opt=RiderOptions(), staged=true)
    n0, n1, n2 = opt.n_load, opt.n_support, opt.n_flight
    mid = (n2+1) ÷ 2
    m, I, g, M = p.mass, p.inertia, p.gravity, opt.rider_mass
    rt = body_point(p,:tail); rc = body_point(p,:rear_slide); rf = body_point(p,:front_slide)
    Lmax, Lmin, Fmax = opt.leg_max, opt.leg_min, force_limit_value(opt; g=g)
    # variable bound on the force magnitudes; only the eccentric branch of the F–v law may exceed Fmax
    Fcap = opt.force_velocity ? opt.fv_eccentric_gain*Fmax : Fmax
    yflat = p.deck_height

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model,"print_level",opt.print_level)
    set_optimizer_attribute(model,"tol",opt.tol)
    set_optimizer_attribute(model,"mu_strategy",opt.mu_strategy)
    set_optimizer_attribute(model,"max_iter",opt.max_iter)

    @variable(model, opt.load_time_bounds[1]    <= T0 <= opt.load_time_bounds[2])
    @variable(model, opt.support_time_bounds[1] <= T1 <= opt.support_time_bounds[2])
    @variable(model, opt.flight_time_bounds[1]  <= T2 <= opt.flight_time_bounds[2])
    h0 = T0/(n0-1); h1 = T1/(n1-1); h2 = T2/(n2-1)

    # ---------------- variables per phase ----------------
    # phase 0: board x only (θ=0, y=yflat); rider full
    @variable(model, -3 <= xb0[1:n0] <= 3);  @variable(model, -6 <= vb0[1:n0] <= 6)
    @variable(model, -3 <= xr0[1:n0] <= 3);  @variable(model, 0 <= yr0[1:n0] <= 3)
    @variable(model, -6 <= vxr0[1:n0] <= 6); @variable(model, -6 <= vyr0[1:n0] <= 6)
    @variable(model, 0 <= Fb0[1:n0-1] <= Fcap); @variable(model, 0 <= Ff0[1:n0-1] <= Fcap)
    @variable(model, opt.front_point_bounds[1] <= s0[1:n0-1] <= opt.front_point_bounds[2])
    @variable(model, Rr0[1:n0-1] >= 0); @variable(model, Rf0[1:n0-1] >= 0)
    # phase 1: board reduced (x, θ); rider full
    @variable(model, -3 <= xb1[1:n1] <= 3);  @variable(model, -6 <= vb1[1:n1] <= 6)
    @variable(model, 0 <= th1[1:n1] <= 1.4); @variable(model, 0 <= om1[1:n1] <= 15)
    @variable(model, -3 <= xr1[1:n1] <= 3);  @variable(model, 0 <= yr1[1:n1] <= 3)
    @variable(model, -6 <= vxr1[1:n1] <= 6); @variable(model, -6 <= vyr1[1:n1] <= 6)
    @variable(model, 0 <= Fb1[1:n1-1] <= Fcap); @variable(model, 0 <= Ff1[1:n1-1] <= Fcap)
    @variable(model, opt.front_point_bounds[1] <= s1[1:n1-1] <= opt.front_point_bounds[2])
    @variable(model, Rr1[1:n1-1] >= 0)
    # phase 2: board full; rider full
    @variable(model, -3 <= xb2[1:n2] <= 3);  @variable(model, -0.2 <= yb2[1:n2] <= 3)
    @variable(model, -1.5 <= th2[1:n2] <= 1.5)
    @variable(model, -6 <= vb2[1:n2] <= 6);  @variable(model, -8 <= vyb2[1:n2] <= 8)
    @variable(model, -20 <= om2[1:n2] <= 20)
    @variable(model, -3 <= xr2[1:n2] <= 3);  @variable(model, 0 <= yr2[1:n2] <= 3)
    @variable(model, -6 <= vxr2[1:n2] <= 6); @variable(model, -6 <= vyr2[1:n2] <= 6)
    @variable(model, 0 <= Fb2[1:n2-1] <= Fcap); @variable(model, 0 <= Ff2[1:n2-1] <= Fcap)
    @variable(model, opt.front_point_bounds[1] <= s2[1:n2-1] <= opt.front_point_bounds[2])
    @variable(model, impulse >= 0)
    # flight-phase horizontal leg-force components on the rider (cone mode only)
    @variable(model, -Fcap <= Gbx[1:n2-1] <= Fcap); @variable(model, -Fcap <= Gfx[1:n2-1] <= Fcap)
    # cone: magnitude N bounds (Gx, F); N is what the reach complementarity gates
    @variable(model, 0 <= Nb2[1:n2-1] <= Fcap); @variable(model, 0 <= Nf2[1:n2-1] <= Fcap)
    if opt.flight_force_dir === :cone
        for k in 1:n2-1
            @constraint(model, Gbx[k]^2 + Fb2[k]^2 <= Nb2[k]^2); @constraint(model, Gfx[k]^2 + Ff2[k]^2 <= Nf2[k]^2)
        end
    elseif opt.flight_force_dir === :friction
        # Fb2/Ff2 = normal component on the rider (>=0, i.e. into the deck on the board); Gbx/Gfx = tangential along deck
        for k in 1:n2-1
            @constraint(model, Gbx[k] <= opt.mu*Fb2[k]); @constraint(model, -Gbx[k] <= opt.mu*Fb2[k])
            @constraint(model, Gfx[k] <= opt.mu*Ff2[k]); @constraint(model, -Gfx[k] <= opt.mu*Ff2[k])
            @constraint(model, Gbx[k]^2 + Fb2[k]^2 <= Nb2[k]^2); @constraint(model, Gfx[k]^2 + Ff2[k]^2 <= Nf2[k]^2)
        end
    else
        fix.(Gbx, 0.0; force=true); fix.(Gfx, 0.0; force=true)
    end

    # ---------------- leg helper ----------------
    # returns (ux, uy, ℓ) for a leg from world foot point (fx,fy) to rider (xr,yr)
    leg(fx,fy,xr,yr) = begin
        ℓ = sqrt((xr-fx)^2 + (yr-fy)^2 + 1e-8)
        ((xr-fx)/ℓ, (yr-fy)/ℓ, ℓ)
    end
    # force direction on the rider: deck normal (rotated by θ) or along the leg
    fdir(th, ux, uy; dir=opt.force_dir) = dir === :normal ? (-sin(th), cos(th)) : (ux, uy)
    legcons!(F, ℓ) = begin
        @constraint(model, F*(ℓ - Lmax) <= 0)     # push only when within reach
        @constraint(model, ℓ >= Lmin)
    end
    # rider must stay above the deck line (board frame)
    above!(xb,yb,th,xr,yr) = @constraint(model, -sin(th)*(xr-xb) + cos(th)*(yr-yb) >= opt.rider_clearance)
    # Hill force–velocity limit on the leg actuator (opt-in).  v = dℓ/dt = (v_rider − v_foot)·u is
    # the leg extension rate (muscle shortening).  Written multiplied out so there is no division:
    #   F·(v_max + v/k) <= F_max·(v_max − v)   and   v <= v_max.
    # For v < 0 (leg compressing, eccentric) the hyperbola exceeds F_max and the variable bound
    # Fcap = fv_eccentric_gain·F_max takes over; for v <= −k·v_max the inequality is vacuous.
    fv!(F, ux, uy, vxr, vyr, vfx, vfy) = begin
        opt.force_velocity || return nothing
        vmax, kk = opt.fv_vmax, opt.fv_k
        v = (vxr - vfx)*ux + (vyr - vfy)*uy
        @constraint(model, F*(vmax + v/kk) <= Fmax*(vmax - v))
        @constraint(model, v <= vmax)
        nothing
    end

    # ---------------- phase 0: flat board ----------------
    for k in 1:n0-1
        tx = xb0[k]+rt[1]; ty = yflat+rt[2]          # tail tip (θ=0)
        fx = xb0[k]+s0[k]; fy = yflat                # front foot on deck
        ubx,uby,ℓb = leg(tx,ty,xr0[k],yr0[k]); ufx,ufy,ℓf = leg(fx,fy,xr0[k],yr0[k])
        legcons!(Fb0[k],ℓb); legcons!(Ff0[k],ℓf)
        # foot velocities: tail moves with the board; front foot also slides along the deck at ds/dt
        sdot0 = k < n0-1 ? (s0[k+1]-s0[k])/h0 : 0.0
        fv!(Fb0[k],ubx,uby,vxr0[k],vyr0[k],vb0[k],0.0)
        fv!(Ff0[k],ufx,ufy,vxr0[k],vyr0[k],vb0[k]+sdot0,0.0)
        above!(xb0[k],yflat,0.0,xr0[k],yr0[k])
        opt.rider_above_feet && (@constraint(model, yr0[k] >= ty + opt.foot_margin); @constraint(model, yr0[k] >= fy + opt.foot_margin))
        ubx,uby = fdir(0.0,ubx,uby); ufx,ufy = fdir(0.0,ufx,ufy)
        # forces on board (opposite to leg forces)
        Bbx = -Fb0[k]*ubx; Bby = -Fb0[k]*uby; Bfx = -Ff0[k]*ufx; Bfy = -Ff0[k]*ufy
        # board statics (θ=0, y fixed): vertical and moment about COM; wheels at ±wheelbase/2
        @constraint(model, Rr0[k] + Rf0[k] + Bby + Bfy - m*g == 0)
        @constraint(model, rc[1]*Rr0[k] + rf[1]*Rf0[k] + (rt[1]*Bby - rt[2]*Bbx) + s0[k]*Bfy == 0)
        # board slides in x; rider dynamics
        @constraint(model, vb0[k+1] == vb0[k] + h0*(Bbx+Bfx)/m)
        @constraint(model, xb0[k+1] == xb0[k] + h0*vb0[k+1])
        @constraint(model, vxr0[k+1] == vxr0[k] + h0*(Fb0[k]*ubx + Ff0[k]*ufx)/M)
        @constraint(model, vyr0[k+1] == vyr0[k] + h0*((Fb0[k]*uby + Ff0[k]*ufy)/M - g))
        @constraint(model, xr0[k+1] == xr0[k] + h0*vxr0[k+1])
        @constraint(model, yr0[k+1] == yr0[k] + h0*vyr0[k+1])
    end
    # phase 0 exit: front wheel unloads
    @constraint(model, Rf0[end] == 0)
    # initial conditions
    @constraints(model, begin
        xb0[1] == 0; vb0[1] == opt.vx0; vxr0[1] == opt.vx0; vyr0[1] == 0
        s0[1] == opt.front_point0
        rt[1] <= xr0[1] <= opt.front_point0                  # COM between the feet
        Fb0[1] == Ff0[1]                                     # equal initial leg forces
    end)
    # equal initial leg lengths
    begin
        ubx,uby,ℓb = leg(rt[1], yflat+rt[2], xr0[1], yr0[1]); ufx,ufy,ℓf = leg(opt.front_point0, yflat, xr0[1], yr0[1])
        @constraint(model, ℓb == ℓf)
    end

    # ---------------- phase 1: reduced pop ----------------
    @constraints(model, begin
        xb1[1] == xb0[end]; vb1[1] == vb0[end]; th1[1] == 0; om1[1] == 0
        xr1[1] == xr0[end]; yr1[1] == yr0[end]; vxr1[1] == vxr0[end]; vyr1[1] == vyr0[end]
        s1[1] == s0[end]
    end)
    for k in 1:n1-1
        th = th1[k]; om = om1[k]
        Xc=_X(th,rc); Yc=_Y(th,rc); Xt=_X(th,rt); Yt=_Y(th,rt)
        yb = -Yc
        rp = (s1[k], 0.0); Xp = cos(th)*s1[k]; Yp = sin(th)*s1[k]
        ubx,uby,ℓb = leg(xb1[k]+Xt, yb+Yt, xr1[k], yr1[k])
        ufx,ufy,ℓf = leg(xb1[k]+Xp, yb+Yp, xr1[k], yr1[k])
        legcons!(Fb1[k],ℓb); legcons!(Ff1[k],ℓf)
        # foot velocities: board pivots about the rear wheel, v_COM = (vb, −Xc·ω); point r: + ω×r
        sdot1 = k < n1-1 ? (s1[k+1]-s1[k])/h1 : 0.0
        fv!(Fb1[k],ubx,uby,vxr1[k],vyr1[k], vb1[k] - om*Yt, -Xc*om + om*Xt)
        fv!(Ff1[k],ufx,ufy,vxr1[k],vyr1[k], vb1[k] - om*Yp + cos(th)*sdot1, -Xc*om + om*Xp + sin(th)*sdot1)
        above!(xb1[k],yb,th,xr1[k],yr1[k])
        opt.rider_above_feet && (@constraint(model, yr1[k] >= yb+Yt + opt.foot_margin); @constraint(model, yr1[k] >= yb+Yp + opt.foot_margin))
        ubx,uby = fdir(th,ubx,uby); ufx,ufy = fdir(th,ufx,ufy)
        Bbx = -Fb1[k]*ubx; Bby = -Fb1[k]*uby; Bfx = -Ff1[k]*ufx; Bfy = -Ff1[k]*ufy
        # virtual-work generalized forces (both components), see point_generalized_force
        Qx = Bbx + Bfx
        Qt = (-Yt*Bbx + (Xt-Xc)*Bby) + (-Yp*Bfx + (Xp-Xc)*Bfy)
        alpha = (Qt + m*g*Xc + m*Xc*Yc*om^2)/(I + m*Xc^2)
        @constraint(model, vb1[k+1] == vb1[k] + h1*Qx/m)
        @constraint(model, om1[k+1] == om + h1*alpha)
        @constraint(model, xb1[k+1] == xb1[k] + h1*vb1[k+1])
        @constraint(model, th1[k+1] == th + h1*om1[k+1])
        @constraint(model, Rr1[k] == m*(-Xc*alpha + Yc*om^2 + g) - Bby - Bfy)
        k > 1 && @constraint(model, -Yc + Yt >= 0)           # tail above ground
        @constraint(model, -Yc + _Y(th,rf) >= -1e-6)          # front wheel above ground
        @constraint(model, vxr1[k+1] == vxr1[k] + h1*(Fb1[k]*ubx + Ff1[k]*ufx)/M)
        @constraint(model, vyr1[k+1] == vyr1[k] + h1*((Fb1[k]*uby + Ff1[k]*ufy)/M - g))
        @constraint(model, xr1[k+1] == xr1[k] + h1*vxr1[k+1])
        @constraint(model, yr1[k+1] == yr1[k] + h1*vyr1[k+1])
    end
    the = th1[end]; Xce=_X(the,rc); Yce=_Y(the,rc); Xte=_X(the,rt); Yte=_Y(the,rt)
    @constraint(model, Yte - Yce == 0)
    @constraint(model, (Xte-Xce)*om1[end] <= -0.05)

    # ---------------- impact (board only) ----------------
    @constraints(model, begin
        xb2[1] == xb1[end]; yb2[1] == -Yce; th2[1] == the; vb2[1] == vb1[end]
        vyb2[1] == -Xce*om1[end] + impulse/m
        om2[1] == om1[end] + Xte*impulse/I
        vyb2[1] + Xte*om2[1] == -p.restitution*(Xte-Xce)*om1[end]
        xr2[1] == xr1[end]; yr2[1] == yr1[end]; vxr2[1] == vxr1[end]; vyr2[1] == vyr1[end]
        s2[1] == s1[end]
    end)

    # ---------------- phase 2: flight ----------------
    for k in 1:n2-1
        th = th2[k]
        Xt=_X(th,rt); Yt=_Y(th,rt); Xp = cos(th)*s2[k]; Yp = sin(th)*s2[k]
        ubx,uby,ℓb = leg(xb2[k]+Xt, yb2[k]+Yt, xr2[k], yr2[k])
        ufx,ufy,ℓf = leg(xb2[k]+Xp, yb2[k]+Yp, xr2[k], yr2[k])
        # flight foot velocities: board COM velocity + ω × r_foot (+ front-foot slide along the deck)
        sdot2 = k < n2-1 ? (s2[k+1]-s2[k])/h2 : 0.0
        vtx = vb2[k] - om2[k]*Yt; vty = vyb2[k] + om2[k]*Xt
        vpx = vb2[k] - om2[k]*Yp + cos(th)*sdot2; vpy = vyb2[k] + om2[k]*Xp + sin(th)*sdot2
        if opt.flight_force_dir in (:cone, :friction)
            legcons!(Nb2[k],ℓb); legcons!(Nf2[k],ℓf)
            fv!(Nb2[k],ubx,uby,vxr2[k],vyr2[k],vtx,vty); fv!(Nf2[k],ufx,ufy,vxr2[k],vyr2[k],vpx,vpy)
        else
            legcons!(Fb2[k],ℓb); legcons!(Ff2[k],ℓf)
            fv!(Fb2[k],ubx,uby,vxr2[k],vyr2[k],vtx,vty); fv!(Ff2[k],ufx,ufy,vxr2[k],vyr2[k],vpx,vpy)
        end
        if opt.feet_on_board
            @constraint(model, ℓb <= Lmax); @constraint(model, ℓf <= Lmax)
        end
        opt.rider_above_feet && (@constraint(model, yr2[k] >= yb2[k]+Yt + opt.foot_margin); @constraint(model, yr2[k] >= yb2[k]+Yp + opt.foot_margin))
        above!(xb2[k],yb2[k],th,xr2[k],yr2[k])
        if opt.flight_force_dir === :cone
            # force on rider = (Gx, F) with F >= 0 (upward on rider, downward on board); on board = negative
            Rbx = Gbx[k]; Rby = Fb2[k]; Rfx = Gfx[k]; Rfy = Ff2[k]
        elseif opt.flight_force_dir === :friction
            # on rider: F along deck normal n=(-sinθ,cosθ) plus G along deck tangent t=(cosθ,sinθ)
            Rbx = -sin(th)*Fb2[k] + cos(th)*Gbx[k]; Rby = cos(th)*Fb2[k] + sin(th)*Gbx[k]
            Rfx = -sin(th)*Ff2[k] + cos(th)*Gfx[k]; Rfy = cos(th)*Ff2[k] + sin(th)*Gfx[k]
        else
            ubx,uby = fdir(th,ubx,uby;dir=opt.flight_force_dir); ufx,ufy = fdir(th,ufx,ufy;dir=opt.flight_force_dir)
            Rbx = Fb2[k]*ubx; Rby = Fb2[k]*uby; Rfx = Ff2[k]*ufx; Rfy = Ff2[k]*ufy
        end
        Bbx = -Rbx; Bby = -Rby; Bfx = -Rfx; Bfy = -Rfy
        alpha = ((Xt*Bby - Yt*Bbx) + (Xp*Bfy - Yp*Bfx))/I
        @constraint(model, vb2[k+1]  == vb2[k]  + h2*(Bbx+Bfx)/m)
        @constraint(model, vyb2[k+1] == vyb2[k] + h2*((Bby+Bfy)/m - g))
        @constraint(model, om2[k+1]  == om2[k]  + h2*alpha)
        @constraint(model, xb2[k+1]  == xb2[k]  + h2*vb2[k+1])
        @constraint(model, yb2[k+1]  == yb2[k]  + h2*vyb2[k+1])
        @constraint(model, th2[k+1]  == th2[k]  + h2*om2[k+1])
        @constraint(model, vxr2[k+1] == vxr2[k] + h2*(Rbx + Rfx)/M)
        @constraint(model, vyr2[k+1] == vyr2[k] + h2*((Rby + Rfy)/M - g))
        @constraint(model, xr2[k+1]  == xr2[k]  + h2*vxr2[k+1])
        @constraint(model, yr2[k+1]  == yr2[k]  + h2*vyr2[k+1])
    end
    for k in 3:n2-1, r in (rf, rc, rt)
        @constraint(model, yb2[k] + _Y(th2[k],r) >= 0)
    end
    # apex node and landing
    if opt.objective === :board_apex
        @constraint(model, vyb2[mid] == 0)
        height = yb2[mid]
    elseif opt.objective === :tail_apex
        height = yb2[mid] + _Y(th2[mid],rt)      # no velocity pin: T2 is free, mid slides
    elseif opt.objective === :lowest_point
        @constraint(model, vyb2[mid] == 0)
        @variable(model, zlow)
        for r in (rt, [p.deck_length/2,0.0], rf, rc)
            @constraint(model, zlow <= yb2[mid] + _Y(th2[mid],r))
        end
        height = zlow
    else
        @constraint(model, vyr2[mid] == 0)
        height = yr2[mid]
    end
    flipcon = isnothing(opt.force_flip) ? nothing : @constraint(model, th2[mid] <= opt.force_flip)
    isnothing(opt.force_theta_mid) || @constraint(model, th2[mid] == opt.force_theta_mid)
    @constraints(model, begin
        th2[end] == 0; yb2[end] == yflat; om2[end] == 0; vyb2[end] <= -0.02
    end)
    # rider still on the board at touchdown: both legs within reach
    begin
        ubx,uby,ℓb = leg(xb2[end]+rt[1], yflat+rt[2], xr2[end], yr2[end])
        ufx,ufy,ℓf = leg(xb2[end]+s2[end], yflat, xr2[end], yr2[end])
        @constraint(model, ℓb <= Lmax); @constraint(model, ℓf <= Lmax)
        # ... and standing over the board: COM horizontally between the feet
        @constraint(model, xb2[end] + rt[1] <= xr2[end]); @constraint(model, xr2[end] <= xb2[end] + s2[end])
        opt.land_com_over_trucks && (@constraint(model, xb2[end] + rc[1] <= xr2[end]); @constraint(model, xr2[end] <= xb2[end] + rf[1]))
        opt.land_matched_velocity && @constraint(model, vxr2[end] == vb2[end])
    end

    rates = sum((Fb0[k+1]-Fb0[k])^2 + (Ff0[k+1]-Ff0[k])^2 + (s0[k+1]-s0[k])^2 for k in 1:n0-2) +
            sum((Fb1[k+1]-Fb1[k])^2 + (Ff1[k+1]-Ff1[k])^2 + (s1[k+1]-s1[k])^2 for k in 1:n1-2) +
            sum((Fb2[k+1]-Fb2[k])^2 + (Ff2[k+1]-Ff2[k])^2 + (s2[k+1]-s2[k])^2 + (Gbx[k+1]-Gbx[k])^2 + (Gfx[k+1]-Gfx[k])^2 for k in 1:n2-2)

    # ---------------- initial guess ----------------
    T0g,T1g,T2g = opt.T0_guess, opt.T1_guess, opt.T2_guess
    set_start_value(T0,T0g); set_start_value(T1,T1g); set_start_value(T2,T2g)
    xr_init = (rt[1]+opt.front_point0)/2
    # equal-leg rider height above flat board
    dxb = xr_init - rt[1]; dxf = opt.front_point0 - xr_init
    # solve (yr-0.14)^2 + dxb^2 = (yr-0.08)^2 + dxf^2
    yr_init = (dxf^2 - dxb^2 + 0.14^2 - 0.08^2)/(2*(0.14-0.08))
    yr_init = max(yr_init, yflat + 0.8)
    theta_hit = 0.905
    for k in 1:n0
        set_start_value(xb0[k],0.0); set_start_value(vb0[k],opt.vx0)
        set_start_value(xr0[k],xr_init); set_start_value(yr0[k],yr_init)
        set_start_value(vxr0[k],opt.vx0); set_start_value(vyr0[k],0.0)
    end
    for k in 1:n0-1
        tau=(k-1)/(n0-1)
        set_start_value(Fb0[k], M*g/2*(1+tau)); set_start_value(Ff0[k], M*g/2*(1-tau))
        set_start_value(s0[k],opt.front_point0); set_start_value(Rr0[k], m*g+M*g*(1+tau)/2); set_start_value(Rf0[k], M*g*(1-tau)/2)
    end
    for k in 1:n1
        tau=(k-1)/(n1-1)
        set_start_value(th1[k],theta_hit*tau^2); set_start_value(om1[k],2theta_hit*tau/T1g)
        set_start_value(xb1[k],0.0); set_start_value(vb1[k],opt.vx0)
        set_start_value(xr1[k],xr_init); set_start_value(yr1[k],yr_init + 0.3*tau)
        set_start_value(vxr1[k],opt.vx0); set_start_value(vyr1[k],0.3/T1g)
    end
    for k in 1:n1-1
        set_start_value(Fb1[k], Fmax*0.8); set_start_value(Ff1[k], 0.0); set_start_value(s1[k],opt.front_point0); set_start_value(Rr1[k], M*g)
    end
    yb_hit = 0.29*sin(theta_hit)+0.08*cos(theta_hit)
    for k in 1:n2
        tau=(k-1)/(n2-1)
        set_start_value(th2[k],theta_hit*(1-tau)); set_start_value(om2[k],-theta_hit/T2g)
        set_start_value(yb2[k], yflat + (yb_hit-yflat)*(1-tau) + 0.4*4*tau*(1-tau))
        set_start_value(vyb2[k], 1.6*(1-2tau)/T2g)
        set_start_value(xb2[k],0.0); set_start_value(vb2[k],opt.vx0)
        set_start_value(xr2[k],xr_init); set_start_value(yr2[k], yr_init + 0.3 + 0.4*4*tau*(1-tau))
        set_start_value(vxr2[k],opt.vx0); set_start_value(vyr2[k], 1.6*(1-2tau)/T2g)
    end
    for k in 1:n2-1
        set_start_value(Fb2[k],0.0); set_start_value(Ff2[k],2.0); set_start_value(s2[k],0.35)
    end
    set_start_value(impulse,1.0)

    if staged
        fix(T0,T0g;force=true); fix(T1,T1g;force=true); fix(T2,T2g;force=true)
        @objective(model, Min, rates)
        optimize!(model)
        println("  stage 1 (fixed times, feasibility): ", termination_status(model),
                "  theta_hit=", round(value(th1[end]),digits=3), " board apex=", round(value(yb2[mid]),digits=3))
        unfix(T0); unfix(T1); unfix(T2)
        set_lower_bound(T0,opt.load_time_bounds[1]); set_upper_bound(T0,opt.load_time_bounds[2])
        set_lower_bound(T1,opt.support_time_bounds[1]); set_upper_bound(T1,opt.support_time_bounds[2])
        set_lower_bound(T2,opt.flight_time_bounds[1]); set_upper_bound(T2,opt.flight_time_bounds[2])
        set_optimizer_attribute(model,"warm_start_init_point","yes")
        set_optimizer_attribute(model,"mu_init",1e-3)
    end
    if opt.objective in (:tail_apex,:lowest_point) && opt.warm_from_board_apex
        @objective(model, Max, yb2[mid] - opt.force_rate_weight*rates)
        optimize!(model)
        println("  stage 2 (board-apex warm start): ", termination_status(model), " board apex=", round(value(yb2[mid]),digits=3))
        set_optimizer_attribute(model,"warm_start_init_point","yes"); set_optimizer_attribute(model,"mu_init",1e-3)
    end
    @objective(model, Max, height - opt.force_rate_weight*rates)
    optimize!(model)
    return (;model,p,opt,mid,flipcon,T0,T1,T2,xb0,vb0,xr0,yr0,vxr0,vyr0,Fb0,Ff0,s0,Rr0,Rf0,
             xb1,vb1,th1,om1,xr1,yr1,vxr1,vyr1,Fb1,Ff1,s1,Rr1,
             xb2,yb2,th2,vb2,vyb2,om2,xr2,yr2,vxr2,vyr2,Fb2,Ff2,Gbx,Gfx,Nb2,Nf2,s2,impulse)
end

function rider_data(sol)
    v(x)=value.(x); p=sol.p; rc=body_point(p,:rear_slide)
    T0,T1,T2=value(sol.T0),value(sol.T1),value(sol.T2)
    t0=collect(range(0,T0,length=sol.opt.n_load)); t1=collect(range(T0,T0+T1,length=sol.opt.n_support))
    t2=collect(range(T0+T1,T0+T1+T2,length=sol.opt.n_flight))
    th1=v(sol.th1); yb1=[-_Y(t,rc) for t in th1]
    (; status=termination_status(sol.model), T0,T1,T2, t0,t1,t2,
       xb0=v(sol.xb0), yb0=fill(p.deck_height,sol.opt.n_load), th0=zeros(sol.opt.n_load),
       xr0=v(sol.xr0), yr0=v(sol.yr0), Fb0=v(sol.Fb0), Ff0=v(sol.Ff0), s0=v(sol.s0), Rr0=v(sol.Rr0), Rf0=v(sol.Rf0),
       xb1=v(sol.xb1), yb1, th1, om1=v(sol.om1), xr1=v(sol.xr1), yr1=v(sol.yr1), vxr1=v(sol.vxr1), vyr1=v(sol.vyr1),
       Fb1=v(sol.Fb1), Ff1=v(sol.Ff1), s1=v(sol.s1), Rr1=v(sol.Rr1), vb1=v(sol.vb1),
       xb2=v(sol.xb2), yb2=v(sol.yb2), th2=v(sol.th2), vb2=v(sol.vb2), vyb2=v(sol.vyb2), om2=v(sol.om2),
       xr2=v(sol.xr2), yr2=v(sol.yr2), vxr2=v(sol.vxr2), vyr2=v(sol.vyr2), Fb2=v(sol.Fb2), Ff2=v(sol.Ff2), Gbx=v(sol.Gbx), Gfx=v(sol.Gfx), s2=v(sol.s2),
       vb0=v(sol.vb0), vxr0=v(sol.vxr0), vyr0=v(sol.vyr0), impulse=value(sol.impulse), mid=sol.mid,
       Nb2=v(sol.Nb2), Nf2=v(sol.Nf2))
end

"""
    leg_extension_rates(sol) -> (; back, front, t)

Leg extension rate v = dℓ/dt = (v_rider − v_foot)·u at every force node of every phase, using
the same foot-velocity kinematics as the force–velocity constraint (front-foot slide included).
Positive = leg extending (muscle shortening).
"""
function leg_extension_rates(sol)
    d=rider_data(sol); p=sol.p; rt=body_point(p,:tail); rc=body_point(p,:rear_slide)
    u(fx,fy,xr,yr)=(l=hypot(xr-fx,yr-fy); ((xr-fx)/l,(yr-fy)/l))
    vb=Float64[]; vf=Float64[]; t=Float64[]
    n0=length(d.xb0); h0=d.T0/(n0-1)
    for k in 1:n0-1
        sd = k<n0-1 ? (d.s0[k+1]-d.s0[k])/h0 : 0.0
        ub=u(d.xb0[k]+rt[1],p.deck_height+rt[2],d.xr0[k],d.yr0[k]); uf=u(d.xb0[k]+d.s0[k],p.deck_height,d.xr0[k],d.yr0[k])
        push!(vb,(d.vxr0[k]-d.vb0[k])*ub[1]+d.vyr0[k]*ub[2]); push!(vf,(d.vxr0[k]-d.vb0[k]-sd)*uf[1]+d.vyr0[k]*uf[2]); push!(t,d.t0[k])
    end
    n1=length(d.th1); h1=d.T1/(n1-1)
    for k in 1:n1-1
        th=d.th1[k]; om=d.om1[k]; Xc=_X(th,rc); Xt=_X(th,rt); Yt=_Y(th,rt); Xp=cos(th)*d.s1[k]; Yp=sin(th)*d.s1[k]
        sd = k<n1-1 ? (d.s1[k+1]-d.s1[k])/h1 : 0.0
        ub=u(d.xb1[k]+Xt,d.yb1[k]+Yt,d.xr1[k],d.yr1[k]); uf=u(d.xb1[k]+Xp,d.yb1[k]+Yp,d.xr1[k],d.yr1[k])
        push!(vb,(d.vxr1[k]-(d.vb1[k]-om*Yt))*ub[1]+(d.vyr1[k]-(-Xc*om+om*Xt))*ub[2])
        push!(vf,(d.vxr1[k]-(d.vb1[k]-om*Yp+cos(th)*sd))*uf[1]+(d.vyr1[k]-(-Xc*om+om*Xp+sin(th)*sd))*uf[2]); push!(t,d.t1[k])
    end
    n2=length(d.yb2); h2=d.T2/(n2-1)
    for k in 1:n2-1
        th=d.th2[k]; om=d.om2[k]; Xt=_X(th,rt); Yt=_Y(th,rt); Xp=cos(th)*d.s2[k]; Yp=sin(th)*d.s2[k]
        sd = k<n2-1 ? (d.s2[k+1]-d.s2[k])/h2 : 0.0
        ub=u(d.xb2[k]+Xt,d.yb2[k]+Yt,d.xr2[k],d.yr2[k]); uf=u(d.xb2[k]+Xp,d.yb2[k]+Yp,d.xr2[k],d.yr2[k])
        push!(vb,(d.vxr2[k]-(d.vb2[k]-om*Yt))*ub[1]+(d.vyr2[k]-(d.vyb2[k]+om*Xt))*ub[2])
        push!(vf,(d.vxr2[k]-(d.vb2[k]-om*Yp+cos(th)*sd))*uf[1]+(d.vyr2[k]-(d.vyb2[k]+om*Xp+sin(th)*sd))*uf[2]); push!(t,d.t2[k])
    end
    (; back=vb, front=vf, t)
end

"Energy, momentum, impact and NE audits for the rider+board solution."
function audit_rider_ollie(sol)
    d=rider_data(sol); p=sol.p; m,I,g=p.mass,p.inertia,p.gravity; M=sol.opt.rider_mass
    rt=body_point(p,:tail); rc=body_point(p,:rear_slide)
    legℓ(fx,fy,xr,yr)=hypot(xr-fx,yr-fy)
    # work of leg forces = Σ F * Δℓ  (positive when the leg extends)
    W=0.0; resid=0.0
    E(yb,vb,vyb,om,yr,vxr,vyr)=0.5m*(vb^2+vyb^2)+0.5I*om^2+m*g*yb+0.5M*(vxr^2+vyr^2)+M*g*yr
    # phase 0
    n0=length(d.xb0); h0=d.T0/(n0-1)
    for k in 1:n0-1
        ℓb1=legℓ(d.xb0[k]+rt[1],p.deck_height+rt[2],d.xr0[k],d.yr0[k]); ℓb2=legℓ(d.xb0[k+1]+rt[1],p.deck_height+rt[2],d.xr0[k+1],d.yr0[k+1])
        ℓf1=legℓ(d.xb0[k]+d.s0[k],p.deck_height,d.xr0[k],d.yr0[k]); ℓf2=legℓ(d.xb0[k+1]+d.s0[k],p.deck_height,d.xr0[k+1],d.yr0[k+1])
        dW = sol.opt.force_dir === :normal ?
             d.Fb0[k]*((d.yr0[k+1]-d.yr0[k])) + d.Ff0[k]*((d.yr0[k+1]-d.yr0[k])) :   # θ=0: normal=(0,1); foot y fixed
             d.Fb0[k]*(ℓb2-ℓb1)+d.Ff0[k]*(ℓf2-ℓf1)
        dE=E(p.deck_height,d.vb0[k+1],0,0,d.yr0[k+1],d.vxr0[k+1],d.vyr0[k+1])-E(p.deck_height,d.vb0[k],0,0,d.yr0[k],d.vxr0[k],d.vyr0[k])
        W+=dW; resid+=dE-dW
    end
    # phase 1
    n1=length(d.th1); h1=d.T1/(n1-1); ne=0.0
    for k in 1:n1-1
        th=d.th1[k]; om=d.om1[k]; Xc=_X(th,rc); Yc=_Y(th,rc); Xt=_X(th,rt); Yt=_Y(th,rt)
        Xp=cos(th)*d.s1[k]; Yp=sin(th)*d.s1[k]
        ℓb1=legℓ(d.xb1[k]+Xt,d.yb1[k]+Yt,d.xr1[k],d.yr1[k]); ℓf1=legℓ(d.xb1[k]+Xp,d.yb1[k]+Yp,d.xr1[k],d.yr1[k])
        th2=d.th1[k+1]; Xt2=_X(th2,rt); Yt2=_Y(th2,rt); Xp2=cos(th2)*d.s1[k]; Yp2=sin(th2)*d.s1[k]
        ℓb2=legℓ(d.xb1[k+1]+Xt2,d.yb1[k+1]+Yt2,d.xr1[k+1],d.yr1[k+1]); ℓf2=legℓ(d.xb1[k+1]+Xp2,d.yb1[k+1]+Yp2,d.xr1[k+1],d.yr1[k+1])
        if sol.opt.force_dir === :normal
            n=(-sin(th),cos(th))
            drb=(d.xr1[k+1]-d.xr1[k]-(d.xb1[k+1]+Xt2-d.xb1[k]-Xt), d.yr1[k+1]-d.yr1[k]-(d.yb1[k+1]+Yt2-d.yb1[k]-Yt))
            drf=(d.xr1[k+1]-d.xr1[k]-(d.xb1[k+1]+Xp2-d.xb1[k]-Xp), d.yr1[k+1]-d.yr1[k]-(d.yb1[k+1]+Yp2-d.yb1[k]-Yp))
            dW=d.Fb1[k]*(n[1]*drb[1]+n[2]*drb[2]) + d.Ff1[k]*(n[1]*drf[1]+n[2]*drf[2])
        else
            dW=d.Fb1[k]*(ℓb2-ℓb1)+d.Ff1[k]*(ℓf2-ℓf1)
        end
        vyb1=-Xc*om; vyb2=-_X(th2,rc)*d.om1[k+1]
        dE=E(d.yb1[k+1],d.vb1[k+1],vyb2,d.om1[k+1],d.yr1[k+1],d.vxr1[k+1],d.vyr1[k+1])-E(d.yb1[k],d.vb1[k],vyb1,om,d.yr1[k],d.vxr1[k],d.vyr1[k])
        W+=dW; resid+=dE-dW
        # full-coordinate angular NE check with reconstructed reaction
        ub=[d.xr1[k]-(d.xb1[k]+Xt), d.yr1[k]-(d.yb1[k]+Yt)]; ub/=norm(ub)
        uf=[d.xr1[k]-(d.xb1[k]+Xp), d.yr1[k]-(d.yb1[k]+Yp)]; uf/=norm(uf)
        if sol.opt.force_dir === :normal; ub=[-sin(th),cos(th)]; uf=copy(ub); end
        Bb=-d.Fb1[k]*ub; Bf=-d.Ff1[k]*uf
        alpha=(d.om1[k+1]-om)/h1
        ne=max(ne, abs(I*alpha - ((Xt*Bb[2]-Yt*Bb[1]) + (Xp*Bf[2]-Yp*Bf[1]) + Xc*d.Rr1[k])))
    end
    q,vm=expand_preimpact_state([d.xb1[end],d.th1[end]],[d.vb1[end],d.om1[end]],p)
    reset=tail_impact_reset(q,vm,p)
    K(v)=0.5m*(v[1]^2+v[2]^2)+0.5I*v[3]^2
    impact_dE=K(reset.vplus)-K(vm)
    # phase 2
    n2=length(d.yb2); h2=d.T2/(n2-1); resid2=0.0; W2=0.0
    for k in 1:n2-1
        th=d.th2[k]; th2=d.th2[k+1]
        ℓb1=legℓ(d.xb2[k]+_X(th,rt),d.yb2[k]+_Y(th,rt),d.xr2[k],d.yr2[k]); ℓb2=legℓ(d.xb2[k+1]+_X(th2,rt),d.yb2[k+1]+_Y(th2,rt),d.xr2[k+1],d.yr2[k+1])
        ℓf1=legℓ(d.xb2[k]+cos(th)*d.s2[k],d.yb2[k]+sin(th)*d.s2[k],d.xr2[k],d.yr2[k]); ℓf2=legℓ(d.xb2[k+1]+cos(th2)*d.s2[k],d.yb2[k+1]+sin(th2)*d.s2[k],d.xr2[k+1],d.yr2[k+1])
        if sol.opt.flight_force_dir in (:normal, :cone, :friction)
            n=(-sin(th),cos(th)); tg=(cos(th),sin(th))
            tb1=(d.xb2[k]+_X(th,rt),d.yb2[k]+_Y(th,rt)); tb2=(d.xb2[k+1]+_X(th2,rt),d.yb2[k+1]+_Y(th2,rt))
            tf1=(d.xb2[k]+cos(th)*d.s2[k],d.yb2[k]+sin(th)*d.s2[k]); tf2=(d.xb2[k+1]+cos(th2)*d.s2[k],d.yb2[k+1]+sin(th2)*d.s2[k])
            drb=(d.xr2[k+1]-d.xr2[k]-(tb2[1]-tb1[1]), d.yr2[k+1]-d.yr2[k]-(tb2[2]-tb1[2]))
            drf=(d.xr2[k+1]-d.xr2[k]-(tf2[1]-tf1[1]), d.yr2[k+1]-d.yr2[k]-(tf2[2]-tf1[2]))
            if sol.opt.flight_force_dir === :cone
                dW=d.Gbx[k]*drb[1]+d.Fb2[k]*drb[2] + d.Gfx[k]*drf[1]+d.Ff2[k]*drf[2]
            elseif sol.opt.flight_force_dir === :friction
                Rb=(n[1]*d.Fb2[k]+tg[1]*d.Gbx[k], n[2]*d.Fb2[k]+tg[2]*d.Gbx[k]); Rf=(n[1]*d.Ff2[k]+tg[1]*d.Gfx[k], n[2]*d.Ff2[k]+tg[2]*d.Gfx[k])
                dW=Rb[1]*drb[1]+Rb[2]*drb[2]+Rf[1]*drf[1]+Rf[2]*drf[2]
            else
                dW=d.Fb2[k]*(n[1]*drb[1]+n[2]*drb[2]) + d.Ff2[k]*(n[1]*drf[1]+n[2]*drf[2])
            end
        else
            dW=d.Fb2[k]*(ℓb2-ℓb1)+d.Ff2[k]*(ℓf2-ℓf1)
        end
        dE=E(d.yb2[k+1],d.vb2[k+1],d.vyb2[k+1],d.om2[k+1],d.yr2[k+1],d.vxr2[k+1],d.vyr2[k+1])-E(d.yb2[k],d.vb2[k],d.vyb2[k],d.om2[k],d.yr2[k],d.vxr2[k],d.vyr2[k])
        W2+=dW; resid2+=dE-dW
    end
    px=vcat(m*d.vb0+M*d.vxr0, m*d.vb1+M*d.vxr1, m*d.vb2+M*d.vxr2)
    # largest violation of ANY model constraint / bound at the returned point (JuMP feasibility report)
    maxviol = try
        rep = primal_feasibility_report(sol.model; atol=0.0, skip_missing=true)
        isempty(rep) ? 0.0 : maximum(values(rep))
    catch e
        @warn "primal_feasibility_report failed" e; NaN
    end
    gaps=minimum(vcat([d.yb2[k]+_Y(d.th2[k],r) for k in 2:n2-1 for r in (rt,rc,body_point(p,:front_slide))]))
    (; status=d.status, force_dir=sol.opt.force_dir, flight_force_dir=sol.opt.flight_force_dir, mu=sol.opt.mu, T0=d.T0, T1=d.T1, T2=d.T2, theta_hit=d.th1[end],
       board_apex_rise=maximum(d.yb2)-p.deck_height, lowest_point_apex=maximum([minimum(d.yb2[k] .+ [_Y(d.th2[k],r) for r in (rt,[p.deck_length/2,0.0],body_point(p,:front_slide),rc)]) for k in eachindex(d.yb2)]), theta_at_mid=d.th2[d.mid], tail_apex_rise=maximum(d.yb2 .+ [_Y(t,rt) for t in d.th2])-(p.deck_height+p.tail_rise), theta_range_flight=extrema(d.th2), rider_apex=maximum(vcat(d.yr0,d.yr1,d.yr2)), rider_rise=maximum(vcat(d.yr0,d.yr1,d.yr2))-d.yr0[1],
       impulse=d.impulse, reset_impulse=reset.impulse, impact_velocity_error=norm([d.vb2[1],d.vyb2[1],d.om2[1]]-reset.vplus),
       impact_energy_change=impact_dE, angular_NE_residual=ne,
       min_reaction=min(minimum(d.Rr0),minimum(d.Rf0),minimum(d.Rr1)), front_wheel_reaction_end=d.Rf0[end],
       leg_work_support=W, energy_residual_support=resid, leg_work_flight=W2, energy_residual_flight=resid2,
       energy_residual_total=resid+resid2, energy_residual_rel=(resid+resid2)/max(abs(W)+abs(W2),1e-12),
       max_constraint_violation=maxviol,
       horizontal_momentum_drift=maximum(px)-minimum(px), min_gap=gaps,
       landing_angle=d.th2[end], landing_vy_board=d.vyb2[end], landing_vy_rider=d.vyr2[end], landing_vx_board=d.vb2[end], landing_vx_rider=d.vxr2[end], landing_com_x_rel=d.xr2[end]-d.xb2[end],
       initial_forces=(d.Fb0[1],d.Ff0[1]), peak_force=maximum(vcat(d.Fb0,d.Ff0,d.Fb1,d.Ff1,d.Fb2,d.Ff2)),
       max_leg_extension_rate=(ext=leg_extension_rates(sol); max(maximum(ext.back),maximum(ext.front))),
       force_velocity=sol.opt.force_velocity,
       front_point_range=extrema(vcat(d.s0,d.s1,d.s2)))
end
end # module
