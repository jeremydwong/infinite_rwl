# =============================================================================
# ollie_rider_3D.jl — rider + board ollie / kickflip in 3D
#
# COORDINATE CONVENTION (world, right-handed, Z UP):
#   X : lateral  (perpendicular to the plane of the 2-D model; "depth")   NEW
#   Y : forward  (direction of travel)                  [was `x` in ollie_rider.jl]
#   Z : vertical, up; gravity = −Z                      [was `y` in ollie_rider.jl]
# Board body frame (origin at the board COM, deck mid-plane):
#   e1 : lateral across the deck (parallel to the axles)
#   e2 : along the deck, nose positive
#   e3 : deck normal, up when the board is flat
# Orientation R = Rz(ψ)·Rx(θ)·Ry(φ)  (world = R · body):
#   ψ  yaw   about world Z            (kept small; free in flight)
#   θ  pitch about the axle axis      (nose-up positive: the ollie rotation)
#   φ  roll  about the board long axis (the KICKFLIP rotation; +2π = one flip)
# Body angular velocity ω = (w1, w2, w3) about (e1, e2, e3); Euler's equations with
#   I1 = pitch inertia, I2 = roll inertia (small), I3 = yaw inertia.
# Variable names: board xb,yb,zb / vxb,vyb,vzb / th,ph,ps / w1,w2,w3;
#                 rider xr,yr,zr / vxr,vyr,vzr.  Suffix 0/1/2 = phase load/pop/flight.
# Front foot on the deck at body point (σ, s, 0): s along the deck, σ lateral (the
# lateral offset is what produces roll torque: τ_roll = σ·N for a normal push N).
# Back foot at the tail tip (0, −deck_length/2, tail_rise).
#
# Nondimensional units as in ollie_rider.jl: board mass 1, g 1, length ℓ_max = 0.9 m.
# Phases: 0 load (flat board sliding on ice, planar + lateral slide), 1 pop (pivot on
# the rear axle; roll & yaw locked by the two rear wheels), tail-strike impact (vertical
# impulse at the tail centre, e = restitution), 2 flight (full 3-D rigid body).
# =============================================================================
module OllieRider3D

using JuMP, Ipopt, LinearAlgebra
include("ollie_reduced_check.jl")
using .OllieReducedCheck

export Rider3DOptions, Board3D, solve_rider_3d, rider3d_data, audit_rider_3d, ground_solution

Base.@kwdef struct Board3D
    mass::Float64 = 1.0
    I_pitch::Float64 = 0.07      # about e1 (axles)          — report Appendix 1
    I_roll::Float64 = 0.008      # about e2 (long axis)      — deck strip + trucks/wheels, see report
    I_yaw::Float64 = 0.075       # about e3 (normal) ≈ I_pitch + I_roll
    gravity::Float64 = 1.0
    deck_length::Float64 = 0.80
    deck_width::Float64 = 0.235  # 8.45" / 0.9 m
    wheelbase::Float64 = 0.58
    axle_half::Float64 = 0.12    # lateral wheel offset
    deck_height::Float64 = 0.08
    tail_rise::Float64 = 0.06
    restitution::Float64 = 0.18
end
planar(b::Board3D) = ReducedBoardParams(mass=b.mass, inertia=b.I_pitch, gravity=b.gravity, deck_length=b.deck_length,
                                        wheelbase=b.wheelbase, deck_height=b.deck_height, tail_rise=b.tail_rise, restitution=b.restitution)

Base.@kwdef struct Rider3DOptions
    rider_mass::Float64 = 34.0
    leg_max::Float64 = 1.0
    leg_min::Float64 = 0.25
    force_limit_bw::Float64 = 3.0        # running-peak scale (was 2)
    n_load::Int = 21
    n_support::Int = 31
    n_flight::Int = 81
    load_time_bounds::Tuple{Float64,Float64} = (0.05, 2.0)
    support_time_bounds::Tuple{Float64,Float64} = (0.05, 2.0)
    flight_time_bounds::Tuple{Float64,Float64} = (0.10, 4.0)
    front_point_bounds::Tuple{Float64,Float64} = (0.0, 0.40)
    front_point0::Float64 = 0.29
    rider_clearance::Float64 = 0.15
    foot_margin::Float64 = 0.2
    foot_slide_max::Float64 = 1.5      # max front-foot sliding speed on the deck |ds/dt|, |dσ/dt| (≈3 m/s); Inf = free
    vy0::Float64 = 0.63                # initial forward rolling speed
    mu::Float64 = 0.0                  # foot–deck friction in flight (0 = normal pushes only)
    force_rate_weight::Float64 = 1e-4
    objective::Symbol = :lowest_point  # :board_apex | :lowest_point | :feasible
    flip_turns::Float64 = 0.0          # required roll at landing in turns (1 = kickflip)
    flip_sign::Float64 = 1.0
    land_roll_rate_zero::Bool = true   # catch the board: w2(end) == 0
    land_yaw_max::Float64 = 0.1        # |ψ(end)| ≤ this: land square (gyroscopic roll–pitch coupling otherwise yaws the board)
    land_matched_velocity::Bool = true
    land_com_over_trucks::Bool = true
    lateral_free::Bool = true          # allow lateral (X) motion of rider/board and σ ≠ 0
    lateral_ground_locked::Bool = true # ground phases planar (feet on the centreline give no lateral force; removes flat directions)
    yaw_free::Bool = true              # yaw DOF in flight (else ψ ≡ 0 with an assumed constraint torque)
    print_level::Int = 0
    max_iter::Int = 10000
    tol::Float64 = 1e-4
    T0_guess::Float64 = 0.3
    T1_guess::Float64 = 0.35
    T2_guess::Float64 = 1.0
    mu_strategy::String = "monotone"
end

force_limit_value(o::Rider3DOptions; g=1.0) = o.force_limit_bw*o.rider_mass*g

# planar helpers (pitch only): forward and up components of body vector r=(long, normal)
_F(th, r) = cos(th)*r[1] - sin(th)*r[2]     # forward (Y)
_U(th, r) = sin(th)*r[1] + cos(th)*r[2]     # up (Z)

"Rotation matrix entries of R = Rz(ψ)Rx(θ)Ry(φ) as a 3×3 of expressions."
function rotmat(ps, th, ph)
    cψ,sψ,cθ,sθ,cφ,sφ = cos(ps),sin(ps),cos(th),sin(th),cos(ph),sin(ph)
    [cψ*cφ - sψ*sθ*sφ   -sψ*cθ   cψ*sφ + sψ*sθ*cφ;
     sψ*cφ + cψ*sθ*sφ    cψ*cθ   sψ*sφ - cψ*sθ*cφ;
    -cθ*sφ               sθ      cθ*cφ]
end

"""
    solve_rider_3d(; b, opt, ground=nothing, roll_targets=nothing)

`ground`: a NamedTuple from `ground_solution(sol)` — fixes phases 0, 1 and the impact
to that ollie (Jeremy's staged kickflip: keep the pop, optimise the flight).
`roll_targets`: continuation sequence of roll targets (turns) solved in order before the
final one (e.g. [0.25, 0.5, 1.0]); each stage warm-starts the next.
"""
function solve_rider_3d(; b=Board3D(), opt=Rider3DOptions(), ground=nothing, roll_targets=nothing, staged=true, warm=nothing)
    n0, n1, n2 = opt.n_load, opt.n_support, opt.n_flight
    mid = (n2+1) ÷ 2
    m, g, M = b.mass, b.gravity, opt.rider_mass
    I1, I2, I3 = b.I_pitch, b.I_roll, b.I_yaw
    p = planar(b)
    rt = body_point(p,:tail); rc = body_point(p,:rear_slide); rf = body_point(p,:front_slide)   # (long, normal)
    Lmax, Lmin, Fmax = opt.leg_max, opt.leg_min, force_limit_value(opt; g=g)
    zflat = b.deck_height
    wl = b.axle_half; halfw = b.deck_width/2
    latb = opt.lateral_free ? 3.0 : 0.0
    sigb = opt.lateral_free ? halfw : 0.0

    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model,"print_level",opt.print_level)
    set_optimizer_attribute(model,"tol",opt.tol)
    set_optimizer_attribute(model,"mu_strategy",opt.mu_strategy)
    set_optimizer_attribute(model,"max_iter",opt.max_iter)

    @variable(model, opt.load_time_bounds[1]    <= T0 <= opt.load_time_bounds[2])
    @variable(model, opt.support_time_bounds[1] <= T1 <= opt.support_time_bounds[2])
    @variable(model, opt.flight_time_bounds[1]  <= T2 <= opt.flight_time_bounds[2])
    h0 = T0/(n0-1); h1 = T1/(n1-1); h2 = T2/(n2-1)

    # ---------------- phase 0: flat board (X,Y slide; θ=φ=ψ=0) ----------------
    @variable(model, -latb <= xb0[1:n0] <= latb); @variable(model, -3 <= yb0[1:n0] <= 3)
    @variable(model, -6 <= vxb0[1:n0] <= 6);      @variable(model, -6 <= vyb0[1:n0] <= 6)
    @variable(model, -latb <= xr0[1:n0] <= latb); @variable(model, -3 <= yr0[1:n0] <= 3); @variable(model, 0 <= zr0[1:n0] <= 3)
    @variable(model, -6 <= vxr0[1:n0] <= 6); @variable(model, -6 <= vyr0[1:n0] <= 6); @variable(model, -6 <= vzr0[1:n0] <= 6)
    @variable(model, 0 <= Fb0[1:n0-1] <= Fmax); @variable(model, 0 <= Ff0[1:n0-1] <= Fmax)
    @variable(model, opt.front_point_bounds[1] <= s0[1:n0-1] <= opt.front_point_bounds[2])
    @variable(model, Rr0[1:n0-1] >= 0); @variable(model, Rf0[1:n0-1] >= 0)
    # ---------------- phase 1: pop, pivot on rear axle (θ; X,Y slide) ----------------
    @variable(model, -latb <= xb1[1:n1] <= latb); @variable(model, -3 <= yb1[1:n1] <= 3)
    @variable(model, -6 <= vxb1[1:n1] <= 6);      @variable(model, -6 <= vyb1[1:n1] <= 6)
    @variable(model, 0 <= th1[1:n1] <= 1.4); @variable(model, 0 <= om1[1:n1] <= 15)
    @variable(model, -latb <= xr1[1:n1] <= latb); @variable(model, -3 <= yr1[1:n1] <= 3); @variable(model, 0 <= zr1[1:n1] <= 3)
    @variable(model, -6 <= vxr1[1:n1] <= 6); @variable(model, -6 <= vyr1[1:n1] <= 6); @variable(model, -6 <= vzr1[1:n1] <= 6)
    @variable(model, 0 <= Fb1[1:n1-1] <= Fmax); @variable(model, 0 <= Ff1[1:n1-1] <= Fmax)
    @variable(model, opt.front_point_bounds[1] <= s1[1:n1-1] <= opt.front_point_bounds[2])
    @variable(model, Rr1[1:n1-1] >= 0)
    # ---------------- phase 2: flight, full 3-D ----------------
    @variable(model, -latb <= xb2[1:n2] <= latb); @variable(model, -3 <= yb2[1:n2] <= 3); @variable(model, -0.2 <= zb2[1:n2] <= 3)
    @variable(model, -6 <= vxb2[1:n2] <= 6); @variable(model, -6 <= vyb2[1:n2] <= 6); @variable(model, -8 <= vzb2[1:n2] <= 8)
    @variable(model, -1.5 <= th2[1:n2] <= 1.5)
    phmax = 2π*abs(opt.flip_turns) + 0.5
    @variable(model, -phmax <= ph2[1:n2] <= phmax)
    psb = opt.yaw_free ? 0.6 : 0.0
    @variable(model, -psb <= ps2[1:n2] <= psb)
    @variable(model, -20 <= w1[1:n2] <= 20); @variable(model, -40 <= w2[1:n2] <= 40); @variable(model, -20 <= w3[1:n2] <= 20)
    @variable(model, -latb <= xr2[1:n2] <= latb); @variable(model, -3 <= yr2[1:n2] <= 3); @variable(model, 0 <= zr2[1:n2] <= 3)
    @variable(model, -6 <= vxr2[1:n2] <= 6); @variable(model, -6 <= vyr2[1:n2] <= 6); @variable(model, -6 <= vzr2[1:n2] <= 6)
    @variable(model, 0 <= Nb2[1:n2-1] <= Fmax); @variable(model, 0 <= Nf2[1:n2-1] <= Fmax)       # normal push magnitudes
    @variable(model, -Fmax <= Gb1[1:n2-1] <= Fmax); @variable(model, -Fmax <= Gb2[1:n2-1] <= Fmax) # tangential (e1 lateral, e2 long)
    @variable(model, -Fmax <= Gf1[1:n2-1] <= Fmax); @variable(model, -Fmax <= Gf2[1:n2-1] <= Fmax)
    @variable(model, opt.front_point_bounds[1] <= s2[1:n2-1] <= opt.front_point_bounds[2])
    @variable(model, -sigb <= sg2[1:n2-1] <= sigb)                                              # front foot lateral offset σ
    @variable(model, impulse >= 0)
    if opt.mu > 0
        for k in 1:n2-1
            @constraint(model, Gb1[k]^2 + Gb2[k]^2 <= (opt.mu*Nb2[k])^2)
            @constraint(model, Gf1[k]^2 + Gf2[k]^2 <= (opt.mu*Nf2[k])^2)
        end
    else
        fix.(Gb1, 0.0; force=true); fix.(Gb2, 0.0; force=true); fix.(Gf1, 0.0; force=true); fix.(Gf2, 0.0; force=true)
    end
    opt.lateral_free || (fix.(sg2, 0.0; force=true))
    if isfinite(opt.foot_slide_max)
        vs = opt.foot_slide_max
        for k in 1:n0-2; @constraint(model, s0[k+1]-s0[k] <= vs*h0); @constraint(model, s0[k+1]-s0[k] >= -vs*h0); end
        for k in 1:n1-2; @constraint(model, s1[k+1]-s1[k] <= vs*h1); @constraint(model, s1[k+1]-s1[k] >= -vs*h1); end
        for k in 1:n2-2; @constraint(model, s2[k+1]-s2[k] <= vs*h2); @constraint(model, s2[k+1]-s2[k] >= -vs*h2); @constraint(model, sg2[k+1]-sg2[k] <= vs*h2); @constraint(model, sg2[k+1]-sg2[k] >= -vs*h2); end
    end
    if opt.lateral_ground_locked
        for v in (xb0, xr0, vxb0, vxr0, xb1, xr1, vxb1, vxr1); fix.(v, 0.0; force=true); end
    end

    # ---------------- helpers ----------------
    leg3(fx,fy,fz,xr,yr,zr) = begin
        ℓ = sqrt((xr-fx)^2 + (yr-fy)^2 + (zr-fz)^2 + 1e-8)
        ((xr-fx)/ℓ, (yr-fy)/ℓ, (zr-fz)/ℓ, ℓ)
    end
    legcons!(F, ℓ) = (@constraint(model, F*(ℓ - Lmax) <= 0); @constraint(model, ℓ >= Lmin))
    # rider above the deck plane (pitch only, ground phases): normal n = (0, −sinθ, cosθ)
    above!(yb,zb,th,yr,zr) = @constraint(model, -sin(th)*(yr-yb) + cos(th)*(zr-zb) >= opt.rider_clearance)

    # ---------------- phase 0 ----------------
    for k in 1:n0-1
        ty = yb0[k]+rt[1]; tz = zflat+rt[2]; tx = xb0[k]
        fy = yb0[k]+s0[k]; fz = zflat; fx = xb0[k]
        ubx,uby,ubz,ℓb = leg3(tx,ty,tz,xr0[k],yr0[k],zr0[k]); ufx,ufy,ufz,ℓf = leg3(fx,fy,fz,xr0[k],yr0[k],zr0[k])
        legcons!(Fb0[k],ℓb); legcons!(Ff0[k],ℓf)
        above!(yb0[k],zflat,0.0,yr0[k],zr0[k])
        @constraint(model, zr0[k] >= tz + opt.foot_margin); @constraint(model, zr0[k] >= fz + opt.foot_margin)
        Bbx = -Fb0[k]*ubx; Bby = -Fb0[k]*uby; Bbz = -Fb0[k]*ubz; Bfx = -Ff0[k]*ufx; Bfy = -Ff0[k]*ufy; Bfz = -Ff0[k]*ufz
        # board statics: vertical and pitch moment about COM; wheels at ±wheelbase/2 (roll moment from lateral
        # foot offsets is zero here: feet on the centreline in the ground phases)
        @constraint(model, Rr0[k] + Rf0[k] + Bbz + Bfz - m*g == 0)
        @constraint(model, rc[1]*Rr0[k] + rf[1]*Rf0[k] + (rt[1]*Bbz - rt[2]*Bby) + s0[k]*Bfz == 0)
        @constraint(model, vxb0[k+1] == vxb0[k] + h0*(Bbx+Bfx)/m)
        @constraint(model, vyb0[k+1] == vyb0[k] + h0*(Bby+Bfy)/m)
        @constraint(model, xb0[k+1] == xb0[k] + h0*vxb0[k+1]); @constraint(model, yb0[k+1] == yb0[k] + h0*vyb0[k+1])
        @constraint(model, vxr0[k+1] == vxr0[k] + h0*(Fb0[k]*ubx + Ff0[k]*ufx)/M)
        @constraint(model, vyr0[k+1] == vyr0[k] + h0*(Fb0[k]*uby + Ff0[k]*ufy)/M)
        @constraint(model, vzr0[k+1] == vzr0[k] + h0*((Fb0[k]*ubz + Ff0[k]*ufz)/M - g))
        @constraint(model, xr0[k+1] == xr0[k] + h0*vxr0[k+1]); @constraint(model, yr0[k+1] == yr0[k] + h0*vyr0[k+1]); @constraint(model, zr0[k+1] == zr0[k] + h0*vzr0[k+1])
    end
    @constraint(model, Rf0[end] == 0)
    @constraints(model, begin
        xb0[1] == 0; yb0[1] == 0; vxb0[1] == 0; vyb0[1] == opt.vy0
        xr0[1] == 0; vxr0[1] == 0; vyr0[1] == opt.vy0; vzr0[1] == 0
        s0[1] == opt.front_point0
        rt[1] <= yr0[1] <= opt.front_point0
        Fb0[1] == Ff0[1]
    end)
    begin
        _,_,_,ℓb = leg3(0.0, rt[1], zflat+rt[2], xr0[1], yr0[1], zr0[1]); _,_,_,ℓf = leg3(0.0, opt.front_point0, zflat, xr0[1], yr0[1], zr0[1])
        @constraint(model, ℓb == ℓf)
    end

    # ---------------- phase 1 ----------------
    @constraints(model, begin
        xb1[1] == xb0[end]; yb1[1] == yb0[end]; vxb1[1] == vxb0[end]; vyb1[1] == vyb0[end]; th1[1] == 0; om1[1] == 0
        xr1[1] == xr0[end]; yr1[1] == yr0[end]; zr1[1] == zr0[end]; vxr1[1] == vxr0[end]; vyr1[1] == vyr0[end]; vzr1[1] == vzr0[end]
        s1[1] == s0[end]
    end)
    for k in 1:n1-1
        th = th1[k]; om = om1[k]
        Fc=_F(th,rc); Uc=_U(th,rc); Ft=_F(th,rt); Ut=_U(th,rt)
        zb = -Uc
        Fp = cos(th)*s1[k]; Up = sin(th)*s1[k]
        ubx,uby,ubz,ℓb = leg3(xb1[k], yb1[k]+Ft, zb+Ut, xr1[k], yr1[k], zr1[k])
        ufx,ufy,ufz,ℓf = leg3(xb1[k], yb1[k]+Fp, zb+Up, xr1[k], yr1[k], zr1[k])
        legcons!(Fb1[k],ℓb); legcons!(Ff1[k],ℓf)
        above!(yb1[k],zb,th,yr1[k],zr1[k])
        @constraint(model, zr1[k] >= zb+Ut + opt.foot_margin); @constraint(model, zr1[k] >= zb+Up + opt.foot_margin)
        Bbx = -Fb1[k]*ubx; Bby = -Fb1[k]*uby; Bbz = -Fb1[k]*ubz; Bfx = -Ff1[k]*ufx; Bfy = -Ff1[k]*ufy; Bfz = -Ff1[k]*ufz
        Qy = Bby + Bfy
        Qt = (-Ut*Bby + (Ft-Fc)*Bbz) + (-Up*Bfy + (Fp-Fc)*Bfz)
        alpha = (Qt + m*g*Fc + m*Fc*Uc*om^2)/(I1 + m*Fc^2)
        @constraint(model, vyb1[k+1] == vyb1[k] + h1*Qy/m)
        @constraint(model, vxb1[k+1] == vxb1[k] + h1*(Bbx+Bfx)/m)          # lateral slide on ice
        @constraint(model, om1[k+1] == om + h1*alpha)
        @constraint(model, xb1[k+1] == xb1[k] + h1*vxb1[k+1]); @constraint(model, yb1[k+1] == yb1[k] + h1*vyb1[k+1])
        @constraint(model, th1[k+1] == th + h1*om1[k+1])
        @constraint(model, Rr1[k] == m*(-Fc*alpha + Uc*om^2 + g) - Bbz - Bfz)
        k > 1 && @constraint(model, -Uc + Ut >= 0)
        @constraint(model, -Uc + _U(th,rf) >= -1e-6)
        @constraint(model, vxr1[k+1] == vxr1[k] + h1*(Fb1[k]*ubx + Ff1[k]*ufx)/M)
        @constraint(model, vyr1[k+1] == vyr1[k] + h1*(Fb1[k]*uby + Ff1[k]*ufy)/M)
        @constraint(model, vzr1[k+1] == vzr1[k] + h1*((Fb1[k]*ubz + Ff1[k]*ufz)/M - g))
        @constraint(model, xr1[k+1] == xr1[k] + h1*vxr1[k+1]); @constraint(model, yr1[k+1] == yr1[k] + h1*vyr1[k+1]); @constraint(model, zr1[k+1] == zr1[k] + h1*vzr1[k+1])
    end
    the = th1[end]; Fce=_F(the,rc); Uce=_U(the,rc); Fte=_F(the,rt); Ute=_U(the,rt)
    @constraint(model, Ute - Uce == 0)
    @constraint(model, (Fte-Fce)*om1[end] <= -0.05)

    # ---------------- impact (vertical impulse at the tail centre; roll/yaw untouched) ----------------
    @constraints(model, begin
        xb2[1] == xb1[end]; yb2[1] == yb1[end]; zb2[1] == -Uce; th2[1] == the; ph2[1] == 0; ps2[1] == 0
        vxb2[1] == vxb1[end]; vyb2[1] == vyb1[end]
        vzb2[1] == -Fce*om1[end] + impulse/m
        w1[1] == om1[end] + Fte*impulse/I1
        w2[1] == 0; w3[1] == 0
        vzb2[1] + Fte*w1[1] == -b.restitution*(Fte-Fce)*om1[end]
        xr2[1] == xr1[end]; yr2[1] == yr1[end]; zr2[1] == zr1[end]; vxr2[1] == vxr1[end]; vyr2[1] == vyr1[end]; vzr2[1] == vzr1[end]
        s2[1] == s1[end]; sg2[1] == 0
    end)

    # ---------------- phase 2: flight ----------------
    tail_b = (0.0, rt[1], rt[2])
    wheels_b = ((-wl, -b.wheelbase/2, -zflat), (wl, -b.wheelbase/2, -zflat), (-wl, b.wheelbase/2, -zflat), (wl, b.wheelbase/2, -zflat))
    nose_b = (0.0, b.deck_length/2, 0.0)
    world(R, c, r) = (c[1] + R[1,1]*r[1] + R[1,2]*r[2] + R[1,3]*r[3],
                      c[2] + R[2,1]*r[1] + R[2,2]*r[2] + R[2,3]*r[3],
                      c[3] + R[3,1]*r[1] + R[3,2]*r[2] + R[3,3]*r[3])
    for k in 1:n2-1
        th, ph, ps = th2[k], ph2[k], ps2[k]
        R = rotmat(ps, th, ph)
        c = (xb2[k], yb2[k], zb2[k])
        n = (R[1,3], R[2,3], R[3,3])                          # deck normal, world
        front_b = (sg2[k], s2[k], 0.0)
        Pt = world(R, c, tail_b); Pf = world(R, c, front_b)
        ubx,uby,ubz,ℓb = leg3(Pt..., xr2[k], yr2[k], zr2[k]); ufx,ufy,ufz,ℓf = leg3(Pf..., xr2[k], yr2[k], zr2[k])
        legcons!(Nb2[k],ℓb); legcons!(Nf2[k],ℓf)
        @constraint(model, ℓb <= Lmax); @constraint(model, ℓf <= Lmax)     # feet stay on the board
        @constraint(model, zr2[k] >= Pt[3] + opt.foot_margin); @constraint(model, zr2[k] >= Pf[3] + opt.foot_margin)
        # a foot can only push while the rider is on the normal side of the deck (gated by the push)
        db = (xr2[k]-Pt[1])*n[1] + (yr2[k]-Pt[2])*n[2] + (zr2[k]-Pt[3])*n[3]
        df = (xr2[k]-Pf[1])*n[1] + (yr2[k]-Pf[2])*n[2] + (zr2[k]-Pf[3])*n[3]
        @constraint(model, Nb2[k]*(db - opt.rider_clearance) >= 0); @constraint(model, Nf2[k]*(df - opt.rider_clearance) >= 0)
        # forces ON THE BOARD in the body frame: normal push −e3·N plus tangential (G1 lateral, G2 long)
        FBb = (Gb1[k], Gb2[k], -Nb2[k]); FBf = (Gf1[k], Gf2[k], -Nf2[k])
        # world force on board = R F_b; on rider the negative
        Wb = (R[1,1]*FBb[1]+R[1,2]*FBb[2]+R[1,3]*FBb[3], R[2,1]*FBb[1]+R[2,2]*FBb[2]+R[2,3]*FBb[3], R[3,1]*FBb[1]+R[3,2]*FBb[2]+R[3,3]*FBb[3])
        Wf = (R[1,1]*FBf[1]+R[1,2]*FBf[2]+R[1,3]*FBf[3], R[2,1]*FBf[1]+R[2,2]*FBf[2]+R[2,3]*FBf[3], R[3,1]*FBf[1]+R[3,2]*FBf[2]+R[3,3]*FBf[3])
        # body torques τ = r × F_b
        cross(r,F) = (r[2]*F[3]-r[3]*F[2], r[3]*F[1]-r[1]*F[3], r[1]*F[2]-r[2]*F[1])
        tb = cross(tail_b, FBb); tf = cross(front_b, FBf)
        τ1 = tb[1]+tf[1]; τ2 = tb[2]+tf[2]; τ3 = tb[3]+tf[3]
        # Euler's equations (body frame)
        @constraint(model, I1*(w1[k+1]-w1[k]) == h2*(τ1 - (I3-I2)*w2[k]*w3[k]))
        @constraint(model, I2*(w2[k+1]-w2[k]) == h2*(τ2 - (I1-I3)*w3[k]*w1[k]))
        @constraint(model, I3*(w3[k+1]-w3[k]) == h2*(τ3 - (I2-I1)*w1[k]*w2[k]))
        # Euler-angle kinematics (implicit, no division):  θ̇ = cφ w1 + sφ w3 ; cθ ψ̇ = −sφ w1 + cφ w3 ; φ̇ = w2 − sθ ψ̇
        thdot = cos(ph)*w1[k+1] + sin(ph)*w3[k+1]
        @constraint(model, th2[k+1] == th + h2*thdot)
        if opt.yaw_free
            @constraint(model, cos(th)*(ps2[k+1]-ps) == h2*(-sin(ph)*w1[k+1] + cos(ph)*w3[k+1]))
            @constraint(model, ph2[k+1] == ph + h2*w2[k+1] - sin(th)*(ps2[k+1]-ps))
        else
            fix(ps2[k+1], 0.0; force=true)
            @constraint(model, -sin(ph)*w1[k+1] + cos(ph)*w3[k+1] == 0)   # yaw locked ⇒ constraint on ω
            @constraint(model, ph2[k+1] == ph + h2*w2[k+1])
        end
        # translation
        @constraint(model, vxb2[k+1] == vxb2[k] + h2*(Wb[1]+Wf[1])/m)
        @constraint(model, vyb2[k+1] == vyb2[k] + h2*(Wb[2]+Wf[2])/m)
        @constraint(model, vzb2[k+1] == vzb2[k] + h2*((Wb[3]+Wf[3])/m - g))
        @constraint(model, xb2[k+1] == xb2[k] + h2*vxb2[k+1]); @constraint(model, yb2[k+1] == yb2[k] + h2*vyb2[k+1]); @constraint(model, zb2[k+1] == zb2[k] + h2*vzb2[k+1])
        @constraint(model, vxr2[k+1] == vxr2[k] - h2*(Wb[1]+Wf[1])/M)
        @constraint(model, vyr2[k+1] == vyr2[k] - h2*(Wb[2]+Wf[2])/M)
        @constraint(model, vzr2[k+1] == vzr2[k] + h2*(-(Wb[3]+Wf[3])/M - g))
        @constraint(model, xr2[k+1] == xr2[k] + h2*vxr2[k+1]); @constraint(model, yr2[k+1] == yr2[k] + h2*vyr2[k+1]); @constraint(model, zr2[k+1] == zr2[k] + h2*vzr2[k+1])
    end
    # ground clearance of wheels, tail and nose during flight
    for k in 3:n2-1
        R = rotmat(ps2[k], th2[k], ph2[k]); c = (xb2[k], yb2[k], zb2[k])
        for r in (wheels_b..., tail_b, nose_b)
            @constraint(model, world(R, c, r)[3] >= 0)
        end
    end
    # apex and landing
    if opt.objective === :board_apex
        @constraint(model, vzb2[mid] == 0); height = zb2[mid]
    elseif opt.objective === :lowest_point
        @constraint(model, vzb2[mid] == 0)
        @variable(model, zlow)
        R = rotmat(ps2[mid], th2[mid], ph2[mid]); c = (xb2[mid], yb2[mid], zb2[mid])
        for r in (wheels_b..., tail_b, nose_b)
            @constraint(model, zlow <= world(R, c, r)[3])
        end
        height = zlow
    else
        height = 0.0
    end
    @variable(model, roll_target)
    fix(roll_target, 2π*opt.flip_turns*opt.flip_sign; force=true)
    @constraints(model, begin
        th2[end] == 0; zb2[end] == zflat; w1[end] == 0; vzb2[end] <= -0.02
        ph2[end] == roll_target
    end)
    opt.land_roll_rate_zero && @constraint(model, w2[end] == 0)
    opt.yaw_free && (@constraint(model, ps2[end] <= opt.land_yaw_max); @constraint(model, ps2[end] >= -opt.land_yaw_max))
    begin
        R = rotmat(ps2[end], 0.0, 2π*opt.flip_turns*opt.flip_sign); c = (xb2[end], yb2[end], zflat)
        Pt = world(R, c, tail_b); Pf = world(R, c, (sg2[end], s2[end], 0.0))
        _,_,_,ℓb = leg3(Pt..., xr2[end], yr2[end], zr2[end]); _,_,_,ℓf = leg3(Pf..., xr2[end], yr2[end], zr2[end])
        @constraint(model, ℓb <= Lmax); @constraint(model, ℓf <= Lmax)
        # COM (in board coords along the deck) between the feet / over the trucks
        dy = cos(ps2[end])*(yr2[end]-yb2[end]) + sin(ps2[end])*(xr2[end]-xb2[end])
        @constraint(model, rt[1] <= dy); @constraint(model, dy <= s2[end])
        opt.land_com_over_trucks && (@constraint(model, rc[1] <= dy); @constraint(model, dy <= rf[1]))
        opt.land_matched_velocity && (@constraint(model, vyr2[end] == vyb2[end]); @constraint(model, vxr2[end] == vxb2[end]))
        # COM laterally over the deck
        dx = cos(ps2[end])*(xr2[end]-xb2[end]) - sin(ps2[end])*(yr2[end]-yb2[end])
        @constraint(model, -halfw <= dx); @constraint(model, dx <= halfw)
    end

    rates = sum((Fb0[k+1]-Fb0[k])^2 + (Ff0[k+1]-Ff0[k])^2 + (s0[k+1]-s0[k])^2 for k in 1:n0-2) +
            sum((Fb1[k+1]-Fb1[k])^2 + (Ff1[k+1]-Ff1[k])^2 + (s1[k+1]-s1[k])^2 for k in 1:n1-2) +
            sum((Nb2[k+1]-Nb2[k])^2 + (Nf2[k+1]-Nf2[k])^2 + (s2[k+1]-s2[k])^2 + 4*(sg2[k+1]-sg2[k])^2 +
                (Gb1[k+1]-Gb1[k])^2 + (Gb2[k+1]-Gb2[k])^2 + (Gf1[k+1]-Gf1[k])^2 + (Gf2[k+1]-Gf2[k])^2 for k in 1:n2-2)

    # ---------------- initial guess ----------------
    T0g,T1g,T2g = opt.T0_guess, opt.T1_guess, opt.T2_guess
    set_start_value(T0,T0g); set_start_value(T1,T1g); set_start_value(T2,T2g)
    yr_init = (rt[1]+opt.front_point0)/2
    dyb = yr_init - rt[1]; dyf = opt.front_point0 - yr_init
    zr_init = max((dyf^2 - dyb^2 + 0.14^2 - 0.08^2)/(2*(0.14-0.08)), zflat + 0.8)
    theta_hit = 0.905
    for k in 1:n0
        set_start_value(xb0[k],0.0); set_start_value(yb0[k],opt.vy0*T0g*(k-1)/(n0-1)); set_start_value(vxb0[k],0.0); set_start_value(vyb0[k],opt.vy0)
        set_start_value(xr0[k],0.0); set_start_value(yr0[k],yr_init+opt.vy0*T0g*(k-1)/(n0-1)); set_start_value(zr0[k],zr_init)
        set_start_value(vxr0[k],0.0); set_start_value(vyr0[k],opt.vy0); set_start_value(vzr0[k],0.0)
    end
    for k in 1:n0-1
        tau=(k-1)/(n0-1)
        set_start_value(Fb0[k], M*g/2*(1+tau)); set_start_value(Ff0[k], M*g/2*(1-tau))
        set_start_value(s0[k],opt.front_point0); set_start_value(Rr0[k], m*g+M*g*(1+tau)/2); set_start_value(Rf0[k], M*g*(1-tau)/2)
    end
    yb_off = opt.vy0*T0g
    for k in 1:n1
        tau=(k-1)/(n1-1)
        set_start_value(th1[k],theta_hit*tau^2); set_start_value(om1[k],2theta_hit*tau/T1g)
        set_start_value(xb1[k],0.0); set_start_value(yb1[k],yb_off+opt.vy0*T1g*tau); set_start_value(vxb1[k],0.0); set_start_value(vyb1[k],opt.vy0)
        set_start_value(xr1[k],0.0); set_start_value(yr1[k],yr_init+yb_off+opt.vy0*T1g*tau); set_start_value(zr1[k],zr_init + 0.3*tau)
        set_start_value(vxr1[k],0.0); set_start_value(vyr1[k],opt.vy0); set_start_value(vzr1[k],0.3/T1g)
    end
    for k in 1:n1-1
        set_start_value(Fb1[k], Fmax*0.8); set_start_value(Ff1[k], 0.0); set_start_value(s1[k],opt.front_point0); set_start_value(Rr1[k], M*g)
    end
    zb_hit = 0.29*sin(theta_hit)+0.08*cos(theta_hit)
    yb_off2 = yb_off + opt.vy0*T1g
    for k in 1:n2
        tau=(k-1)/(n2-1)
        set_start_value(th2[k],theta_hit*(1-tau)); set_start_value(w1[k],-theta_hit/T2g)
        set_start_value(ph2[k], 2π*opt.flip_turns*opt.flip_sign*tau); set_start_value(w2[k], 2π*opt.flip_turns*opt.flip_sign/T2g)
        set_start_value(ps2[k],0.0); set_start_value(w3[k],0.0)
        set_start_value(zb2[k], zflat + (zb_hit-zflat)*(1-tau) + 0.4*4*tau*(1-tau))
        set_start_value(vzb2[k], 1.6*(1-2tau)/T2g)
        set_start_value(xb2[k],0.0); set_start_value(yb2[k],yb_off2+opt.vy0*T2g*tau); set_start_value(vxb2[k],0.0); set_start_value(vyb2[k],opt.vy0)
        set_start_value(xr2[k],0.0); set_start_value(yr2[k],yr_init+yb_off2+opt.vy0*T2g*tau); set_start_value(zr2[k], zr_init + 0.3 + 0.4*4*tau*(1-tau))
        set_start_value(vxr2[k],0.0); set_start_value(vyr2[k],opt.vy0); set_start_value(vzr2[k], 1.6*(1-2tau)/T2g)
    end
    for k in 1:n2-1
        set_start_value(Nb2[k],0.0); set_start_value(Nf2[k],2.0); set_start_value(s2[k],0.35); set_start_value(sg2[k],0.0)
    end
    set_start_value(impulse,1.0)

    # ---------------- optional: freeze the ground phases + impact to a given ollie ----------------
    if ground !== nothing
        gvars = (; T0, T1, xb0, yb0, vxb0, vyb0, xr0, yr0, zr0, vxr0, vyr0, vzr0, Fb0, Ff0, s0, Rr0, Rf0,
                   xb1, yb1, vxb1, vyb1, th1, om1, xr1, yr1, zr1, vxr1, vyr1, vzr1, Fb1, Ff1, s1, Rr1, impulse)
        for (name, var) in pairs(gvars)
            val = getfield(ground, name)
            if var isa AbstractArray
                for i in eachindex(var); fix(var[i], val[i]; force=true); end
            else
                fix(var, val; force=true)
            end
        end
        # flight warm start from the ollie flight if provided
        if hasproperty(ground, :flight)
            f = ground.flight
            set_start_value(T2, f.T2)
            nf = length(f.th2)
            if nf == n2
                for k in 1:n2
                    set_start_value(th2[k], f.th2[k]); set_start_value(w1[k], f.w1[k]); set_start_value(zb2[k], f.zb2[k]); set_start_value(vzb2[k], f.vzb2[k])
                    set_start_value(yb2[k], f.yb2[k]); set_start_value(vyb2[k], f.vyb2[k]); set_start_value(yr2[k], f.yr2[k]); set_start_value(zr2[k], f.zr2[k])
                    set_start_value(vyr2[k], f.vyr2[k]); set_start_value(vzr2[k], f.vzr2[k])
                end
                for k in 1:n2-1; set_start_value(Nb2[k], f.Nb2[k]); set_start_value(Nf2[k], f.Nf2[k]); set_start_value(s2[k], f.s2[k]); end
            end
        end
    end

    # ---------------- optional: full warm start from a rider3d_data NamedTuple (any flight mesh) ----------------
    if warm !== nothing
        allvars = (; T0, T1, T2, impulse, xb0, yb0, vxb0, vyb0, xr0, yr0, zr0, vxr0, vyr0, vzr0, Fb0, Ff0, s0, Rr0, Rf0,
                     xb1, yb1, vxb1, vyb1, th1, om1, xr1, yr1, zr1, vxr1, vyr1, vzr1, Fb1, Ff1, s1, Rr1,
                     xb2, yb2, zb2, vxb2, vyb2, vzb2, th2, ph2, ps2, w1, w2, w3, xr2, yr2, zr2, vxr2, vyr2, vzr2,
                     Nb2, Nf2, Gb1, Gb2, Gf1, Gf2, s2, sg2)
        interp(vals, n) = length(vals) == n ? vals : [begin
                τ = (i-1)/(n-1)*(length(vals)-1); j = clamp(floor(Int, τ)+1, 1, length(vals)-1); f = τ-(j-1)
                (1-f)*vals[j] + f*vals[j+1] end for i in 1:n]
        for (name, var) in pairs(allvars)
            hasproperty(warm, name) || continue
            val = getfield(warm, name)
            if var isa AbstractArray
                vv = interp(collect(val), length(var))
                for i in eachindex(var); is_fixed(var[i]) || set_start_value(var[i], vv[i]); end
            else
                is_fixed(var) || set_start_value(var, val)
            end
        end
        staged = false
        set_optimizer_attribute(model,"warm_start_init_point","yes"); set_optimizer_attribute(model,"mu_init",1e-3)
    end
    stat = nothing
    if staged && ground === nothing
        fix(T0,T0g;force=true); fix(T1,T1g;force=true); fix(T2,T2g;force=true)
        @objective(model, Min, rates); optimize!(model)
        println("  stage 1 (fixed times, feasibility): ", termination_status(model), "  theta_hit=", round(value(th1[end]),digits=3))
        unfix(T0); unfix(T1); unfix(T2)
        set_lower_bound(T0,opt.load_time_bounds[1]); set_upper_bound(T0,opt.load_time_bounds[2])
        set_lower_bound(T1,opt.support_time_bounds[1]); set_upper_bound(T1,opt.support_time_bounds[2])
        set_lower_bound(T2,opt.flight_time_bounds[1]); set_upper_bound(T2,opt.flight_time_bounds[2])
        set_optimizer_attribute(model,"warm_start_init_point","yes"); set_optimizer_attribute(model,"mu_init",1e-3)
    end
    # continuation in the roll target
    if roll_targets !== nothing
        for rtg in roll_targets
            fix(roll_target, 2π*rtg*opt.flip_sign; force=true)
            @objective(model, Min, rates); optimize!(model)
            println("  roll continuation target=$(rtg) turns: ", termination_status(model), "  T2=", round(value(T2),digits=3), " zb apex=", round(maximum(value.(zb2)),digits=3))
            set_optimizer_attribute(model,"warm_start_init_point","yes"); set_optimizer_attribute(model,"mu_init",1e-3)
        end
        fix(roll_target, 2π*opt.flip_turns*opt.flip_sign; force=true)
    end
    if opt.objective === :feasible
        @objective(model, Min, rates)
    else
        @objective(model, Max, height - opt.force_rate_weight*rates)
    end
    optimize!(model)
    return (; model, b, p, opt, mid, T0, T1, T2, impulse,
              xb0, yb0, vxb0, vyb0, xr0, yr0, zr0, vxr0, vyr0, vzr0, Fb0, Ff0, s0, Rr0, Rf0,
              xb1, yb1, vxb1, vyb1, th1, om1, xr1, yr1, zr1, vxr1, vyr1, vzr1, Fb1, Ff1, s1, Rr1,
              xb2, yb2, zb2, vxb2, vyb2, vzb2, th2, ph2, ps2, w1, w2, w3, xr2, yr2, zr2, vxr2, vyr2, vzr2,
              Nb2, Nf2, Gb1, Gb2, Gf1, Gf2, s2, sg2)
end

"Values of the ground-phase variables (+ flight warm start) of a solved model, for `ground=` reuse."
function ground_solution(sol)
    v(x) = value.(x)
    (; T0=value(sol.T0), T1=value(sol.T1), xb0=v(sol.xb0), yb0=v(sol.yb0), vxb0=v(sol.vxb0), vyb0=v(sol.vyb0),
       xr0=v(sol.xr0), yr0=v(sol.yr0), zr0=v(sol.zr0), vxr0=v(sol.vxr0), vyr0=v(sol.vyr0), vzr0=v(sol.vzr0),
       Fb0=v(sol.Fb0), Ff0=v(sol.Ff0), s0=v(sol.s0), Rr0=v(sol.Rr0), Rf0=v(sol.Rf0),
       xb1=v(sol.xb1), yb1=v(sol.yb1), vxb1=v(sol.vxb1), vyb1=v(sol.vyb1), th1=v(sol.th1), om1=v(sol.om1),
       xr1=v(sol.xr1), yr1=v(sol.yr1), zr1=v(sol.zr1), vxr1=v(sol.vxr1), vyr1=v(sol.vyr1), vzr1=v(sol.vzr1),
       Fb1=v(sol.Fb1), Ff1=v(sol.Ff1), s1=v(sol.s1), Rr1=v(sol.Rr1), impulse=value(sol.impulse),
       flight=(; T2=value(sol.T2), th2=v(sol.th2), w1=v(sol.w1), zb2=v(sol.zb2), vzb2=v(sol.vzb2), yb2=v(sol.yb2), vyb2=v(sol.vyb2),
                 yr2=v(sol.yr2), zr2=v(sol.zr2), vyr2=v(sol.vyr2), vzr2=v(sol.vzr2), Nb2=v(sol.Nb2), Nf2=v(sol.Nf2), s2=v(sol.s2)))
end

function rider3d_data(sol)
    v(x)=value.(x); p=sol.p; rc=body_point(p,:rear_slide)
    T0,T1,T2=value(sol.T0),value(sol.T1),value(sol.T2)
    n0,n1,n2 = sol.opt.n_load, sol.opt.n_support, sol.opt.n_flight
    t0=collect(range(0,T0,length=n0)); t1=collect(range(T0,T0+T1,length=n1)); t2=collect(range(T0+T1,T0+T1+T2,length=n2))
    th1=v(sol.th1); zb1=[-_U(t,rc) for t in th1]
    (; status=termination_status(sol.model), T0,T1,T2,t0,t1,t2, mid=sol.mid, impulse=value(sol.impulse),
       xb0=v(sol.xb0), yb0=v(sol.yb0), zb0=fill(p.deck_height,n0), vxb0=v(sol.vxb0), vyb0=v(sol.vyb0),
       xr0=v(sol.xr0), yr0=v(sol.yr0), zr0=v(sol.zr0), vxr0=v(sol.vxr0), vyr0=v(sol.vyr0), vzr0=v(sol.vzr0),
       Fb0=v(sol.Fb0), Ff0=v(sol.Ff0), s0=v(sol.s0), Rr0=v(sol.Rr0), Rf0=v(sol.Rf0),
       xb1=v(sol.xb1), yb1=v(sol.yb1), zb1, vxb1=v(sol.vxb1), vyb1=v(sol.vyb1), th1, om1=v(sol.om1),
       xr1=v(sol.xr1), yr1=v(sol.yr1), zr1=v(sol.zr1), vxr1=v(sol.vxr1), vyr1=v(sol.vyr1), vzr1=v(sol.vzr1),
       Fb1=v(sol.Fb1), Ff1=v(sol.Ff1), s1=v(sol.s1), Rr1=v(sol.Rr1),
       xb2=v(sol.xb2), yb2=v(sol.yb2), zb2=v(sol.zb2), vxb2=v(sol.vxb2), vyb2=v(sol.vyb2), vzb2=v(sol.vzb2),
       th2=v(sol.th2), ph2=v(sol.ph2), ps2=v(sol.ps2), w1=v(sol.w1), w2=v(sol.w2), w3=v(sol.w3),
       xr2=v(sol.xr2), yr2=v(sol.yr2), zr2=v(sol.zr2), vxr2=v(sol.vxr2), vyr2=v(sol.vyr2), vzr2=v(sol.vzr2),
       Nb2=v(sol.Nb2), Nf2=v(sol.Nf2), Gb1=v(sol.Gb1), Gb2=v(sol.Gb2), Gf1=v(sol.Gf1), Gf2=v(sol.Gf2), s2=v(sol.s2), sg2=v(sol.sg2))
end

"""
Audit: energy bookkeeping over the flight (leg work vs ΔE), world angular-momentum balance of the
board in flight (ΔH vs ∫τ dt), and geometric summaries (apex, roll at landing, peak forces).
"""
function audit_rider_3d(sol)
    d = rider3d_data(sol); b = sol.b; m,g,M = b.mass,b.gravity,sol.opt.rider_mass
    I = Diagonal([b.I_pitch, b.I_roll, b.I_yaw])
    n2 = length(d.th2); h2 = d.T2/(n2-1)
    Rk(k) = rotmat(d.ps2[k], d.th2[k], d.ph2[k])
    p = sol.p; rt = body_point(p,:tail)
    tail_b = [0.0, rt[1], rt[2]]
    E(k) = 0.5m*(d.vxb2[k]^2+d.vyb2[k]^2+d.vzb2[k]^2) + 0.5*dot([d.w1[k],d.w2[k],d.w3[k]], I*[d.w1[k],d.w2[k],d.w3[k]]) + m*g*d.zb2[k] +
           0.5M*(d.vxr2[k]^2+d.vyr2[k]^2+d.vzr2[k]^2) + M*g*d.zr2[k]
    work = 0.0; dH_res = zeros(3); Hprev = nothing; Hres_max = 0.0
    for k in 1:n2-1
        R = Rk(k); FBb = [d.Gb1[k], d.Gb2[k], -d.Nb2[k]]; FBf = [d.Gf1[k], d.Gf2[k], -d.Nf2[k]]
        front_b = [d.sg2[k], d.s2[k], 0.0]
        Pt = [d.xb2[k],d.yb2[k],d.zb2[k]] + R*tail_b; Pf = [d.xb2[k],d.yb2[k],d.zb2[k]] + R*front_b
        Wb = R*FBb; Wf = R*FBf
        # power of the leg forces: force on board · v_board_point + force on rider · v_rider
        ω = [d.w1[k+1], d.w2[k+1], d.w3[k+1]]
        vb = [d.vxb2[k+1], d.vyb2[k+1], d.vzb2[k+1]]
        vPt = vb + R*cross(ω, tail_b); vPf = vb + R*cross(ω, front_b)
        vr = [d.vxr2[k+1], d.vyr2[k+1], d.vzr2[k+1]]
        work += h2*(dot(Wb, vPt) + dot(Wf, vPf) - dot(Wb+Wf, vr))
        τb = cross(tail_b, FBb) + cross(front_b, FBf)
        Hk = Rk(k)*(I*[d.w1[k],d.w2[k],d.w3[k]]); Hk1 = Rk(k+1)*(I*[d.w1[k+1],d.w2[k+1],d.w3[k+1]])
        res = Hk1 - Hk - h2*(R*τb)
        Hres_max = max(Hres_max, norm(res))
    end
    ΔE = E(n2) - E(1)
    lowest(k) = begin
        R = Rk(k); c = [d.xb2[k],d.yb2[k],d.zb2[k]]
        wl = b.axle_half; pts = ([-wl,-b.wheelbase/2,-b.deck_height],[wl,-b.wheelbase/2,-b.deck_height],[-wl,b.wheelbase/2,-b.deck_height],[wl,b.wheelbase/2,-b.deck_height],tail_b,[0,b.deck_length/2,0])
        minimum(c[3] + (R*pt)[3] for pt in pts)
    end
    (; status=d.status, T0=d.T0, T1=d.T1, T2=d.T2, theta_hit=d.th1[end], impulse=d.impulse,
       board_apex_rise=maximum(d.zb2)-b.deck_height, lowest_point_apex=maximum(lowest(k) for k in 1:n2),
       theta_at_mid=d.th2[d.mid], roll_at_mid=d.ph2[d.mid], roll_end=d.ph2[end], roll_turns=d.ph2[end]/(2π),
       yaw_range=extrema(d.ps2), max_roll_rate=maximum(abs.(d.w2)), roll_rate_end=d.w2[end],
       rider_apex=maximum(vcat(d.zr0,d.zr1,d.zr2)),
       peak_force_ground=maximum(vcat(d.Fb0,d.Ff0,d.Fb1,d.Ff1)), peak_normal_flight=(maximum(d.Nb2), maximum(d.Nf2)),
       sigma_range=extrema(d.sg2), s_range=extrema(d.s2), lateral_board_range=extrema(d.xb2), lateral_rider_range=extrema(d.xr2),
       flight_leg_work=work, flight_energy_change=ΔE, flight_energy_residual=work-ΔE,
       semi_implicit_gravity_bias=(n2-1)*0.5*(m+M)*g^2*h2^2,   # symplectic-Euler drift −½(m+M)g²h² per step, explains most of the residual
       angular_momentum_residual_max=Hres_max,
       landing_v_board=(d.vxb2[end], d.vyb2[end], d.vzb2[end]), landing_v_rider=(d.vxr2[end], d.vyr2[end], d.vzr2[end]),
       landing_com_rel=(d.xr2[end]-d.xb2[end], d.yr2[end]-d.yb2[end]))
end

end # module
