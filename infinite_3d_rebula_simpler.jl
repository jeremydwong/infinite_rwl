# 3D Point Mass Walking Model with Two Force Actuators acting along the leg.
# Extension of the y-z 2-D model by adding lateral x.
# if you want to run it without
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
# using Revise
using InfiniteOpt, Ipopt, Distributions, LinearAlgebra, Plots

# Keep Qt/plotting artifacts out of batch optimization runs. Plotting functions
# load Plots only when they are actually requested.
_load_plots_3d!() = true

const g = 1  # Gravity acts in -z only.

"""
    point_mass_walker_3d(; c_fr=0.05, c_fr2=0.0001, step_length=nothing, step_speed=0.4,
                          step_width=0.1, z0_fixed=0.98, cost_type=:squared)

Solve the 3D point mass walking optimization problem using InfiniteOpt.

Models a point mass with two leg force actuators (trailing and leading legs) that push
along the leg direction in 3D space. Minimizes a cost combining mechanical work and
force rate penalties.

# Coordinate System
- x: lateral direction
- y: forward direction
- z: vertical direction (gravity acts in -z)

# Arguments
- `c_fr::Float64`: Linear force rate penalty coefficient (default: 0.05)
- `c_fr2::Float64`: Squared force rate penalty coefficient (default: 0.0001)
- `step_length`: Step length in forward y (default: 2*z_0*sin(α) ≈ 0.65)
- `step_speed`: Dimensionless forward overground speed,
  `step_length / step_time` (default: 0.4, about 1.25 m/s for L=1 m)
- `step_width`: Lateral distance between feet in x (default: 0.1)
- `z0_fixed`: Fixed midstance COM height, matching the 2-D model (default: 0.98)
- `cost_type::Symbol`: Objective function type:
  - `:linear` → minimize work + c_fr * ∫|F̈|dt
  - `:squared` → minimize work + c_fr2 * ∫F̈²dt

# Returns
- `model::InfiniteModel`: The solved optimization model

# Example
```julia
model = point_mass_walker_3d()
o = object_dictionary(model)
value(o[:cost_work])     # Mechanical work cost
value(o[:px])            # Lateral position
value(o[:py])            # Forward position
value(o[:pz])            # Vertical position
```
"""
function point_mass_walker_3d(;
        c_fr = 0.05,
        c_fr2 = 0.0001,
        step_length = nothing,
        step_speed = 0.4,
        step_width = 0.1,      # Lateral foot placement width
        px_initial = nothing,  # Defaults halfway from midline to the stance foot
        z0_fixed = 0.98,
        initial_forward_velocity = nothing,
        forward_velocity_constraint = :endpoints,
        enforce_leg_reach = true,
        cost_type = :squared,
        num_supports = 101,
        max_cpu_time = 120.0,
        solve = true,
    )
    ## 3D Point Mass Walking Model
    model = InfiniteModel(Ipopt.Optimizer)

    # Model parameters
    c_t = 5.0    # Time penalty coefficient
    @finite_parameter(model, z_0 == 0.95)  # Nominal leg length for default step geometry
    @finite_parameter(model, α == 0.35)     # Leg angle parameter

    # Step length: use provided value or default
    sl = isnothing(step_length) ? 2*parameter_value(z_0)*sin(parameter_value(α)) : step_length

    step_width >= 0 || throw(ArgumentError("step_width must be nonnegative"))
    isnothing(z0_fixed) || z0_fixed > 0 || throw(ArgumentError("z0_fixed must be positive or nothing"))
    isnothing(step_speed) || step_speed > 0 || throw(ArgumentError("step_speed must be positive"))

    # `step_speed` is overground forward speed, not speed along a diagonal
    # displacement through the x-y plane. Thus v*=0.4 corresponds to about 1.25 m/s
    # for L=1 m, and every width uses the same forward step time.
    t_f_val = isnothing(step_speed) ? 1.2 : sl / step_speed
    # Keeping the COM slightly medial to the stance foot gives the stance leg a
    # lateral force direction at the boundary and is much better conditioned than
    # placing it exactly over the foot. The next endpoint is its negative mirror.
    px0 = isnothing(px_initial) ? step_width / 4 : px_initial
    zstart = isnothing(z0_fixed) ? 0.95 : z0_fixed

    # Infinite time parameter
    @infinite_parameter(model, τ ∈ [0, 1], num_supports=num_supports, derivative_method = OrthogonalCollocation(2))

    # State variables with bounds and initial guesses
    @variable(model, px, Infinite(τ), start = (t) -> px0*(1-2t)) # lateral
    @variable(model, py, Infinite(τ), start = (t) -> sl*t)        # forward
    @variable(model, pz >= 0.1, Infinite(τ), start = (t) -> zstart + 0.03*sin(π*t)) # vertical

    @variable(model, vx, Infinite(τ), start = (t) -> -2px0/t_f_val)
    @variable(model, vy, Infinite(τ), start = (t) -> sl/t_f_val)
    @variable(model, vz, Infinite(τ), start = (t) -> 0.0)

    # Variable: Force magnitude along the leg, and must be positive (legs can only push)
    @variable(model, F_trail >= 0, Infinite(τ), start = (t) -> cos(π*t/2)^2)
    @variable(model, F_lead >= 0, Infinite(τ), start = (t) -> sin(π*t/2)^2)

    # Time scaling variable
    @variable(model, 0.001 <= t_f <= 10, start = t_f_val)

    # Fixed leg contact positions in 3D
    @finite_parameter(model, P_trail_x == step_width/2)
    @finite_parameter(model, P_trail_y == 0.0)
    @finite_parameter(model, P_trail_z == 0.0)

    @finite_parameter(model, P_lead_x == -step_width/2)
    @finite_parameter(model, P_lead_y == sl)
    @finite_parameter(model, P_lead_z == 0.0)

    # Leg vectors (from contact points to COM) in 3D
    @expression(model, trail_leg_x, px - P_trail_x)
    @expression(model, trail_leg_y, py - P_trail_y)
    @expression(model, trail_leg_z, pz - P_trail_z)

    @expression(model, lead_leg_x, px - P_lead_x)
    @expression(model, lead_leg_y, py - P_lead_y)
    @expression(model, lead_leg_z, pz - P_lead_z)

    # Leg lengths in 3D
    @expression(model, trail_leg_length, sqrt(trail_leg_x^2 + trail_leg_y^2 + trail_leg_z^2))
    @expression(model, lead_leg_length, sqrt(lead_leg_x^2 + lead_leg_y^2 + lead_leg_z^2))

    # Unit vectors along each leg in 3D
    @expression(model, trail_unit_x, trail_leg_x / trail_leg_length)
    @expression(model, trail_unit_y, trail_leg_y / trail_leg_length)
    @expression(model, trail_unit_z, trail_leg_z / trail_leg_length)

    @expression(model, lead_unit_x, lead_leg_x / lead_leg_length)
    @expression(model, lead_unit_y, lead_leg_y / lead_leg_length)
    @expression(model, lead_unit_z, lead_leg_z / lead_leg_length)

    # Force components for each leg in 3D
    @expression(model, Ftrail_x, F_trail * trail_unit_x)
    @expression(model, Ftrail_y, F_trail * trail_unit_y)
    @expression(model, Ftrail_z, F_trail * trail_unit_z)

    @expression(model, Flead_x, F_lead * lead_unit_x)
    @expression(model, Flead_y, F_lead * lead_unit_y)
    @expression(model, Flead_z, F_lead * lead_unit_z)

    # Total force components
    @expression(model, Ftot_x, Ftrail_x + Flead_x)
    @expression(model, Ftot_y, Ftrail_y + Flead_y)
    @expression(model, Ftot_z, Ftrail_z + Flead_z)

    # Leg velocity (rate of change of leg length) in 3D
    # d/dt(leg_length) = (leg · vel) / leg_length
    @expression(model, trail_leg_velocity,
            (trail_leg_x*vx + trail_leg_y*vy + trail_leg_z*vz) / trail_leg_length)
    @expression(model, lead_leg_velocity,
            (lead_leg_x*vx + lead_leg_y*vy + lead_leg_z*vz) / lead_leg_length)

    # Power slack variables for each leg
    @variable(model, pospower_trail >= 0, Infinite(τ))
    @variable(model, negpower_trail <= 0, Infinite(τ))
    @variable(model, pospower_lead >= 0, Infinite(τ))
    @variable(model, negpower_lead <= 0, Infinite(τ))

    @expression(model, mechpower_trail, F_trail * trail_leg_velocity)
    @expression(model, mechpower_lead, F_lead * lead_leg_velocity)

    # Power splitting constraints
    @constraint(model, pospower_trail >= mechpower_trail)
    @constraint(model, pospower_trail >= 0)
    @constraint(model, negpower_trail <= mechpower_trail)
    @constraint(model, negpower_trail <= 0)

    @constraint(model, pospower_lead >= mechpower_lead)
    @constraint(model, pospower_lead >= 0)
    @constraint(model, negpower_lead <= mechpower_lead)
    @constraint(model, negpower_lead <= 0)

    # System dynamics in 3D
    @constraint(model, ∂(px, τ) == t_f * vx)
    @constraint(model, ∂(py, τ) == t_f * vy)
    @constraint(model, ∂(pz, τ) == t_f * vz)
    @constraint(model, ∂(vx, τ) == t_f * Ftot_x)
    @constraint(model, ∂(vy, τ) == t_f * Ftot_y)
    @constraint(model, ∂(vz, τ) == t_f * (Ftot_z - g))

    # Force rate variables
    @variable(model, Fdot_trail, Infinite(τ), start = (t) -> -π*sin(π*t)/(2*t_f_val))
    @variable(model, Fdot_lead, Infinite(τ), start = (t) -> π*sin(π*t)/(2*t_f_val))

    @variable(model, Fddot_trail_p >= 0, Infinite(τ))
    @variable(model, Fddot_trail_m >= 0, Infinite(τ))
    @variable(model, Fddot_lead_p >= 0, Infinite(τ))
    @variable(model, Fddot_lead_m >= 0, Infinite(τ))

    @variable(model, fdot_scale == 1)
    @variable(model, fddot_scale == 1)
    @constraint(model, ∂(F_trail, τ) == t_f * (Fdot_trail)/fdot_scale)
    @constraint(model, ∂(F_lead, τ) == t_f * (Fdot_lead)/fdot_scale)
    @constraint(model, ∂(Fdot_trail, τ) == t_f * ((Fddot_trail_p - Fddot_trail_m)/fddot_scale))
    @constraint(model, ∂(Fdot_lead, τ) == t_f * ((Fddot_lead_p - Fddot_lead_m)/fddot_scale))

    # Leg length constraints (forces only active when leg length <= 1)
    if enforce_leg_reach
        @constraint(model, F_trail * (trail_leg_length - 1) <= 0)
        @constraint(model, F_lead * (lead_leg_length - 1) <= 0)
    end

    # Boundary conditions
    @constraint(model, px(0) == px0)
    @constraint(model, py(0) == 0)
    if !isnothing(z0_fixed)
        @constraint(model, pz(0) == z0_fixed)
    end

    # Final conditions (symmetric gait)
    # For alternating gait: next step swaps left/right, so:
    @constraint(model, px(1) == -px0)
    @constraint(model, py(1) == sl)
    @constraint(model, pz(1) == pz(0))
    @constraint(model, vx(1) == vx(0))
    @constraint(model, vz(0) == 0)
    @constraint(model, vz(1) == 0)
    if forward_velocity_constraint == :constant
        @constraint(model, vy == step_speed)
    elseif forward_velocity_constraint == :endpoints
        vy_endpoint = isnothing(initial_forward_velocity) ? step_speed : initial_forward_velocity
        @constraint(model, vy(0) == vy_endpoint)
        @constraint(model, vy(1) == vy_endpoint)
    elseif forward_velocity_constraint == :periodic
        @constraint(model, vy(1) == vy(0))
    else
        throw(ArgumentError("forward_velocity_constraint must be :constant, :endpoints, or :periodic"))
    end

    # Force boundary conditions
    @constraint(model, F_trail(0) == 1)
    @constraint(model, F_lead(0) == 0)
    @constraint(model, F_trail(1) == 0)
    @constraint(model, F_lead(1) == 1)
    @constraint(model, Fdot_lead(0) == -Fdot_trail(1))
    @constraint(model, Fdot_trail(0) == -Fdot_lead(1))

    @constraint(model, t_f == t_f_val)

    # Objective function
    @expression(model, cost_work, integral(pospower_trail, τ) * t_f + integral(pospower_lead, τ) * t_f -
            integral(negpower_trail, τ) * t_f - integral(negpower_lead, τ) * t_f)

    @expression(model, cost_fr, c_fr*integral(Fddot_trail_p, τ) * t_f + c_fr*integral(Fddot_lead_p, τ)*t_f +
        c_fr*integral(Fddot_trail_m, τ) * t_f + c_fr*integral(Fddot_lead_m, τ)*t_f)

    @expression(model, cost_fr2, c_fr2*integral(Fddot_trail_p^2, τ) * t_f + c_fr2*integral(Fddot_lead_p^2, τ)*t_f +
        c_fr2*integral(Fddot_trail_m^2, τ) * t_f + c_fr2*integral(Fddot_lead_m^2, τ)*t_f)

    @expression(model, cost_time, c_t*t_f)

    if cost_type == :linear
        @objective(model, Min, cost_work + cost_fr)
    elseif cost_type == :squared
        @objective(model, Min, cost_work + cost_fr2)
    else
        error("cost_type must be :linear or :squared")
    end

    # Solver parameters
    set_optimizer_attribute(model, "max_cpu_time", max_cpu_time)
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "print_level", 0)
    set_optimizer_attribute(model, "warm_start_init_point", "yes")

    solve && optimize!(model)
    return model
end

"""
    plot_results_3d(model)

Plot the solution trajectories from a solved 3D walking optimization model.

This is the direct 3-D analogue of `infinite_rebula_simpler.plot_results`.
It preserves the same 4x2 layout and panel order, adding z traces only where
the 3-D model has an additional component.
"""
function plot_results_3d(model; show_legend=true)
    _load_plots_3d!()
    o = object_dictionary(model)
    t_ = value(o[:τ]) * value(o[:t_f])

    px_v = value(o[:px])
    py_v = value(o[:py])
    pz_v = value(o[:pz])
    vx_v = value(o[:vx])
    vy_v = value(o[:vy])
    vz_v = value(o[:vz])

    println("Boundary conditions check:")
    println("  px(0)=$(round(px_v[1],digits=4)), px(1)=$(round(px_v[end],digits=4))")
    println("  py(0)=$(round(py_v[1],digits=4)), py(1)=$(round(py_v[end],digits=4))")
    println("  pz(0)=$(round(pz_v[1],digits=4)), pz(1)=$(round(pz_v[end],digits=4))")
    println("  vx(0)=$(round(vx_v[1],digits=4)), vx(1)=$(round(vx_v[end],digits=4))")
    println("  vy(0)=$(round(vy_v[1],digits=4)), vy(1)=$(round(vy_v[end],digits=4))")
    println("  vz(0)=$(round(vz_v[1],digits=4)), vz(1)=$(round(vz_v[end],digits=4))")

    txt = "Work: " * string(round(value(o[:cost_work]), digits=3)) *
          " FR2: " * string(round(value(o[:cost_fr2]), digits=3))
    leg_pos = show_legend ? :topleft : false

    # Get foot positions
    P_trail_x = parameter_value(o[:P_trail_x])
    P_trail_y = parameter_value(o[:P_trail_y])
    P_trail_z = parameter_value(o[:P_trail_z])
    P_lead_x = parameter_value(o[:P_lead_x])
    P_lead_y = parameter_value(o[:P_lead_y])
    P_lead_z = parameter_value(o[:P_lead_z])

    # Get leg lengths for filtering
    trail_len = value(o[:trail_leg_length])
    lead_len = value(o[:lead_leg_length])

    # Panel 1: the identical sagittal projection used in the 2-D figure.
    f = plot(py_v,pz_v,layout=(4,2),size=(800,800),subplot=1,
             xlabel="y (forward)",ylabel="z (vertical)",title=txt,label="COM",legend=leg_pos,
             aspect_ratio=:equal)

    # Add start (green) and end (red) dots
    scatter!(f,[py_v[1]],[pz_v[1]],subplot=1,color=:green,markersize=8,label="start")
    scatter!(f,[py_v[end]],[pz_v[end]],subplot=1,color=:red,markersize=8,label="end")

    # Draw leg lines every 10 nodes, only if leg length <= 1
    for i in 1:10:length(px_v)
        # Trail leg
        if trail_len[i] <= 1.0
            plot!(f, [P_trail_y, py_v[i]], [P_trail_z, pz_v[i]], subplot=1,
                  color=:blue, linewidth=1, alpha=0.5, label=(i==1 ? "trail" : nothing))
        end
        # Lead leg
        if lead_len[i] <= 1.0
            plot!(f, [P_lead_y, py_v[i]], [P_lead_z, pz_v[i]], subplot=1,
                  color=:red, linewidth=1, alpha=0.5, label=(i==1 ? "lead" : nothing))
        end
    end

    # Panel 2: position versus time, with lateral position added.
    plot!(f,t_,px_v,subplot=2,ylabel="pxyz",xlabel="time",label="px",legend=leg_pos)
    plot!(f,t_,py_v,subplot=2,label="py",legend=leg_pos)
    plot!(f,t_,pz_v,subplot=2,label="pz",legend=leg_pos)

    # Panel 3: velocity versus time, with lateral velocity added.
    plot!(f,t_,vx_v,subplot=3,label="vx",ylabel="vel",legend=leg_pos)
    plot!(f,t_,vy_v,subplot=3,label="vy",legend=leg_pos)
    plot!(f,t_,vz_v,subplot=3,label="vz",legend=leg_pos)

    # Panel 4: solid=forward y, dashed=vertical z, dotted=lateral x.
    plot!(f,t_,value(o[:Ftrail_y]),subplot=4,ylabel="force",color=:blue,linestyle=:solid,label="Ftrail_y",legend=leg_pos)
    plot!(f,t_,value(o[:Ftrail_z]),subplot=4,color=:blue,linestyle=:dash,label="Ftrail_z",legend=leg_pos)
    plot!(f,t_,value(o[:Ftrail_x]),subplot=4,color=:blue,linestyle=:dot,label="Ftrail_x",legend=leg_pos)
    plot!(f,t_,value(o[:Flead_y]),subplot=4,color=:red,linestyle=:solid,label="Flead_y",legend=leg_pos)
    plot!(f,t_,value(o[:Flead_z]),subplot=4,color=:red,linestyle=:dash,label="Flead_z",legend=leg_pos)
    plot!(f,t_,value(o[:Flead_x]),subplot=4,color=:red,linestyle=:dot,label="Flead_x",legend=leg_pos)

    # Panel 5: leg lengths.
    trail_len_clipped = [l > 1.0 ? NaN : l for l in trail_len]
    lead_len_clipped = [l > 1.0 ? NaN : l for l in lead_len]
    plot!(f,t_,trail_len_clipped,subplot=5,linewidth=2,ylabel="leg length",color=:blue,label="trail",legend=leg_pos)
    plot!(f,t_,lead_len_clipped,subplot=5,color=:red,label="lead",legend=leg_pos)
    hline!(f,[1.0],subplot=5,linestyle=:dash,color=:black,label="max",legend=leg_pos)

    # Panel 6: force rate (same variables as the 2-D diagnostic).
    plot!(f,t_,value(o[:Fddot_trail_p])-value(o[:Fddot_trail_m]),subplot=6,ylabel="frate",color=:blue,label="trail",legend=leg_pos)
    plot!(f,t_,value(o[:Fddot_lead_p])-value(o[:Fddot_lead_m]),subplot=6,color=:red,label="lead",legend=leg_pos)

    # Panel 7: leg velocity.
    plot!(f,t_,value(o[:trail_leg_velocity]),subplot=7,ylabel="dlegdt",color=:blue,label="trail",legend=leg_pos)
    plot!(f,t_,value(o[:lead_leg_velocity]),subplot=7,color=:red,label="lead",legend=leg_pos)

    # 8. Dynamics violation check
    vx_dot = value(∂(o[:vx], o[:τ]))
    vy_dot = value(∂(o[:vy], o[:τ]))
    vz_dot = value(∂(o[:vz], o[:τ]))
    t_f_v = value(o[:t_f])

    dyn_viol_x = vx_dot - value(o[:Ftot_x]) * t_f_v
    dyn_viol_y = vy_dot - value(o[:Ftot_y]) * t_f_v
    dyn_viol_z = vz_dot - (value(o[:Ftot_z]) .- g) * t_f_v

    plot!(f,t_,dyn_viol_x,subplot=8,xlabel="time",color=:blue,label="x",legend=leg_pos)
    plot!(f,t_,dyn_viol_y,subplot=8,color=:red,label="y",legend=leg_pos)
    plot!(f,t_,dyn_viol_z,subplot=8,color=:green,label="z",legend=leg_pos)

    return f
end

"""
    plot_rebula_reference_comparison(model3d, model2d)

Compare one 3-D solution directly with a solved `point_mass_walker()` reference.
The four panels deliberately expose the quantities that caught the pathological
3-D solution: COM height, vertical velocity, individual/total vertical GRF, and
fore-aft velocity. The supplied models must already be solved.
"""
function plot_rebula_reference_comparison(model3d, model2d)
    _load_plots_3d!()
    a, b = object_dictionary(model3d), object_dictionary(model2d)
    ta = value(a[:τ]) .* value(a[:t_f]); tb = value(b[:τ]) .* value(b[:t_f])
    f = plot(layout=(2,2), size=(950,700))
    plot!(f,ta,value(a[:pz]),subplot=1,label="3-D",ylabel="COM height z")
    plot!(f,tb,value(b[:pz]),subplot=1,label="2-D reference",linestyle=:dash)
    plot!(f,ta,value(a[:vz]),subplot=2,label="3-D",ylabel="vertical velocity vz")
    plot!(f,tb,value(b[:vz]),subplot=2,label="2-D reference",linestyle=:dash)
    plot!(f,ta,value(a[:Ftrail_z]),subplot=3,label="3-D trail",ylabel="vertical GRF / BW",xlabel="time")
    plot!(f,ta,value(a[:Flead_z]),subplot=3,label="3-D lead")
    plot!(f,ta,value(a[:Ftot_z]),subplot=3,label="3-D total",linewidth=3)
    plot!(f,tb,value(b[:Ftot_z]),subplot=3,label="2-D total",linestyle=:dash,linewidth=2)
    plot!(f,ta,value(a[:vy]),subplot=4,label="3-D",ylabel="forward velocity vy",xlabel="time")
    plot!(f,tb,value(b[:vy]),subplot=4,label="2-D reference",linestyle=:dash)
    return f
end

"""
    check_energy_balance(model)

Verify energy conservation in the 3D walking model.

Computes:
- Total mechanical work done by leg forces
- Change in kinetic energy
- Change in potential energy
- Energy balance error

Returns a NamedTuple with all energy components.
"""
function check_energy_balance(model)
    o = object_dictionary(model)
    t_ = value(o[:τ]) * value(o[:t_f])
    dt = diff(t_)

    # Velocities
    vx_v = value(o[:vx])
    vy_v = value(o[:vy])
    vz_v = value(o[:vz])
    pz_v = value(o[:pz])

    # Kinetic energy: KE = 0.5 * (vx^2 + vy^2 + vz^2)
    KE = 0.5 .* (vx_v.^2 .+ vy_v.^2 .+ vz_v.^2)
    KE_initial = KE[1]
    KE_final = KE[end]
    ΔKE = KE_final - KE_initial

    # Potential energy uses vertical z.
    PE = g .* pz_v
    PE_initial = PE[1]
    PE_final = PE[end]
    ΔPE = PE_final - PE_initial

    # Mechanical power and work
    power_trail = value(o[:mechpower_trail])
    power_lead = value(o[:mechpower_lead])
    total_power = power_trail .+ power_lead

    # Integrate power to get work (trapezoidal rule)
    work_total = sum((total_power[1:end-1] .+ total_power[2:end]) ./ 2 .* dt)

    # Positive work (energy added to system)
    pos_power = max.(total_power, 0)
    work_positive = sum((pos_power[1:end-1] .+ pos_power[2:end]) ./ 2 .* dt)

    # Negative work (energy removed from system)
    neg_power = min.(total_power, 0)
    work_negative = sum((neg_power[1:end-1] .+ neg_power[2:end]) ./ 2 .* dt)

    # Energy balance: Work = ΔKE + ΔPE
    energy_in = work_positive
    energy_out = -work_negative
    ΔE_mechanical = ΔKE + ΔPE
    balance_error = work_total - ΔE_mechanical

    # Cost work from optimization (uses slack variables)
    cost_work = value(o[:cost_work])

    return (
        KE_initial = KE_initial,
        KE_final = KE_final,
        ΔKE = ΔKE,
        PE_initial = PE_initial,
        PE_final = PE_final,
        ΔPE = ΔPE,
        work_positive = work_positive,
        work_negative = work_negative,
        work_total = work_total,
        ΔE_mechanical = ΔE_mechanical,
        balance_error = balance_error,
        cost_work = cost_work,
        KE = KE,
        PE = PE,
        total_power = total_power,
        time = t_
    )
end

"""
    plot_energy_balance(energy)

Create a visualization of energy balance from check_energy_balance output.
"""
function plot_energy_balance(energy)
    _load_plots_3d!()
    t_ = energy.time

    # 1. Kinetic and Potential Energy over time (initialize figure with first plot)
    f = plot(t_, energy.KE, layout=(2, 2), size=(900, 700),
             subplot=1, label="KE", ylabel="Energy", title="Energy Components", linewidth=2)
    plot!(f, t_, energy.PE, subplot=1, label="PE", linewidth=2)
    plot!(f, t_, energy.KE .+ energy.PE, subplot=1, label="Total (KE+PE)", linewidth=2, linestyle=:dash)

    # 2. Power over time
    plot!(f, t_, energy.total_power, subplot=2, label="Total Power", ylabel="Power",
          title="Mechanical Power", linewidth=2)
    hline!(f, [0], subplot=2, label="", color=:black, linestyle=:dash)

    # 3. Cumulative work
    dt = diff(t_)
    interval_work=(energy.total_power[1:end-1].+energy.total_power[2:end])./2 .* dt
    cumulative_work = vcat(0.0,cumsum(interval_work))
    plot!(f, t_, cumulative_work, subplot=3, label="Cumulative Work", ylabel="Work",
          title="Cumulative Work Done", linewidth=2)

    # 4. Energy balance summary (bar chart)
    labels = ["ΔKE", "ΔPE", "Work", "Error"]
    values = [energy.ΔKE, energy.ΔPE, energy.work_total, energy.balance_error]
    bar!(f, labels, values, subplot=4, title="Energy Balance Summary", ylabel="Energy", legend=false)

    return f
end

"""
    run_step_width_sweep(widths=0.0:0.025:0.20; step_speed=0.4, kwargs...)

Solve otherwise identical dimensionless walking problems over prescribed step
widths. `step_speed` is fixed forward overground speed, so every entry uses
`t_f = step_length / step_speed`. The default endpoint geometry
places the COM halfway between the midline and positive-x stance foot initially,
then at its negative mirror finally.

Returns a named tuple containing the models, solver status, objective components,
step times, and simple endpoint residuals. Failed solves are retained and their
numeric results are `NaN`, which makes plotting and batch diagnosis straightforward.
"""
function run_step_width_sweep(widths=0.0:0.025:0.20;
                              step_speed=0.4, step_length=nothing,
                              z0_fixed=0.98, kwargs...)
    width_values = Float64.(collect(widths))
    isempty(width_values) && throw(ArgumentError("widths cannot be empty"))
    any(width_values .< 0) && throw(ArgumentError("step widths must be nonnegative"))

    n = length(width_values)
    models = Vector{Any}(undef, n)
    status = Vector{Any}(undef, n)
    converged = falses(n)
    objective = fill(NaN, n)
    work = fill(NaN, n)
    force_rate = fill(NaN, n)
    step_times = fill(NaN, n)
    endpoint_residual = fill(NaN, n)

    # Keep this calculation identical to point_mass_walker_3d's default.
    sl = isnothing(step_length) ? 2 * 0.95 * sin(0.35) : step_length
    for (i, width) in pairs(width_values)
        try
            model = point_mass_walker_3d(; step_width=width, step_speed=step_speed,
                                         step_length=step_length,
                                         z0_fixed=z0_fixed, kwargs...)
            models[i] = model
            status[i] = termination_status(model)
            converged[i] = status[i] in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL)
            if converged[i]
                o = object_dictionary(model)
                objective[i] = objective_value(model)
                work[i] = value(o[:cost_work])
                force_rate[i] = value(o[:cost_fr2])
                step_times[i] = value(o[:t_f])
                px_start, px_end = value(o[:px](0)), value(o[:px](1))
                pz_start, pz_end = value(o[:pz](0)), value(o[:pz](1))
                endpoint_residual[i] = max(abs(px_start-width/4),abs(px_end+width/4),
                                           abs(pz_start-z0_fixed),abs(pz_end-z0_fixed))
            end
        catch err
            models[i] = nothing
            status[i] = err
        end
        isnan(step_times[i]) && (step_times[i] = sl / step_speed)
    end

    return (; widths=width_values, step_speed, step_length=sl, step_times,
            models, status, converged, objective, work, force_rate,
            endpoint_residual)
end

"""Plot objective and work for the converged members of a width sweep."""
function plot_step_width_sweep(results)
    _load_plots_3d!()
    mask = results.converged
    f = plot(results.widths[mask], results.objective[mask]; marker=:circle,
             xlabel="dimensionless step width", ylabel="dimensionless cost",
             label="objective", title="Fixed-speed step-width sweep")
    plot!(f, results.widths[mask], results.work[mask]; marker=:circle, label="work")
    return f
end

"""
    run_pz_sweep(; pz_range=0.0:0.005:0.05, kwargs...)

Sweep over initial COM lateral positions (pz_initial) to investigate whether
zero z-excursion is most economical.

# Arguments
- `pz_range`: Range of pz_initial values to test (default: 0.0:0.005:0.05, i.e., 0 to 5cm)
- `kwargs...`: Additional keyword arguments passed to `point_mass_walker_3d()`

# Returns
- `results::NamedTuple`: Contains:
  - `pz_values`: Array of pz_initial values tested
  - `costs`: Array of total objective costs
  - `cost_work`: Array of mechanical work costs
  - `cost_fr`: Array of force rate costs
  - `models`: Array of solved models for inspection
  - `converged`: Boolean array indicating which models converged

# Example
```julia
results = run_pz_sweep()
plot_pz_sweep(results)

# With custom parameters
results = run_pz_sweep(pz_range=0.0:0.01:0.03, c_fr=0.1)
```
"""
function run_pz_sweep(; pz_range=0.0:0.005:0.05, kwargs...)
    pz_values = collect(pz_range)
    n = length(pz_values)

    costs = zeros(n)
    cost_work = zeros(n)
    cost_fr = zeros(n)
    converged = falses(n)
    models = Vector{Any}(undef, n)

    println("Running pz_initial sweep: $(pz_values[1]) to $(pz_values[end]) ($(n) points)")

    for (i, pz) in enumerate(pz_values)
        print("  pz_initial = $(round(pz, digits=4))... ")
        try
            m = point_mass_walker_3d(; px_initial=pz, kwargs...)
            models[i] = m
            o = object_dictionary(m)

            if termination_status(m) in [MOI.LOCALLY_SOLVED, MOI.OPTIMAL]
                converged[i] = true
                costs[i] = objective_value(m)
                cost_work[i] = value(o[:cost_work])
                cost_fr[i] = value(o[:cost_fr])
                println("converged, cost = $(round(costs[i], digits=4))")
            else
                println("did not converge: $(termination_status(m))")
                costs[i] = NaN
                cost_work[i] = NaN
                cost_fr[i] = NaN
            end
        catch e
            println("error: $e")
            costs[i] = NaN
            cost_work[i] = NaN
            cost_fr[i] = NaN
        end
    end

    return (
        pz_values = pz_values,
        costs = costs,
        cost_work = cost_work,
        cost_fr = cost_fr,
        models = models,
        converged = converged
    )
end

"""
    plot_pz_sweep(results)

Plot the results of a pz_initial sweep.

Creates a figure showing:
1. Total cost vs pz_initial
2. Work cost vs pz_initial
3. Force rate cost vs pz_initial
4. Cost breakdown bar chart for converged solutions

# Arguments
- `results`: Output from `run_pz_sweep()`

# Returns
- `f::Plot`: The plot figure
"""
function plot_pz_sweep(results)
    _load_plots_3d!()
    pz = results.pz_values .* 100  # Convert to cm for display

    # Filter converged results
    mask = results.converged
    pz_conv = pz[mask]
    costs_conv = results.costs[mask]
    work_conv = results.cost_work[mask]
    fr_conv = results.cost_fr[mask]

    # 1. Total cost vs pz_initial (initialize figure with first plot)
    f = plot(pz_conv, costs_conv, layout=(2, 2), size=(900, 700),
             subplot=1, label="Total Cost", ylabel="Cost",
             xlabel="Initial COM-Z (cm)", title="Total Cost vs Initial Lateral Position",
             linewidth=2, marker=:circle, markersize=5)

    # Find minimum
    if length(costs_conv) > 0
        min_idx = argmin(costs_conv)
        scatter!(f, [pz_conv[min_idx]], [costs_conv[min_idx]], subplot=1,
                 label="Min at pz=$(round(pz_conv[min_idx], digits=2))cm", markersize=10, color=:red)
    end

    # 2. Work cost vs pz_initial
    plot!(f, pz_conv, work_conv, subplot=2, label="Work Cost", ylabel="Cost",
          xlabel="Initial COM-Z (cm)", title="Mechanical Work vs Initial Lateral Position",
          linewidth=2, marker=:circle, markersize=5, color=:blue)

    # 3. Force rate cost vs pz_initial
    plot!(f, pz_conv, fr_conv, subplot=3, label="Force Rate Cost", ylabel="Cost",
          xlabel="Initial COM-Z (cm)", title="Force Rate Cost vs Initial Lateral Position",
          linewidth=2, marker=:circle, markersize=5, color=:orange)

    # 4. Stacked comparison
    if length(costs_conv) > 0
        plot!(f, pz_conv, work_conv, subplot=4, label="Work", ylabel="Cost",
              xlabel="Initial COM-Z (cm)", title="Cost Breakdown",
              linewidth=2, fillrange=0, fillalpha=0.3, color=:blue)
        plot!(f, pz_conv, work_conv .+ fr_conv, subplot=4, label="Work + FR",
              linewidth=2, fillrange=work_conv, fillalpha=0.3, color=:orange)
    end

    return f
end

# Run the demonstration only when this file is executed, not when it is included
# by a notebook or test.
if abspath(PROGRAM_FILE) == @__FILE__
println("=" ^ 60)
println("Running 3D Point Mass Walking Model")
println("=" ^ 60)

model = point_mass_walker_3d()

println("\n" * "=" ^ 60)
println("Results")
println("=" ^ 60)

o = object_dictionary(model)
println("Termination status: ", termination_status(model))
println("Objective value: ", round(objective_value(model), digits=4))
println("Cost work: ", round(value(o[:cost_work]), digits=4))
println("Cost force rate: ", round(value(o[:cost_fr]), digits=4))
println("Step time (t_f): ", round(value(o[:t_f]), digits=4))

# Check energy balance
println("\n" * "=" ^ 60)
println("Energy Balance Check")
println("=" ^ 60)

energy = check_energy_balance(model)
println("Initial KE: ", round(energy.KE_initial, digits=4))
println("Final KE: ", round(energy.KE_final, digits=4))
println("ΔKE: ", round(energy.ΔKE, digits=4))
println("Initial PE: ", round(energy.PE_initial, digits=4))
println("Final PE: ", round(energy.PE_final, digits=4))
println("ΔPE: ", round(energy.ΔPE, digits=4))
println("Work (positive): ", round(energy.work_positive, digits=4))
println("Work (negative): ", round(energy.work_negative, digits=4))
println("Work (net): ", round(energy.work_total, digits=4))
println("ΔE mechanical (ΔKE + ΔPE): ", round(energy.ΔE_mechanical, digits=4))
println("Energy balance error: ", round(energy.balance_error, digits=6))
println("Cost work (from optimization): ", round(energy.cost_work, digits=4))

# Create plots
f_traj = plot_results_3d(model)
f_energy = plot_energy_balance(energy)

println("\n" * "=" ^ 60)
println("Plots created: f_traj (trajectories), f_energy (energy balance)")
println("=" ^ 60)
end
