# 3D Point Mass Walking Model with Two Force Actuators acting along the leg.
# Extension of the 2D model to include lateral (z) dimension.
# if you want to run it without
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
# using Revise
using InfiniteOpt, Ipopt, Plots, Distributions, LinearAlgebra

const g = 1  # Gravity (acts in -y direction only)

"""
    point_mass_walker_3d(; c_fr=0.05, c_fr2=0.01, step_length=nothing, step_speed=nothing,
                          step_width=0.1, cost_type=:linear)

Solve the 3D point mass walking optimization problem using InfiniteOpt.

Models a point mass with two leg force actuators (trailing and leading legs) that push
along the leg direction in 3D space. Minimizes a cost combining mechanical work and
force rate penalties.

# Coordinate System
- x: forward direction (direction of travel)
- y: vertical direction (gravity acts in -y)
- z: lateral direction (side-to-side)

# Arguments
- `c_fr::Float64`: Linear force rate penalty coefficient (default: 0.05)
- `c_fr2::Float64`: Squared force rate penalty coefficient (default: 0.01)
- `step_length`: Step length in x-direction (default: 2*y_0*sin(α) ≈ 0.65)
- `step_speed`: Step speed in m/s; determines step time as t_f = step_length/step_speed
                (default: step_length/1.2)
- `step_width`: Lateral distance between feet in z-direction (default: 0.1)
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
value(o[:px])            # Position x trajectory
value(o[:pz])            # Position z trajectory (lateral)
```
"""
function point_mass_walker_3d(;
        c_fr = 0.05,
        c_fr2 = 0.01,
        step_length = nothing,
        step_speed = nothing,
        step_width = 0.1,      # Lateral foot placement width
        cost_type = :linear
    )
    ## 3D Point Mass Walking Model
    model = InfiniteModel(Ipopt.Optimizer)

    # Model parameters
    c_t = 5.0    # Time penalty coefficient
    @finite_parameter(model, y_0 == 0.95)  # Initial leg length / COM height
    @finite_parameter(model, α == 0.35)     # Leg angle parameter

    # Step length: use provided value or default
    sl = isnothing(step_length) ? 2*parameter_value(y_0)*sin(parameter_value(α)) : step_length

    # Step time: derive from step_speed if provided, otherwise default to 1.2
    t_f_val = isnothing(step_speed) ? 1.2 : sl / step_speed

    # Infinite time parameter
    @infinite_parameter(model, τ ∈ [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))

    # State variables with bounds and initial guesses
    # x: forward, y: vertical, z: lateral
    @variable(model, px, Infinite(τ), start = (t) -> sl*t)  # x position (forward)
    @variable(model, py >= 0.1, Infinite(τ), start = (t) -> parameter_value(y_0) + cos(π*t)*.05)  # y position (vertical)
    @variable(model, pz, Infinite(τ), start = (t) -> 0.0)   # z position (lateral)

    @variable(model, vx, Infinite(τ), start = (t) -> sl/t_f_val)  # x velocity
    @variable(model, vy, Infinite(τ), start = (t) -> 0.0)  # y velocity
    @variable(model, vz, Infinite(τ), start = (t) -> 0.0)  # z velocity

    # Variable: Force magnitude along the leg, and must be positive (legs can only push)
    @variable(model, F_trail >= 0, Infinite(τ),
            start = (t) -> 1.0 * cos(2π*t) * (t < 0.5 ? 1.0 : 0.0))
    @variable(model, F_lead >= 0, Infinite(τ),
            start = (t) -> 1.0 * -sin(2π*t) * (t <= 0.5 ? 0 : 1.0))

    # Time scaling variable
    @variable(model, 0.001 <= t_f <= 10, start = t_f_val)

    # Fixed leg contact positions in 3D
    # Trailing foot at origin, leading foot at (sl, 0, 0)
    # Feet are laterally offset by step_width/2 on each side
    @finite_parameter(model, P_trail_x == 0.0)
    @finite_parameter(model, P_trail_y == 0.0)
    @finite_parameter(model, P_trail_z == -step_width/2)  # Trailing foot on left

    @finite_parameter(model, P_lead_x == sl)
    @finite_parameter(model, P_lead_y == 0.0)
    @finite_parameter(model, P_lead_z == step_width/2)    # Leading foot on right

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
    @constraint(model, ∂(vy, τ) == t_f * (Ftot_y - g))  # Gravity only in y
    @constraint(model, ∂(vz, τ) == t_f * Ftot_z)        # No gravity in z

    # Force rate variables
    @variable(model, Fdot_trail, Infinite(τ))
    @variable(model, Fdot_lead, Infinite(τ))

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
    @constraint(model, F_trail * (trail_leg_length - 1) <= 0)
    @constraint(model, F_lead * (lead_leg_length - 1) <= 0)

    # Boundary conditions
    # Initial conditions
    @constraint(model, px(0) == 0)
    @constraint(model, py(0) == y_0)
    @constraint(model, pz(0) == 0)  # Start at center laterally
    @constraint(model, vy(0) == 0)
    @constraint(model, vz(0) == 0)  # No initial lateral velocity

    # Final conditions (symmetric gait)
    @constraint(model, px(1) == sl)
    @constraint(model, py(1) == parameter_value(y_0))
    @constraint(model, pz(1) == 0)  # Return to center laterally (symmetric)
    @constraint(model, vy(1) == 0)
    @constraint(model, vz(1) == 0)  # No final lateral velocity
    @constraint(model, vx(1) == vx(0))

    # Force boundary conditions
    @constraint(model, F_trail(1) == 0)
    @constraint(model, F_lead(0) == 0)
    @constraint(model, Fdot_lead(0) == Fdot_trail(1))
    @constraint(model, Fdot_trail(0) == Fdot_lead(1))

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
    set_optimizer_attribute(model, "max_cpu_time", 120.0)
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "warm_start_init_point", "yes")

    optimize!(model)
    return model
end

"""
    plot_results_3d(model)

Plot the solution trajectories from a solved 3D walking optimization model.

Creates a multi-subplot figure showing:
1. 3D COM trajectory
2. Top view (x-z plane)
3. Position vs time (px, py, pz)
4. Velocity vs time (vx, vy, vz)
5. Force components vs time
6. Leg lengths vs time
7. Mechanical power vs time
8. Energy balance check
"""
function plot_results_3d(model)
    o = object_dictionary(model)
    t_ = value(o[:τ]) * value(o[:t_f])

    px_v = value(o[:px])
    py_v = value(o[:py])
    pz_v = value(o[:pz])
    vx_v = value(o[:vx])
    vy_v = value(o[:vy])
    vz_v = value(o[:vz])

    # Create figure
    f = plot(layout = (4, 2), size = (1000, 1000))

    txt = "Work: " * string(round(value(o[:cost_work]), digits=3)) *
          " FR: " * string(round(value(o[:cost_fr]), digits=3))

    # 1. 3D trajectory
    plot!(px_v, pz_v, py_v, subplot=1, xlabel="x (forward)", ylabel="z (lateral)", zlabel="y (vertical)",
          title=txt, label="COM", linewidth=2)

    # 2. Top view (x-z plane)
    plot!(px_v, pz_v, subplot=2, xlabel="x (forward)", ylabel="z (lateral)",
          title="Top View", label="COM path", linewidth=2)
    # Add foot positions
    scatter!([value(o[:P_trail_x])], [value(o[:P_trail_z])], subplot=2, label="Trail foot", markersize=8)
    scatter!([value(o[:P_lead_x])], [value(o[:P_lead_z])], subplot=2, label="Lead foot", markersize=8)

    # 3. Position vs time
    plot!(t_, px_v, subplot=3, label="px", ylabel="position", linewidth=2)
    plot!(t_, py_v, subplot=3, label="py", linewidth=2)
    plot!(t_, pz_v, subplot=3, label="pz", linewidth=2)

    # 4. Velocity vs time
    plot!(t_, vx_v, subplot=4, label="vx", ylabel="velocity", linewidth=2)
    plot!(t_, vy_v, subplot=4, label="vy", linewidth=2)
    plot!(t_, vz_v, subplot=4, label="vz", linewidth=2)

    # 5. Force components
    plot!(t_, value(o[:Ftot_x]), subplot=5, label="Fx", ylabel="force", linewidth=2)
    plot!(t_, value(o[:Ftot_y]), subplot=5, label="Fy", linewidth=2)
    plot!(t_, value(o[:Ftot_z]), subplot=5, label="Fz", linewidth=2)

    # 6. Leg lengths
    plot!(t_, value(o[:trail_leg_length]), subplot=6, label="trail", ylabel="leg length", linewidth=2)
    plot!(t_, value(o[:lead_leg_length]), subplot=6, label="lead", linewidth=2)
    hline!([1.0], subplot=6, label="max length", linestyle=:dash, color=:red)

    # 7. Mechanical power
    plot!(t_, value(o[:mechpower_trail]), subplot=7, label="trail power", ylabel="power", linewidth=2)
    plot!(t_, value(o[:mechpower_lead]), subplot=7, label="lead power", linewidth=2)

    # 8. Dynamics violation check
    vx_dot = value(∂(o[:vx], o[:τ]))
    vy_dot = value(∂(o[:vy], o[:τ]))
    vz_dot = value(∂(o[:vz], o[:τ]))
    t_f_v = value(o[:t_f])

    dyn_viol_x = vx_dot - value(o[:Ftot_x]) * t_f_v
    dyn_viol_y = vy_dot - (value(o[:Ftot_y]) .- g) * t_f_v
    dyn_viol_z = vz_dot - value(o[:Ftot_z]) * t_f_v

    plot!(t_, dyn_viol_x, subplot=8, label="x viol", ylabel="dynamics violation", xlabel="time", linewidth=2)
    plot!(t_, dyn_viol_y, subplot=8, label="y viol", linewidth=2)
    plot!(t_, dyn_viol_z, subplot=8, label="z viol", linewidth=2)

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
    dt = t_[2] - t_[1]

    # Velocities
    vx_v = value(o[:vx])
    vy_v = value(o[:vy])
    vz_v = value(o[:vz])
    py_v = value(o[:py])

    # Kinetic energy: KE = 0.5 * (vx^2 + vy^2 + vz^2)
    KE = 0.5 .* (vx_v.^2 .+ vy_v.^2 .+ vz_v.^2)
    KE_initial = KE[1]
    KE_final = KE[end]
    ΔKE = KE_final - KE_initial

    # Potential energy: PE = g * py
    PE = g .* py_v
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
    t_ = energy.time

    f = plot(layout=(2, 2), size=(900, 700))

    # 1. Kinetic and Potential Energy over time
    plot!(t_, energy.KE, subplot=1, label="KE", ylabel="Energy", title="Energy Components", linewidth=2)
    plot!(t_, energy.PE, subplot=1, label="PE", linewidth=2)
    plot!(t_, energy.KE .+ energy.PE, subplot=1, label="Total (KE+PE)", linewidth=2, linestyle=:dash)

    # 2. Power over time
    plot!(t_, energy.total_power, subplot=2, label="Total Power", ylabel="Power",
          title="Mechanical Power", linewidth=2)
    hline!([0], subplot=2, label="", color=:black, linestyle=:dash)

    # 3. Cumulative work
    dt = t_[2] - t_[1]
    cumulative_work = cumsum(energy.total_power) .* dt
    plot!(t_, cumulative_work, subplot=3, label="Cumulative Work", ylabel="Work",
          title="Cumulative Work Done", linewidth=2)

    # 4. Energy balance summary (bar chart)
    labels = ["ΔKE", "ΔPE", "Work", "Error"]
    values = [energy.ΔKE, energy.ΔPE, energy.work_total, energy.balance_error]
    bar!(labels, values, subplot=4, title="Energy Balance Summary", ylabel="Energy", legend=false)

    return f
end

# Run the 3D model
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
