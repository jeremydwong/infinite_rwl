# 2D Point Mass Walking Model with Two Force Actuators acting along the leg.
# if you want to run it without 
# using Pkg
# Pkg.activate(".")
# Pkg.instantiate()
# using Revise
using InfiniteOpt, Ipopt, Distributions, LinearAlgebra, Plots

_load_plots_2d!() = isdefined(@__MODULE__, :Plots) || (@eval using Plots; true)

const g = 1  # Gravity

"""
    point_mass_walker(; c_fr=0.05, c_fr2=0.01, y0_fixed = 0.98,step_length=nothing, step_speed=nothing, cost_type=:linear)

Solve the 2D point mass walking optimization problem using InfiniteOpt.

Models a point mass with two leg force actuators (trailing and leading legs) that push
along the leg direction. Minimizes a cost combining mechanical work and force rate penalties.

# Arguments
- `c_fr::Float64`: Linear force rate penalty coefficient (default: 0.05)
- `c_fr2::Float64`: Squared force rate penalty coefficient (default: 0.01)
- `y0_fixed::Float64`: Fixed initial height (default: 0.98); if you wish to unconstrain, set to `nothing`
- `step_length`: Step length in meters (default: 2*y_0*sin(α) ≈ 0.65)
- `step_speed`: Step speed in m/s; determines step time as t_f = step_length/step_speed
                (default: step_length/1.2)
- `cost_type::Symbol`: Objective function type:
  - `:linear` → minimize work + c_fr * ∫|F̈|dt
  - `:squared` → minimize work + c_fr2 * ∫F̈²dt

# Returns
- `model::InfiniteModel`: The solved optimization model

# Accessing Results
After solving, use `object_dictionary(model)` to access named variables and expressions:
```julia
model = point_mass_walker()
o = object_dictionary(model)
value(o[:cost_work])     # Mechanical work cost
value(o[:px])            # Position x trajectory (array)
value(o[:F_trail])       # Trailing leg force trajectory
termination_status(model) # Check solver status
objective_value(model)    # Total objective value
```

# Example
```julia
# Default parameters
model = point_mass_walker()

# Custom step with squared force rate penalty
model = point_mass_walker(step_length=0.5, step_speed=0.6, cost_type=:squared, c_fr2=0.02)
```
"""
function point_mass_walker(;
        c_fr = 0.05,
        c_fr2 = 0.01,
        y0_fixed = 0.98,
        step_length = nothing,
        step_speed = nothing,
        cost_type = :squared
    )
    ## 2D Point Mass Walking Model
    model = InfiniteModel(Ipopt.Optimizer)

    # Model parameters
    c_t = 5.0    # Time penalty coefficient
    k_b = 0.0    # Damping coefficient
    ω0 = 0.3   # angular velocity 0
    # α = 0.35
    @finite_parameter(model,y_0==0.95)  # Initial leg length
    @finite_parameter(model,α==0.35)

    # Step length: use provided value or default
    sl = isnothing(step_length) ? 2*parameter_value(y_0)*sin(parameter_value(α)) : step_length

    # Step time: derive from step_speed if provided, otherwise default to 1.2
    t_f_val = isnothing(step_speed) ? 1.2 : sl / step_speed

    # Infinite time parameter
    @infinite_parameter(model, τ ∈ [0, 1], num_supports=101, derivative_method = OrthogonalCollocation(2))

    # State variables with bounds and initial guesses
    @variable(model, px, Infinite(τ), start = (t) -> sl*t)  # x position
    @variable(model, py >= 0.1, Infinite(τ), start = (t) -> parameter_value(y_0) + cos(π*t)*.1)  # y position (must be positive)
    @variable(model, vx, Infinite(τ), start = (t) -> 1.0)  # x velocity
    @variable(model, vy, Infinite(τ), start = (t) -> 0.0)  # y velocity

    # Variable: Force magnitude along the leg, and must be positive (legs can only push)
    # Initial guesses for smooth bell-shaped force profiles
    @variable(model, F_trail >= 0, Infinite(τ), 
            start = (t) -> 1.0 * cos(2π*t) * (t < 0.5 ? 1.0 : 0.0))  # trailing leg force magnitude
    @variable(model, F_lead >= 0, Infinite(τ), 
            start = (t) -> 1.0 * -sin(2π*t) * (t <= 0.5 ? 0 : 1.0))  # leading leg force magnitude

    # Time scaling variable
    @variable(model, 0.001 <= t_f <= 10, start = 1)

    # Fixed leg contact positions; contact points are currently [y=0] beginning and end.
    @finite_parameter(model, P_trail_x==0.0)  # Trailing leg x-position
    @finite_parameter(model, P_trail_y==0.0)  # Trailing leg y-position
    @finite_parameter(model, P_lead_x==sl)  # Leading leg x-position
    @finite_parameter(model, P_lead_y==0.0)   # Leading leg y-position

    # Leg vectors (from contact points to COM)
    @expression(model, trail_leg_x, px - P_trail_x)
    @expression(model, trail_leg_y, py - P_trail_y)
    @expression(model, lead_leg_x, px - P_lead_x)
    @expression(model, lead_leg_y, py - P_lead_y)

    # Leg lengths
    @expression(model, trail_leg_length, sqrt(trail_leg_x^2 + trail_leg_y^2))
    @expression(model, lead_leg_length, sqrt(lead_leg_x^2 + lead_leg_y^2))

    # Unit vectors along each leg (with small epsilon to avoid division by zero)
    @expression(model, trail_unit_x, trail_leg_x / (trail_leg_length))
    @expression(model, trail_unit_y, trail_leg_y / (trail_leg_length))
    @expression(model, lead_unit_x, lead_leg_x / (lead_leg_length))
    @expression(model, lead_unit_y, lead_leg_y / (lead_leg_length))

    # Force components for each leg
    @expression(model, Ftrail_x, F_trail * trail_unit_x)
    @expression(model, Ftrail_y, F_trail * trail_unit_y)
    @expression(model, Flead_x,  F_lead * lead_unit_x)
    @expression(model, Flead_y,  F_lead * lead_unit_y)

    # Total force components
    @expression(model, Ftot_x, Ftrail_x + Flead_x)
    @expression(model, Ftot_y, Ftrail_y + Flead_y)  # Excluding gravity

    # trail_leg_velocity
    @expression(model, trail_leg_velocity, 
            (px*vx + py*vy) / (trail_leg_length))

    # lead_leg_velocity
    @expression(model, lead_leg_velocity, 
                (-P_lead_x*vx + px*vx -P_lead_y*vy + py*vy) / (lead_leg_length))

    # Step 2: Compute mechanical power as force magnitude times leg-lengthening velocity
    # Note that positive power occurs when the leg is extending (trail_leg_velocity > 0)
    # Power slack variables for each leg
    @variable(model, pospower_trail >= 0, Infinite(τ))
    @variable(model, negpower_trail <= 0, Infinite(τ))
    @variable(model, pospower_lead >= 0, Infinite(τ))
    @variable(model, negpower_lead <= 0, Infinite(τ))

    @expression(model, mechpower_trail, F_trail * trail_leg_velocity)
    @expression(model, mechpower_lead, F_lead * lead_leg_velocity)

    # Step 3: Update the power splitting constraints
    # Power is positive when doing work (leg extending)
    @constraint(model, pospower_trail >= mechpower_trail)
    @constraint(model, pospower_trail >= 0)
    @constraint(model, negpower_trail <= mechpower_trail)
    @constraint(model, negpower_trail <= 0)

    @constraint(model, pospower_lead >= mechpower_lead)
    @constraint(model, pospower_lead >= 0)
    @constraint(model, negpower_lead <= mechpower_lead)
    @constraint(model, negpower_lead <= 0)

    # Step 4: System dynamics
    @constraint(model, ∂(px, τ) == t_f * vx)
    @constraint(model, ∂(py, τ) == t_f * vy)
    @constraint(model, ∂(vx, τ) == t_f * (Ftot_x))
    @constraint(model, ∂(vy, τ) == t_f * (Ftot_y - g))

    # Step 5: Force rate. Force dot variables
    ### begin
    @variable(model, Fdot_trail, Infinite(τ))
    @variable(model, Fdot_lead, Infinite(τ))

    # Force dot dot variables (split into positive and negative components)
    @variable(model, Fddot_trail_p >= 0, Infinite(τ))
    @variable(model, Fddot_trail_m >= 0, Infinite(τ))
    @variable(model, Fddot_lead_p >= 0, Infinite(τ))
    @variable(model, Fddot_lead_m >= 0, Infinite(τ))

    # Update force dynamics with scaling
    @variable(model, fdot_scale == 1)
    @variable(model, fddot_scale == 1)
    @constraint(model, ∂(F_trail, τ) == t_f * (Fdot_trail)/fdot_scale)
    @constraint(model, ∂(F_lead, τ) == t_f * (Fdot_lead)/fdot_scale)
    @constraint(model, ∂(Fdot_trail, τ) == t_f * ((Fddot_trail_p - Fddot_trail_m)/fddot_scale))
    @constraint(model, ∂(Fdot_lead, τ) == t_f * ((Fddot_lead_p - Fddot_lead_m)/fddot_scale))
    # have not needed the following two complimentarity constraints:
    # @constraint(model, Fddot_trail_p * Fddot_trail_m <= 1e-6) # complimentarity constraint
    # @constraint(model, Fddot_lead_p * Fddot_lead_m <= 1e-6)   # complimentarity constraint

    # Leg length constraints (forces only active when leg length <= 1)
    # Complementarity: F*(length-1) <= 0 means when length>1, F must be <=0.
    # Combined with F >= 0 bound, this forces F=0 when length>1.
    @constraint(model, F_trail * (trail_leg_length - 1) <= 0)
    @constraint(model, F_lead * (lead_leg_length - 1) <= 0)

    # Step 6: box constraints on initial/final states.
    # Initial and final boundary conditions for a complete step
    # Initial conditions
    @constraint(model, px(0) == 0)
    
    #check if y0_fixed is set to empty, in which case do not constrain it
    if !isnothing(y0_fixed)
        @constraint(model, py(0) == y0_fixed)  # Initial height (fixed)
    end
    # if py(0) is left free, then the optimizer to find optimal initial COM height (initial guess is y_0)
    # @constraint(model, vx(0) == y_0 * ω0)
    # Note: vy(0) is left free for optimizer to find optimal initial vertical velocity.
    # If solution finding becomes problematic, reconsider adding: @constraint(model, vy(0) == 0)

    # Final conditions (symmetric gait)
    @constraint(model, px(1) == sl)  # One full step
    @constraint(model, py(1) == py(0))  # Same height at end as start (symmetric)
    @constraint(model, vx(1) == vx(0))  # Same horizontal velocity
    # note to self: Y here IS SUPPOSED TO BE THE SAME AT BEGINNING AND END, 
    # BECAUSE HERE Y IS HEIGHT.
    # IN OUR 3D MODEL, Y IS LATERAL POSITION, AND IT IS NOT SUPPOSED TO BE THE SAME AT BEGINNING AND END.
    @constraint(model, vy(0) == 0)  # Symmetric vertical velocity: going down at start = going up at end
    @constraint(model, vy(1) == 0)  # Symmetric vertical velocity: going down at start = going up at end
    
    
    # Force boundary conditions
    @constraint(model, F_trail(0) == 1)
    @constraint(model, F_lead(0) == 0)
    @constraint(model, F_trail(1) == 0)
    @constraint(model, F_lead(1) == 1)

    @constraint(model, Fdot_lead(0) == -Fdot_trail(1))
    @constraint(model, Fdot_trail(0) == -Fdot_lead(1))

    @constraint(model, t_f == t_f_val)

    # Objective function: minimize work, force rate, and time. 
    # they only get added to the objective 
    @expression(model, cost_work,integral(pospower_trail, τ) * t_f + integral(pospower_lead, τ) * t_f - 
            integral(negpower_trail, τ) * t_f - integral(negpower_lead, τ) * t_f)

    @expression(model, cost_fr, c_fr*integral(Fddot_trail_p, τ) * t_f +c_fr*integral(Fddot_lead_p,τ)*t_f + 
    c_fr*integral(Fddot_trail_m,τ) * t_f + c_fr*integral(Fddot_lead_m,τ)*t_f)

    @expression(model, cost_fr2, c_fr2*integral(Fddot_trail_p^2, τ) * t_f + c_fr2*integral(Fddot_lead_p^2, τ)*t_f +
        c_fr2*integral(Fddot_trail_m^2, τ) * t_f + c_fr2*integral(Fddot_lead_m^2, τ)*t_f)

    @expression(model, cost_time, c_t*t_f)

    # Set objective based on cost_type
    if cost_type == :linear
        @objective(model, Min, cost_work + cost_fr)
    elseif cost_type == :squared
        @objective(model, Min, cost_work + cost_fr2)
    else
        error("cost_type must be :linear or :squared")
    end

    # Set solver parameters
    set_optimizer_attribute(model, "max_cpu_time", 120.0)
    set_optimizer_attribute(model, "tol", 1e-3)
    set_optimizer_attribute(model, "max_iter", 500)
    set_optimizer_attribute(model, "print_level", 0)
    # set_optimizer_attribute(model, "nlp_scaling_method", "gradient-based")
    set_optimizer_attribute(model, "warm_start_init_point", "yes")

    # Solve the model
    optimize!(model)
    return model
end

"""
    plot_results(model)

Plot the solution trajectories from a solved walking optimization model.

Creates an 8-subplot figure showing:
1. COM trajectory (x vs y)
2. Position vs time (px, py)
3. Velocity vs time (vx, vy)
4. Force components vs time (Ftrail_x, Ftrail_y, Flead_x, Flead_y, total FY)
5. Leg lengths vs time
6. Force rate (F̈) vs time
7. Leg lengthening velocity vs time
8. Dynamics violation (should be near zero)

# Arguments
- `model::InfiniteModel`: A solved model from `point_mass_walker()`

# Returns
- `f`: The Plots.jl figure object

# Example
```julia
model = point_mass_walker()
f = plot_results(model)
savefig(f, "walking_solution.png")
```
"""
function plot_results(model; show_legend=true)
    _load_plots_2d!()
    o = object_dictionary(model)
    t_ = value(o[:τ])*value(o[:t_f])

    # Print boundary condition diagnostics
    px_v = value(o[:px])
    py_v = value(o[:py])
    vx_v = value(o[:vx])
    vy_v = value(o[:vy])
    println("Boundary conditions check:")
    println("  px(0)=$(round(px_v[1], digits=4)), px(1)=$(round(px_v[end], digits=4))")
    println("  py(0)=$(round(py_v[1], digits=4)), py(1)=$(round(py_v[end], digits=4))")
    println("  vx(0)=$(round(vx_v[1], digits=4)), vx(1)=$(round(vx_v[end], digits=4))")
    println("  vy(0)=$(round(vy_v[1], digits=4)), vy(1)=$(round(vy_v[end], digits=4))")

    txt="work:"*string(round(value(o[:cost_work]),digits=2)) * " FR2:" * string(round(value(o[:cost_fr2]),digits=2))
    leg_pos = show_legend ? :topleft : false

    # Get foot positions
    P_trail_x = parameter_value(o[:P_trail_x])
    P_trail_y = parameter_value(o[:P_trail_y])
    P_lead_x = parameter_value(o[:P_lead_x])
    P_lead_y = parameter_value(o[:P_lead_y])

    # Get leg lengths for filtering
    trail_len = value(o[:trail_leg_length])
    lead_len = value(o[:lead_leg_length])

    # Initialize plot with layout - COM trajectory
    f = plot(px_v, py_v, layout=(4,2), size=(800, 800),
             subplot=1, xlabel="x", ylabel="y", title=txt, label="COM", legend=leg_pos,
             aspect_ratio=:equal)

    # Add start (green) and end (red) dots
    scatter!(f, [px_v[1]], [py_v[1]], subplot=1, color=:green, markersize=8, label="start")
    scatter!(f, [px_v[end]], [py_v[end]], subplot=1, color=:red, markersize=8, label="end")

    # Draw leg lines every 10 nodes, only if leg length <= 1
    for i in 1:10:length(px_v)
        # Trail leg
        if trail_len[i] <= 1.0
            plot!(f, [P_trail_x, px_v[i]], [P_trail_y, py_v[i]], subplot=1,
                  color=:blue, linewidth=1, alpha=0.5, label=(i==1 ? "trail" : nothing))
        end
        # Lead leg
        if lead_len[i] <= 1.0
            plot!(f, [P_lead_x, px_v[i]], [P_lead_y, py_v[i]], subplot=1,
                  color=:red, linewidth=1, alpha=0.5, label=(i==1 ? "lead" : nothing))
        end
    end

    plot!(f, t_, px_v, subplot=2, ylabel="pxy", xlabel="time", label="px", legend=leg_pos)
    plot!(f, t_, py_v, subplot=2, label="py", legend=leg_pos)

    plot!(f, t_, vx_v, subplot=3, label="vx", ylabel="vel", legend=leg_pos)
    plot!(f, t_, vy_v, subplot=3, label="vy", legend=leg_pos)

    # Forces - color coded: blue=trail, red=lead, solid=x, dashed=y
    plot!(f, t_, value(o[:Ftrail_x]), subplot=4, ylabel="force", color=:blue, linestyle=:solid, label="Ftrail_x", legend=leg_pos)
    plot!(f, t_, value(o[:Ftrail_y]), subplot=4, color=:blue, linestyle=:dash, label="Ftrail_y", legend=leg_pos)
    plot!(f, t_, value(o[:Flead_x]), subplot=4, color=:red, linestyle=:solid, label="Flead_x", legend=leg_pos)
    plot!(f, t_, value(o[:Flead_y]), subplot=4, color=:red, linestyle=:dash, label="Flead_y", legend=leg_pos)

    # Leg lengths with max length line - clip values > 1 to NaN
    trail_len_clipped = [l > 1.0 ? NaN : l for l in trail_len]
    lead_len_clipped = [l > 1.0 ? NaN : l for l in lead_len]
    plot!(f, t_, trail_len_clipped, subplot=5, linewidth=2, ylabel="leg length", color=:blue, label="trail", legend=leg_pos)
    plot!(f, t_, lead_len_clipped, subplot=5, color=:red, label="lead", legend=leg_pos)
    hline!(f, [1.0], subplot=5, linestyle=:dash, color=:black, label="max", legend=leg_pos)

    # Force rate
    plot!(f, t_, value(o[:Fddot_trail_p])-value(o[:Fddot_trail_m]), subplot=6, ylabel="frate", color=:blue, label="trail", legend=leg_pos)
    plot!(f, t_, value(o[:Fddot_lead_p])-value(o[:Fddot_lead_m]), subplot=6, color=:red, label="lead", legend=leg_pos)

    # Leg velocity
    plot!(f, t_, value(o[:trail_leg_velocity]), subplot=7, ylabel="dlegdt", color=:blue, label="trail", legend=leg_pos)
    plot!(f, t_, value(o[:lead_leg_velocity]), subplot=7, color=:red, label="lead", legend=leg_pos)

    # Dynamics violation
    vx_dot = value(∂(o[:vx],o[:τ]))
    vy_dot = value(∂(o[:vy],o[:τ]))
    force_viol_x = vx_dot - value(o[:Ftot_x])*value(o[:t_f])
    force_viol_y = vy_dot - (value(o[:Ftot_y]) .- g)*value(o[:t_f])
    plot!(f, t_, force_viol_x, subplot=8, xlabel="time", color=:blue, label="x", legend=leg_pos)
    plot!(f, t_, force_viol_y, subplot=8, color=:red, label="y", legend=leg_pos)

    return f
end

if abspath(PROGRAM_FILE) == @__FILE__
    model = point_mass_walker()
    f = plot_results(model)
    o = object_dictionary(model)
    println("Cost work: ", value(o[:cost_work]))
    println("Cost fr (linear): ", value(o[:cost_fr]))
    println("Cost fr2 (squared): ", value(o[:cost_fr2]))
    println("Final time: ", value(o[:t_f]))
end

model = point_mass_walker(y0_fixed = 0.98)
plot_results(model)
## Example: Changing objective and re-solving
# You can modify the objective using expressions defined in the model:
# @objective(model, Min, o[:cost_work] + o[:cost_fr2])
# optimize!(model)

"""
    run_parameter_sweep(; cost_type=:linear, step_lengths=0.4:0.1:0.8, step_speeds=0.3:0.1:0.7, kwargs...)

Run a parameter sweep over step lengths and speeds, solving the walking optimization
for each combination.

# Arguments
- `cost_type::Symbol`: Objective type, `:linear` (work + c_fr*|Fddot|) or `:squared` (work + c_fr2*Fddot²)
- `step_lengths`: Range of step lengths to sweep (default: 0.4:0.1:0.8)
- `step_speeds`: Range of step speeds to sweep (default: 0.3:0.1:0.7)
- `kwargs...`: Additional keyword arguments passed to `point_mass_walker()`, e.g.:
  - `c_fr`: Linear force rate penalty coefficient (default: 0.05)
  - `c_fr2`: Squared force rate penalty coefficient (default: 0.01)

# Returns
- `results`: Vector of NamedTuples with cost metrics for each successful solve
- `models`: 2D Matrix of InfiniteModel objects indexed by [speed_idx, length_idx]
- `step_speeds`: Vector of step speed values used
- `step_lengths`: Vector of step length values used
- `status_matrix`: 2D Matrix of convergence status symbols (:converged, :error, or solver status)

# Examples
```julia
# Default sweep with linear cost
results, models, speeds, lengths, status = run_parameter_sweep()

# Sweep with squared cost and custom coefficient
results, models, speeds, lengths, status = run_parameter_sweep(cost_type=:squared, c_fr2=0.02)

# Custom sweep ranges with modified linear penalty
results, models, speeds, lengths, status = run_parameter_sweep(
    step_lengths=0.5:0.05:0.7,
    step_speeds=0.4:0.05:0.6,
    c_fr=0.1
)

# Plot with convergence status
sweep_plot = plot_sweep_results(results; status_matrix=status,
                                 step_speeds=speeds, step_lengths=lengths)
```
"""
function run_parameter_sweep(; cost_type=:linear,
        step_lengths=0.4:0.1:0.8,
        step_speeds=0.3:0.1:0.7,
        kwargs...)

    # Storage for results and models
    results = []
    # 2D matrix of models indexed by [speed_idx, length_idx]
    models = Matrix{Union{Nothing, InfiniteModel}}(nothing, length(step_speeds), length(step_lengths))
    # Track convergence status for all grid points
    status_matrix = Matrix{Symbol}(undef, length(step_speeds), length(step_lengths))

    for (j, sl) in enumerate(step_lengths)
        for (i, ss) in enumerate(step_speeds)
            println("Running: step_length=$sl, step_speed=$ss, cost_type=$cost_type")
            try
                m = point_mass_walker(; step_length=sl, step_speed=ss, cost_type=cost_type, kwargs...)
                status = termination_status(m)
                if status == MOI.LOCALLY_SOLVED || status == MOI.OPTIMAL
                    o = object_dictionary(m)
                    push!(results, (
                        step_length = sl,
                        step_speed = ss,
                        cost_work = value(o[:cost_work]),
                        cost_fr = value(o[:cost_fr]),
                        cost_fr2 = value(o[:cost_fr2]),
                        total_cost = objective_value(m),
                        cost_type = cost_type
                    ))
                    models[i, j] = m  # Store the model
                    status_matrix[i, j] = :converged
                else
                    println("  Solver did not converge: ", status)
                    status_matrix[i, j] = Symbol(status)
                end
            catch e
                println("  Failed: ", e)
                status_matrix[i, j] = :error
            end
        end
    end
    return results, models, collect(step_speeds), collect(step_lengths), status_matrix
end

"""
    plot_sweep_results(results; interactive=true, status_matrix=nothing,
                       step_speeds=nothing, step_lengths=nothing)

Plot the results of a parameter sweep as 3D surfaces and contour plots.

# Arguments
- `results`: Vector of NamedTuples from `run_parameter_sweep()`
- `interactive`: If true, attempts to use PlotlyJS backend for interactive 3D rotation
- `status_matrix`: Optional convergence status matrix from `run_parameter_sweep()`
- `step_speeds`: Optional vector of step speeds (required if status_matrix provided)
- `step_lengths`: Optional vector of step lengths (required if status_matrix provided)

# Returns
A combined plot with 8 subplots:
- Row 1: Total cost (3D surface, contour)
- Row 2: Work cost (3D surface, contour)
- Row 3: Cost of Transport (3D surface, contour) - with adaptive scaling to 5-95th percentile
- Row 4: Convergence status heatmap, Summary statistics

# Example
```julia
results, models, speeds, lengths, status = run_parameter_sweep()
sweep_plot = plot_sweep_results(results; status_matrix=status,
                                 step_speeds=speeds, step_lengths=lengths)
savefig(sweep_plot, "sweep_results.png")
```
"""
function plot_sweep_results(results; interactive=true, status_matrix=nothing,
        step_speeds=nothing, step_lengths=nothing)
    if isempty(results)
        println("No valid results to plot")
        return nothing
    end

    # Use PlotlyJS backend for interactive 3D rotation (if available)
    if interactive
        try
            plotlyjs()
        catch
            println("PlotlyJS not available. Install with: using Pkg; Pkg.add(\"PlotlyJS\")")
            println("Using default backend (non-interactive 3D)")
        end
    end

    # Extract data
    sls = [r.step_length for r in results]
    sss = [r.step_speed for r in results]
    cost_type = results[1].cost_type

    # Get unique values for grid (speed on x-axis, length on y-axis)
    unique_ss = sort(unique(sss))  # x-axis: speed
    unique_sl = sort(unique(sls))  # y-axis: length

    # Create matrices for surface/contour plots
    # Matrix is [y_idx, x_idx] = [length_idx, speed_idx]
    cost_matrix = fill(NaN, length(unique_sl), length(unique_ss))
    work_matrix = fill(NaN, length(unique_sl), length(unique_ss))
    cot_matrix = fill(NaN, length(unique_sl), length(unique_ss))  # Cost of Transport

    for r in results
        i = findfirst(==(r.step_speed), unique_ss)   # x index (speed)
        j = findfirst(==(r.step_length), unique_sl)  # y index (length)
        cost_matrix[j, i] = r.total_cost
        work_matrix[j, i] = r.cost_work
        # Cost of Transport = cost per unit distance
        cot_matrix[j, i] = r.total_cost / r.step_length
    end

    cost_label = cost_type == :linear ? "Linear" : "Squared"

    # Compute adaptive clim for COT based on data distribution (clip outliers)
    valid_cot = filter(!isnan, cot_matrix[:])
    if !isempty(valid_cot)
        cot_lo = quantile(valid_cot, 0.05)
        cot_hi = quantile(valid_cot, 0.95)
        # Clamp the matrix for better visualization
        cot_matrix_clamped = clamp.(cot_matrix, cot_lo, cot_hi)
    else
        cot_matrix_clamped = cot_matrix
        cot_lo, cot_hi = 0.0, 1.0
    end

    # Print summary stats to console
    n_total = length(unique_ss) * length(unique_sl)
    n_converged = length(results)
    n_failed = n_total - n_converged
    println("Sweep summary:")
    println("  Converged: $n_converged / $n_total")
    println("  Failed: $n_failed")
    if !isempty(valid_cot)
        println("  COT range: $(round(minimum(valid_cot), digits=2)) - $(round(maximum(valid_cot), digits=2))")
        println("  COT 5-95%: $(round(cot_lo, digits=2)) - $(round(cot_hi, digits=2))")
    end

    # Create all plots first, then combine
    # Use @layout macro approach for PlotlyJS compatibility
    plots = []

    # 1. Total Cost 3D surface
    push!(plots, surface(unique_ss, unique_sl, cost_matrix,
        xlabel="Speed", ylabel="Length", zlabel="Cost",
        title="Total Cost ($cost_label)", legend=false))

    # 2. Total Cost contours
    push!(plots, contour(unique_ss, unique_sl, cost_matrix,
        xlabel="Speed", ylabel="Length",
        title="Cost Contours", fill=true, levels=20, legend=false))

    # 3. Work Cost 3D surface
    push!(plots, surface(unique_ss, unique_sl, work_matrix,
        xlabel="Speed", ylabel="Length", zlabel="Work",
        title="Work Cost", legend=false))

    # 4. Work Cost contours
    push!(plots, contour(unique_ss, unique_sl, work_matrix,
        xlabel="Speed", ylabel="Length",
        title="Work Contours", fill=true, levels=20, legend=false))

    # 5. Cost of Transport 3D surface (clamped)
    push!(plots, surface(unique_ss, unique_sl, cot_matrix_clamped,
        xlabel="Speed", ylabel="Length", zlabel="COT",
        title="COT (clamped)", clims=(cot_lo, cot_hi), legend=false))

    # 6. Cost of Transport contours (with adaptive levels)
    cot_levels = range(cot_lo, cot_hi, length=20)
    push!(plots, contour(unique_ss, unique_sl, cot_matrix_clamped,
        xlabel="Speed", ylabel="Length",
        title="COT Contours (5-95%)", fill=true, levels=cot_levels,
        clims=(cot_lo, cot_hi), legend=false))

    # 7. Convergence status heatmap (if status_matrix provided)
    if !isnothing(status_matrix) && !isnothing(step_speeds) && !isnothing(step_lengths)
        # Convert status to numeric for heatmap
        # 1 = converged, 0 = other status, -1 = error
        status_numeric = zeros(size(status_matrix))
        for i in eachindex(status_matrix)
            if status_matrix[i] == :converged
                status_numeric[i] = 1.0
            elseif status_matrix[i] == :error
                status_numeric[i] = -1.0
            else
                status_numeric[i] = 0.0
            end
        end
        # Transpose to match [length, speed] indexing for heatmap
        push!(plots, heatmap(step_speeds, step_lengths, status_numeric',
            xlabel="Speed", ylabel="Length",
            title="Status (1=OK,0=NC,-1=Err)",
            color=:RdYlGn, clims=(-1, 1), legend=false))
    else
        push!(plots, plot(title="Status (N/A)", axis=false, legend=false))
    end

    # 8. Summary text plot
    stats_text = "Conv: $n_converged/$n_total"
    push!(plots, plot(annotation=(0.5, 0.5, text(stats_text, 10, :center)),
        axis=false, grid=false, title="Summary", legend=false))

    # Combine all plots with explicit layout
    combined = plot(plots..., layout=(4, 2), size=(1000, 1400))
    return combined
end

"""
    inspect_model(models, step_speeds, step_lengths, speed_idx, length_idx)

Plot time-varying position, velocity, and force for a specific model from the sweep.
Returns the plot and prints key metrics.

Example:
    results, models, speeds, lengths = run_parameter_sweep()
    inspect_model(models, speeds, lengths, 3, 2)  # speed index 3, length index 2
"""
function inspect_model(models, step_speeds, step_lengths, speed_idx, length_idx)
    m = models[speed_idx, length_idx]
    if isnothing(m)
        println("No valid model at speed_idx=$speed_idx, length_idx=$length_idx")
        return nothing
    end

    ss = step_speeds[speed_idx]
    sl = step_lengths[length_idx]
    println("Inspecting model: step_speed=$ss, step_length=$sl")

    o = object_dictionary(m)
    println("  Cost work: ", round(value(o[:cost_work]), digits=4))
    println("  Cost fr: ", round(value(o[:cost_fr]), digits=4))
    println("  Total cost: ", round(objective_value(m), digits=4))
    println("  t_f: ", round(value(o[:t_f]), digits=4))

    # Complementarity diagnostics
    diagnose_complementarity(m)

    # Use the existing plot_results function
    return plot_results(m)
end

"""
    diagnose_complementarity(model)

Check whether the complementarity constraints are satisfied:
- F_trail should be 0 when trail_leg_length > 1
- F_lead should be 0 when lead_leg_length > 1

Prints violations where force > threshold when leg is overextended.
"""
function diagnose_complementarity(model; force_threshold=1e-3, length_threshold=1.001)
    o = object_dictionary(model)

    F_trail_v = value(o[:F_trail])
    F_lead_v = value(o[:F_lead])
    trail_len = value(o[:trail_leg_length])
    lead_len = value(o[:lead_leg_length])
    τ_v = value(o[:τ])

    println("\nComplementarity diagnostics:")
    println("  (Force should be ~0 when leg_length > 1)")

    # Check trail leg
    trail_violations = findall((trail_len .> length_threshold) .& (F_trail_v .> force_threshold))
    if isempty(trail_violations)
        println("  Trail leg: OK (no violations)")
    else
        println("  Trail leg: $(length(trail_violations)) violations")
        for idx in trail_violations[1:min(5, length(trail_violations))]
            println("    τ=$(round(τ_v[idx], digits=3)): len=$(round(trail_len[idx], digits=4)), F=$(round(F_trail_v[idx], digits=4))")
        end
        if length(trail_violations) > 5
            println("    ... and $(length(trail_violations)-5) more")
        end
    end

    # Check lead leg
    lead_violations = findall((lead_len .> length_threshold) .& (F_lead_v .> force_threshold))
    if isempty(lead_violations)
        println("  Lead leg: OK (no violations)")
    else
        println("  Lead leg: $(length(lead_violations)) violations")
        for idx in lead_violations[1:min(5, length(lead_violations))]
            println("    τ=$(round(τ_v[idx], digits=3)): len=$(round(lead_len[idx], digits=4)), F=$(round(F_lead_v[idx], digits=4))")
        end
        if length(lead_violations) > 5
            println("    ... and $(length(lead_violations)-5) more")
        end
    end

    # Symmetry check
    println("\nSymmetry diagnostics:")
    px_v = value(o[:px])
    py_v = value(o[:py])
    vx_v = value(o[:vx])
    vy_v = value(o[:vy])
    sl = px_v[end]  # step length from solution

    # For symmetric gait: px(τ) + px(1-τ) should equal sl
    # And py(τ) should equal py(1-τ), etc.
    n = length(τ_v)
    mid = (n + 1) ÷ 2

    # Check position symmetry around midpoint
    px_sym_err = maximum(abs.(px_v[1:mid] .+ reverse(px_v[mid:end])[1:mid] .- sl))
    py_sym_err = maximum(abs.(py_v[1:mid] .- reverse(py_v[mid:end])[1:mid]))
    vx_sym_err = maximum(abs.(vx_v[1:mid] .- reverse(vx_v[mid:end])[1:mid]))
    vy_sym_err = maximum(abs.(vy_v[1:mid] .+ reverse(vy_v[mid:end])[1:mid]))

    println("  px symmetry error: $(round(px_sym_err, digits=4)) (should be ~0)")
    println("  py symmetry error: $(round(py_sym_err, digits=4)) (should be ~0)")
    println("  vx symmetry error: $(round(vx_sym_err, digits=4)) (should be ~0)")
    println("  vy symmetry error: $(round(vy_sym_err, digits=4)) (should be ~0)")

    # Check force symmetry
    F_trail_sym_err = maximum(abs.(F_trail_v[1:mid] .- reverse(F_lead_v[mid:end])[1:mid]))
    println("  F_trail/F_lead symmetry error: $(round(F_trail_sym_err, digits=4)) (should be ~0)")
end

# Uncomment to run the parameter sweep:
# results, models, speeds, lengths = run_parameter_sweep(cost_type=:linear)
# sweep_plot = plot_sweep_results(results)
#
# # Inspect a specific model (e.g., speed index 3, length index 2)
# inspect_model(models, speeds, lengths, 3, 2)
