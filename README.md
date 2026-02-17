# README #

### Project: Simulations that demonstrate how to perform optimization for human movement experiments. I emphasize solving for movement time as a decision variable, and typically am interested in energy in the objective.

Using InfiniteOpt, demonstrate Energy+Time power laws in eye movements, reaching, walking.

### Log
2025-10-23: Rebula_demo.ipynb (mybinder)   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jeremydwong/infinite_learning/main?filepath=Rebula_demo.ipynb) <br>ln
2025-01-15: eye movements 2nd order.

### What is this repository for? ###

* Quick start (instantiation, runnable scripts, dependencies) <br>
* Who do I talk to? <br>

### Quick start ###

* Julia 1.11
```
    Using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    include("infinite_2nd.jl") #or any of the other infinite_[*] files.
```
* Dependencies
    noted within Project.toml.

---

## Tutorial: InfiniteOpt Common Pitfalls and Tips

### Fix: `value()` vs `parameter_value()` for Finite Parameters

When building an InfiniteOpt model, you might want to use a `@finite_parameter` value to compute other quantities *before* the model is solved. A common mistake is using `value()`, which queries **solution results** (post-optimization).

**Wrong** - causes `LoadError: Unable to query value since up-to-date solution results are not available`:
```julia
@finite_parameter(model, y_0 == 0.95)
@finite_parameter(model, α == 0.35)
sl = 2 * value(y_0) * sin(value(α))  # ERROR: value() is for post-solve!
```

**Correct** - use `parameter_value()` to extract the numeric value before solving:
```julia
@finite_parameter(model, y_0 == 0.95)
@finite_parameter(model, α == 0.35)
sl = 2 * parameter_value(y_0) * sin(parameter_value(α))  # Works!
```

### Fix: `set_optimizer_attributes` with InfiniteModel

The plural `set_optimizer_attributes` with `Pair` syntax doesn't work with `InfiniteModel`. Use individual calls instead:

**Wrong**:
```julia
set_optimizer_attributes(model, "tol" => 1e-3, "max_iter" => 500)
```

**Correct**:
```julia
set_optimizer_attribute(model, "tol", 1e-3)
set_optimizer_attribute(model, "max_iter", 500)
```

### Accessing Model Results After Solving

After calling `optimize!(model)`, you can access all named variables, parameters, and expressions using `object_dictionary()`:

```julia
model = point_mass_walker()  # Returns solved model

# Get the dictionary of all named objects
o = object_dictionary(model)

# Access solution values
println("Cost work: ", value(o[:cost_work]))
println("Cost fr: ", value(o[:cost_fr]))
println("Final time: ", value(o[:t_f]))

# Access time-varying variables (returns arrays at support points)
px_vals = value(o[:px])  # Position x over time
τ_vals = value(o[:τ])    # Normalized time supports

# Check optimization status
println("Status: ", termination_status(model))
println("Objective: ", objective_value(model))
```

### Changing the Objective and Re-solving

Since expressions like `cost_work`, `cost_fr`, `cost_fr2` are stored in the model, you can change the objective externally:

```julia
model = point_mass_walker()
o = object_dictionary(model)

# Change to squared force rate objective
@objective(model, Min, o[:cost_work] + o[:cost_fr2])
optimize!(model)

# Check new results
println("New objective: ", objective_value(model))
```

---

## `point_mass_walker()` Function Reference

The main optimization function accepts these optional keyword arguments:

```julia
model = point_mass_walker(;
    c_fr = 0.05,           # Linear force rate penalty coefficient
    c_fr2 = 0.01,          # Squared force rate penalty coefficient
    step_length = nothing, # Step length (default: 2*y_0*sin(α) ≈ 0.65)
    step_speed = nothing,  # Step speed (default: step_length/1.2)
    cost_type = :linear    # :linear (work+fr) or :squared (work+fr2)
)
```

**Examples**:
```julia
# Default parameters
model = point_mass_walker()

# Custom step length and speed
model = point_mass_walker(step_length=0.5, step_speed=0.5)

# Use squared force rate penalty
model = point_mass_walker(cost_type=:squared, c_fr2=0.02)
```

### Parameter Sweep

The file includes functions for sweeping across step lengths and speeds:

```julia
results = run_parameter_sweep()
sweep_plot = plot_sweep_results(results)
```

---

### Who do I talk to? ###
* [https://scholar.google.com/citations?user=h89-UMkAAAAJ&hl=en&oi=sra](Jeremy Wong)