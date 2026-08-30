### A Pluto.jl notebook ###
# v0.20
# Kickflip playground — run with:
#   julia --project=. -e 'using Pluto; Pluto.run(notebook="kickflip_playground.jl")'
# Sliders re-solve the flight with the pop frozen to the cached 3-D ollie (~30–60 s per solve),
# so nudge one slider at a time. Coordinates: X lateral, Y forward, Z up (see ollie_rider_3D.jl).

using Markdown
using InteractiveUtils

# ╔═╡ a1000001-0000-0000-0000-000000000001
begin
    import Pkg
    Pkg.activate(Base.current_project(@__DIR__))
    using PlutoUI, Plots, Serialization
    include(joinpath(@__DIR__, "ollie_rider_3D.jl"))
    using .OllieRider3D
    md"**Kickflip playground** — model `ollie_rider_3D.jl` (X lateral, Y forward, Z up)"
end

# ╔═╡ a1000002-0000-0000-0000-000000000002
begin
    const STAGE1 = joinpath(@__DIR__, "simulation_reports", "ollie3d_stage1.jls")
    const STAGE2 = joinpath(@__DIR__, "simulation_reports", "kickflip3d_stage2.jls")
    r1 = deserialize(STAGE1)   # (; g, d, a) — 3-D ollie, ground phases reused below
    r2 = deserialize(STAGE2)   # (; d, a, opt) — kickflip warm start
    md"Loaded cached stages: ollie apex $(round(r1.a.lowest_point_apex, digits=3)), kickflip apex $(round(r2.a.lowest_point_apex, digits=3))"
end

# ╔═╡ a1000003-0000-0000-0000-000000000003
md"""
| control | slider |
|---|---|
| flip turns | $(@bind flip_turns Slider(0.0:0.25:2.0, default=1.0, show_value=true)) |
| force-rate weight (log₁₀) | $(@bind log_frw Slider(-4.0:0.5:1.0, default=-4.0, show_value=true)) |
| foot slide max | $(@bind slide Slider(0.5:0.25:3.0, default=1.5, show_value=true)) |
| foot–deck μ (flight) | $(@bind mu Slider(0.0:0.1:1.0, default=0.0, show_value=true)) |
| flip sign | $(@bind fsign Select([1.0 => "kickflip (+)", -1.0 => "heelflip (−)"])) |
| re-solve | $(@bind go CounterButton("solve")) |
"""

# ╔═╡ a1000004-0000-0000-0000-000000000004
begin
    go  # depend on the button so the cell re-runs on click
    opt = Rider3DOptions(n_flight=61, flip_turns=flip_turns, flip_sign=fsign, objective=:lowest_point,
                         force_rate_weight=10.0^log_frw, foot_slide_max=slide, mu=mu)
    sol = solve_rider_3d(opt=opt, ground=r1.g, warm=r2.d)
    a = audit_rider_3d(sol); d = rider3d_data(sol)
    md"**$(a.status)** — lowest-point apex $(round(a.lowest_point_apex, digits=3)), roll $(round(a.roll_turns, digits=3)) turns, peak N (back, front) $(round.(a.peak_normal_flight, digits=1)) BW, max roll rate $(round(a.max_roll_rate, digits=2))"
end

# ╔═╡ a1000005-0000-0000-0000-000000000005
begin
    t = d.t2
    p1 = plot(t, d.th2, label="θ pitch", ylabel="rad")
    plot!(p1, t, d.ph2, label="φ roll"); plot!(p1, t, d.ps2, label="ψ yaw")
    p2 = plot(t[1:end-1], d.Nb2, label="N back", ylabel="board weights", color=:orange)
    plot!(p2, t[1:end-1], d.Nf2, label="N front", color=:green)
    p3 = plot(t[1:end-1], d.s2, label="s (long)", ylabel="deck coords")
    plot!(p3, t[1:end-1], d.sg2, label="σ (lateral)")
    p4 = plot(t, d.zb2, label="board z", ylabel="height", xlabel="t")
    plot!(p4, t, d.zr2, label="rider z")
    plot(p1, p2, p3, p4, layout=(4,1), size=(760, 900), legend=:topright)
end

# ╔═╡ Cell order:
# ╠═a1000001-0000-0000-0000-000000000001
# ╠═a1000002-0000-0000-0000-000000000002
# ╟─a1000003-0000-0000-0000-000000000003
# ╠═a1000004-0000-0000-0000-000000000004
# ╠═a1000005-0000-0000-0000-000000000005
