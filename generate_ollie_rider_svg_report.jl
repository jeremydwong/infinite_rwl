#!/usr/bin/env julia
# Vector report for the rider + board ollie (ollie_rider.jl).
# Two solves: (A) objective=:board_apex, (B) objective=:lowest_point.
# Only B (lowest board point) is ANIMATED — inline SVG driven by requestAnimationFrame with a
# scrub slider (SMIL was chunky: ~170 discrete-visibility text nodes re-evaluated per frame).
# A is reported as static snapshots + time series. All numbers are nondimensional (board
# mass 1, g 1, ℓ_max 1); SI equivalents via RiderScales are shown alongside.
#
#   julia --project=. generate_ollie_rider_svg_report.jl            # solve (minutes each), cache
#   julia --project=. generate_ollie_rider_svg_report.jl --resolve  # ignore the cache
#
# Cache: simulation_reports/ollie_rider_svg_cache.jls (rider_data + audit + options per case).
include("ollie_rider.jl"); using .OllieRider
const FORCE_LIMITS_SECTION = let f=joinpath(@__DIR__,"simulation_reports","force_limits_section.html"); isfile(f) ? read(f,String) : "" end
using Serialization, Printf, JuMP, LinearAlgebra

const REPORT_DIR = joinpath(@__DIR__, "simulation_reports"); mkpath(REPORT_DIR)
const CACHE = joinpath(REPORT_DIR, "ollie_rider_svg_cache.jls")
const p = OllieRider.ReducedBoardParams(inertia=0.07)   # pitch inertia from Appendix 1 (real deck + Indy 149 trucks, model geometry)
const SC = RiderScales(); const OPT0 = RiderOptions(); const DIM = dimensional(SC, OPT0)
const FMAX = force_limit_value(OPT0)              # board-weight units (= force_limit_bw rider weights)
si_len(x) = "$(round(x*DIM.length, digits=3)) m"; si_time(x) = "$(round(x*DIM.time, digits=3)) s"
si_force(x) = "$(round(x*DIM.force, digits=0)) N = $(round(x*DIM.force_bw, digits=2)) BW"
Xr(th,r) = cos(th)*r[1]-sin(th)*r[2]; Yr(th,r) = sin(th)*r[1]+cos(th)*r[2]
const rt = OllieRider.body_point(p,:tail); const rc = OllieRider.body_point(p,:rear_slide)
const rf = OllieRider.body_point(p,:front_slide); const rnose = [p.deck_length/2, 0.0]
const rkink = [-0.32, 0.0]                      # tail kink on the deck line
const rwheel = ([-0.29,-0.08], [0.29,-0.08])    # wheel centres (body frame)
const WHEEL_R = p.deck_height*0.6               # 0.048 world units
const V_FORWARD = 0.0                           # no cosmetic drift: vx0 supplies the real forward motion
const POST_LANDING = 1.2                        # simulated post-landing coast appended to the animations only (s)
const VX0 = 0.63                                # initial rolling speed: 2 m/s / sqrt(g L), L=1 m (Jeremy)
const LAND_MATCHED_V = true                     # landing: rider vx == board vx
const LAND_COM_TRUCKS = true                    # landing: rider COM x between the wheel contacts
const ABSORB_T = 0.3                            # leg absorption time after touchdown (s)

h(s) = replace(string(s), "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;")
r3(x) = round(x, digits=3); r4(x) = round(x, digits=4); f2(x) = string(round(x, digits=2))

# ------------------------------------------------------------------ solve / cache
function run_case(objective; warm=true)
    opt = RiderOptions(objective=objective, tol=1e-4, max_iter=10000, warm_from_board_apex=warm, land_matched_velocity=LAND_MATCHED_V, land_com_over_trucks=LAND_COM_TRUCKS, vx0=VX0)
    t = @elapsed sol = solve_rider_ollie(p=p, opt=opt)
    d = rider_data(sol); a = audit_rider_ollie(sol)
    println("$(objective) warm=$(warm): $(a.status) in $(round(t,digits=1)) s  lowest-point apex=$(a.lowest_point_apex) theta_mid=$(a.theta_at_mid)")
    (; d, a, wall=t, objective, warm, status=string(a.status),
       obj_value=try objective_value(sol.model) catch; NaN end,
       converged = a.status in (MOI.LOCALLY_SOLVED, MOI.OPTIMAL, MOI.ALMOST_LOCALLY_SOLVED))
end
cases = if isfile(CACHE) && !("--resolve" in ARGS)
    println("using cache $(CACHE)"); deserialize(CACHE)
else
    c = [run_case(:board_apex), run_case(:lowest_point; warm=true), run_case(:lowest_point; warm=false)]
    serialize(CACHE, c); c
end
# B: pick the lowest_point variant (warm / cold start) with the higher lowest-point apex, preferring
# converged ones and a level board (|θ(mid)| small) as tie-break.
Bcands = cases[2:3]
score(c) = (c.converged ? 1 : 0, c.a.lowest_point_apex - 0.5*abs(c.a.theta_at_mid))
Bsel = argmax(i -> score(Bcands[i]), 1:2)
println("B variants: warm lowest-point apex=$(Bcands[1].a.lowest_point_apex) θmid=$(Bcands[1].a.theta_at_mid) ($(Bcands[1].status)); cold: $(Bcands[2].a.lowest_point_apex) θmid=$(Bcands[2].a.theta_at_mid) ($(Bcands[2].status)); selected $(Bsel==1 ? "warm" : "cold")")
allB = Bcands; cases = [cases[1], Bcands[Bsel]]
const TAGS = ("A","B"); const NAMES = ("A: maximise board-COM apex", "B: maximise lowest-board-point height at apex ($(Bsel==1 ? "warm-started from A" : "cold start"))")

# ------------------------------------------------------------------ frames (world geometry per node)
"Build one frame per state node; controls pair with state k (last state reuses control n-1)."
function frames(d; v_forward=V_FORWARD, M=OPT0.rider_mass)
    F = []
    ctl(v,k) = v[min(k,length(v))]
    m, I, g = p.mass, p.inertia, p.gravity
    # per-node mechanical energies (board weight · length units); board vertical velocity in the pivot phase is −X(θ,rc)·ω
    KE0 = 0.5m*d.vb0.^2 .+ 0.5M*(d.vxr0.^2 .+ d.vyr0.^2);                 PE0 = m*g*p.deck_height .+ M*g*d.yr0
    vyb1 = [-Xr(d.th1[k], rc)*d.om1[k] for k in eachindex(d.th1)]
    KE1 = 0.5m*(d.vb1.^2 .+ vyb1.^2) .+ 0.5I*d.om1.^2 .+ 0.5M*(d.vxr1.^2 .+ d.vyr1.^2); PE1 = m*g*d.yb1 .+ M*g*d.yr1
    KE2 = 0.5m*(d.vb2.^2 .+ d.vyb2.^2) .+ 0.5I*d.om2.^2 .+ 0.5M*(d.vxr2.^2 .+ d.vyr2.^2); PE2 = m*g*d.yb2 .+ M*g*d.yr2
    # cumulative leg work = within-phase energy change (impact steps excluded and booked in Eloss), split by sign
    Wnet = 0.0; Wpos = 0.0; Wneg = 0.0; Eloss = 0.0
    push_phase!(ph, t, xb, yb, th, xr, yr, Fb, Ff, s, Gbx, Gfx, skipfirst, KE, PE) = begin
        for k in (skipfirst ? 2 : 1):length(t)
            if k >= 2
                dE = (KE[k]+PE[k]) - (KE[k-1]+PE[k-1]); Wnet += dE; Wpos += max(dE, 0.0); Wneg += min(dE, 0.0)
            end
            θ = th[k]; xo = xb[k] + v_forward*t[k]; yo = yb[k]
            world(r) = (xo + Xr(θ,r), yo + Yr(θ,r))
            sk = ctl(s,k); tail = world(rt); front = world([sk,0.0]); rider = (xr[k] + v_forward*t[k], yr[k])
            u(foot) = begin dx = rider[1]-foot[1]; dy = rider[2]-foot[2]; ℓ = hypot(dx,dy); (dx/ℓ, dy/ℓ, ℓ) end
            ubx,uby,ℓb = u(tail); ufx,ufy,ℓf = u(front)
            if ph == 2
                Fbb = (-ctl(Gbx,k), -ctl(Fb,k)); Ffb = (-ctl(Gfx,k), -ctl(Ff,k))
            else
                Fbb = (-ctl(Fb,k)*ubx, -ctl(Fb,k)*uby); Ffb = (-ctl(Ff,k)*ufx, -ctl(Ff,k)*ufy)
            end
            nθ = (-sin(θ), cos(θ)); fbn = abs(Fbb[1]*nθ[1] + Fbb[2]*nθ[2]); ffn = abs(Ffb[1]*nθ[1] + Ffb[2]*nθ[2])
            scop = fbn + ffn > 1e-9 ? (fbn*rt[1] + ffn*sk)/(fbn+ffn) : NaN
            cop = isnan(scop) ? (NaN, NaN) : world([scop, 0.0])
            push!(F, (; ph, t=t[k], θ, xo, yo, tail, kink=world(rkink), nose=world(rnose), cop, scop,
                       wheels=(world(rwheel[1]), world(rwheel[2])), front, rider, Fbb, Ffb,
                       ℓb, ℓf, s=sk, Fb=ctl(Fb,k), Ff=ctl(Ff,k),
                       lowest=minimum(last.(world.((rt, rnose, rf, rc)))), tail_y=tail[2],
                       KE=KE[k], PE=PE[k], Wnet, Wpos, Wneg, Eloss))
        end
    end
    z = zeros(length(d.t2))
    push_phase!(0, d.t0, d.xb0, d.yb0, d.th0, d.xr0, d.yr0, d.Fb0, d.Ff0, d.s0, z, z, false, KE0, PE0)
    push_phase!(1, d.t1, d.xb1, d.yb1, d.th1, d.xr1, d.yr1, d.Fb1, d.Ff1, d.s1, z, z, true, KE1, PE1)
    Eloss += (KE1[end]+PE1[end]) - (KE2[1]+PE2[1])          # tail-strike impact loss (phase 1 → 2 reset)
    push_phase!(2, d.t2, d.xb2, d.yb2, d.th2, d.xr2, d.yr2, d.Fb2, d.Ff2, d.s2, d.Gbx, d.Gfx, true, KE2, PE2)
    F
end

"""Append a SIMULATED post-landing coast (animations only): board flat on the ice at its landing
horizontal velocity; rider decelerated vertically to rest over `absorb` s (clamped at foot_margin above
the feet), then coasting; arrows = implied absorbing leg forces M(a+g)/2 per foot, then M g/2."""
# Two-wheel touchdown wrench (animation only). Both wheels (body-frame x = ±0.29) touch the ice at the same
# instant (landing is θ=0, ω=0). Ice is frictionless, so the horizontal impulse is zero (e=1 in x) and the vertical
# impact is perfectly inelastic (e=0 in y): vertical impulses J1 (rear), J2 (front) satisfy J1+J2 = −m·vy and
# x1·J1 + x2·J2 = −I·ω, leaving the board flat, at rest vertically, riding away at its landing vx.
function landing_wrench(d; m=p.mass, I=p.inertia)
    vx, vy, ω = d.vb2[end], d.vyb2[end], d.om2[end]
    x1, x2 = rwheel[1][1], rwheel[2][1]
    A = [1.0 1.0; x1 x2]; J = A \ [-m*vy, -I*ω]
    (; J1=J[1], J2=J[2], Jy=J[1]+J[2], Jx=0.0, moment=x1*J[1]+x2*J[2], vx_before=vx, vy_before=vy, om_before=ω,
       vplus=(vx, 0.0, 0.0), KE_lost=0.5*m*vy^2 + 0.5*I*ω^2)
end

function coast_frames(F, d; post_landing=POST_LANDING, absorb=ABSORB_T, v_forward=V_FORWARD, M=RiderOptions().rider_mass, g=p.gravity, foot_margin=RiderOptions().foot_margin, dt=0.02)
    post_landing <= 0 && return F
    L = F[end]; tL = L.t; xbL = d.xb2[end]; wr = landing_wrench(d); vb = wr.vplus[1]; sL = d.s2[end]
    xr0, yr0, vxr, vyr = d.xr2[end], d.yr2[end], d.vxr2[end], d.vyr2[end]
    a = -vyr/absorb                                       # constant upward deceleration (vyr < 0 at touchdown)
    tail = (rt[1], p.deck_height + rt[2]); front = (sL, p.deck_height)   # body-frame foot points (θ=0)
    yfloor = max(tail[2], front[2]) + foot_margin
    out = copy(F)
    m = p.mass; Wnet, Wpos, Wneg = L.Wnet, L.Wpos, L.Wneg
    Eloss = L.Eloss + wr.KE_lost                          # landing impact (e=0 vertically) booked as a loss, not leg work
    Eprev = 0.5m*vb^2 + 0.5M*(vxr^2 + vyr^2) + m*g*p.deck_height + M*g*yr0    # post-impact energy at touchdown
    for τ in dt:dt:post_landing
        t = tL + τ; xo = xbL + vb*τ + v_forward*t; yo = p.deck_height
        yr = τ <= absorb ? yr0 + vyr*τ + 0.5a*τ^2 : yr0 + vyr*absorb + 0.5a*absorb^2
        yr = max(yr, yfloor)
        xr = xr0 + vxr*τ + v_forward*t
        Fper = τ <= absorb ? M*(a+g)/2 : M*g/2
        vyr_t = τ <= absorb ? vyr + a*τ : 0.0
        KE = 0.5m*vb^2 + 0.5M*(vxr^2 + vyr_t^2); PE = m*g*p.deck_height + M*g*yr
        dE = KE + PE - Eprev; Eprev = KE + PE; Wnet += dE; Wpos += max(dE, 0.0); Wneg += min(dE, 0.0)
        tw = (xo+tail[1], tail[2]); fw = (xo+front[1], front[2]); rider = (xr, yr)
        scop = (tail[1] + sL)/2
        push!(out, (; ph=3, t, θ=0.0, xo, yo, tail=tw, kink=(xo+rkink[1], yo), nose=(xo+rnose[1], yo), cop=(xo+scop, yo), scop,
                     wheels=((xo+rwheel[1][1], yo+rwheel[1][2]), (xo+rwheel[2][1], yo+rwheel[2][2])), front=fw, rider,
                     Fbb=(0.0, -Fper), Ffb=(0.0, -Fper), ℓb=hypot(xr-tw[1], yr-tw[2]), ℓf=hypot(xr-fw[1], yr-fw[2]),
                     s=sL, Fb=Fper, Ff=Fper, lowest=0.0, tail_y=tail[2], rideaway=vb, wrench=wr,
                     KE, PE, Wnet, Wpos, Wneg, Eloss))
    end
    out
end

"Fixed world window containing the whole motion (with force arrows), equal px/unit on x and y."
function window(Fs...; width, height, L=20, R=20, T=40, B=30, arrow_len=0.5)
    xs = Float64[]; ys = Float64[]
    for F in Fs, f in F
        for pt in (f.tail, f.nose, f.rider, f.wheels..., f.front) push!(xs, pt[1]); push!(ys, pt[2]) end
        for (foot, Fv) in ((f.tail, f.Fbb), (f.front, f.Ffb))
            push!(xs, foot[1] + arrow_len*Fv[1]/FMAX); push!(ys, foot[2] + arrow_len*Fv[2]/FMAX)
        end
    end
    xmin, xmax = extrema(xs) .+ (-0.12, 0.12); ymin = min(minimum(ys), 0.0) - 0.15; ymax = maximum(ys) + 0.12
    pw = width-L-R; ph = height-T-B; scale = min(pw/(xmax-xmin), ph/(ymax-ymin))
    xoff = L + (pw - scale*(xmax-xmin))/2; yoff = T + (ph - scale*(ymax-ymin))/2
    (; X = x -> xoff + scale*(x-xmin), Y = y -> yoff + scale*(ymax-y), scale, xoff, yoff, xmin, xmax, ymin, ymax, width, height, arrow_len)
end

const PHASE_NAMES = ("load: flat board sliding on ice", "pop: pivot on rear wheel", "flight", "post-landing ride-away (simulated, not optimized)")
phase_name(F, ph) = ph == 3 ? (f = F[findfirst(g->g.ph==3, F)]; "$(PHASE_NAMES[4]): both wheels land together, vertical impulse $(f2(f.wrench.Jy)) kills vy (e=0), x untouched (e=1) → rides away at vx = $(f2(f.rideaway))") : PHASE_NAMES[ph+1]
const COL_BACK = "#e07b00"; const COL_FRONT = "#2a9d3f"

"Arrow (shaft polyline + head polygon) for a force on the board applied at `foot`. Returns (shaft pts, head pts)."
function arrow_geom(W, foot, Fv; Fmax=FMAX)
    mag = hypot(Fv...); len = W.arrow_len*mag/Fmax
    if mag < 1e-9
        x = W.X(foot[1]); y = W.Y(foot[2]); return ("$x,$y $x,$y", "$x,$y $x,$y $x,$y")
    end
    ux, uy = Fv[1]/mag, Fv[2]/mag; tip = (foot[1]+len*ux, foot[2]+len*uy)
    hs = min(0.06, 0.6*len); base = (tip[1]-hs*ux, tip[2]-hs*uy); nx, ny = -uy, ux; hw = 0.5*hs
    P(pt) = "$(f2(W.X(pt[1]))),$(f2(W.Y(pt[2])))"
    (P(foot)*" "*P(base), P(tip)*" "*P((base[1]+hw*nx, base[2]+hw*ny))*" "*P((base[1]-hw*nx, base[2]-hw*ny)))
end

"Static drawing of one frame (used by snapshots, and as the initial state of the animation)."
function draw_frame(io, W, f; forces=true)
    P(pt) = "$(f2(W.X(pt[1]))) $(f2(W.Y(pt[2])))"
    println(io, "<line x1='$(f2(W.X(f.tail[1])))' y1='$(f2(W.Y(f.tail[2])))' x2='$(f2(W.X(f.rider[1])))' y2='$(f2(W.Y(f.rider[2])))' stroke='$COL_BACK' stroke-width='4' stroke-linecap='round' opacity='0.85'/>")
    println(io, "<line x1='$(f2(W.X(f.front[1])))' y1='$(f2(W.Y(f.front[2])))' x2='$(f2(W.X(f.rider[1])))' y2='$(f2(W.Y(f.rider[2])))' stroke='$COL_FRONT' stroke-width='4' stroke-linecap='round' opacity='0.85'/>")
    println(io, "<path d='M $(P(f.tail)) L $(P(f.kink)) L $(P(f.nose))' fill='none' stroke='#202020' stroke-width='10' stroke-linecap='round' stroke-linejoin='round'/>")
    for w in f.wheels println(io, "<circle cx='$(f2(W.X(w[1])))' cy='$(f2(W.Y(w[2])))' r='$(f2(W.scale*WHEEL_R))' fill='#1261a0' stroke='#0b3d66' stroke-width='1.5'/>") end
    println(io, "<circle cx='$(f2(W.X(f.rider[1])))' cy='$(f2(W.Y(f.rider[2])))' r='9' fill='#d1495b' stroke='#7a1f2b' stroke-width='1.5'/>")
    isnan(f.cop[1]) || println(io, "<circle cx='$(f2(W.X(f.cop[1])))' cy='$(f2(W.Y(f.cop[2])))' r='7' fill='#8e44ad' opacity='0.35'/>")
    if forces
        for (foot, Fv, c) in ((f.tail, f.Fbb, COL_BACK), (f.front, f.Ffb, COL_FRONT))
            sh, hd = arrow_geom(W, foot, Fv)
            println(io, "<polyline points='$sh' stroke='$c' stroke-width='3.5' fill='none'/><polygon points='$hd' fill='$c'/>")
        end
    end
end

"""One small time-series panel (static polylines) at (ox,oy) of plot size pw×ph, t ∈ [0,T].
`series` = [(ys, color, name, dash)], `yticks` = [(value, label)], `vlines` = phase-start times."""
function mini_panel(io, ox, oy, pw, ph, ts, series, T; title="", yticks=[], vlines=Float64[], ylo=nothing, yhi=nothing, cursor_id="")
    ally = vcat([s[1] for s in series]...); ally = filter(isfinite, ally)
    lo = ylo === nothing ? minimum(ally) : ylo; hi = yhi === nothing ? maximum(ally) : yhi
    hi <= lo && (hi = lo + 1.0); pad = 0.04*(hi-lo); lo -= pad; hi += pad
    X(t) = ox + pw*t/T; Y(y) = oy + ph*(hi-y)/(hi-lo)
    println(io, "<text x='$ox' y='$(oy-5)' font-size='11' font-weight='bold'>$(h(title))</text>")
    println(io, "<rect x='$ox' y='$oy' width='$pw' height='$ph' fill='none' stroke='#999' stroke-width='1'/>")
    for (v, lab) in yticks
        lo <= v <= hi || continue
        println(io, "<line x1='$ox' y1='$(f2(Y(v)))' x2='$(ox+pw)' y2='$(f2(Y(v)))' stroke='#e4e4e4' stroke-width='1'/>",
            "<text x='$(ox-4)' y='$(f2(Y(v)+3.5))' text-anchor='end' font-size='10' fill='#555'>$lab</text>")
    end
    for tb in vlines
        println(io, "<line x1='$(f2(X(tb)))' y1='$oy' x2='$(f2(X(tb)))' y2='$(oy+ph)' stroke='#bbb' stroke-width='1' stroke-dasharray='3 3'/>")
    end
    for (ys, c, nm, dash) in series
        pts = join(["$(f2(X(t))),$(f2(Y(y)))" for (t,y) in zip(ts, ys) if isfinite(y)], " ")
        println(io, "<polyline points='$pts' fill='none' stroke='$c' stroke-width='1.5'$(dash == "" ? "" : " stroke-dasharray='$dash'")/>")
    end
    lx = ox + 4
    for (ys, c, nm, dash) in series
        println(io, "<line x1='$lx' y1='$(oy+ph-6)' x2='$(lx+12)' y2='$(oy+ph-6)' stroke='$c' stroke-width='2'$(dash == "" ? "" : " stroke-dasharray='$dash'")/><text x='$(lx+15)' y='$(oy+ph-3)' font-size='10' fill='#333'>$(h(nm))</text>")
        lx += 15 + 6*length(nm) + 8
    end
    cursor_id == "" || println(io, "<line id='$cursor_id' x1='$ox' y1='$oy' x2='$ox' y2='$(oy+ph)' stroke='#d1495b' stroke-width='1.5'/>")
    nothing
end

"""Inline SVG + JS animation of one solve: frames are embedded as JSON, a requestAnimationFrame loop
interpolates linearly between them in physical time (× `speed`), and a range slider scrubs. Returns an
HTML fragment (svg + controls + script); ids are prefixed by `tag` so several can coexist."""
function animation_html(F, W; label="", tag="B", speed=0.25)
    T = F[end].t
    P(pt) = "$(f2(W.X(pt[1]))) $(f2(W.Y(pt[2])))"
    fr(f) = "[" * join(f2.([f.t, f.ph, f.tail..., f.kink..., f.nose..., f.wheels[1]..., f.wheels[2]..., f.front..., f.rider..., f.Fbb..., f.Ffb..., f.xo, f.yo, (isnan(f.cop[1]) ? (f.xo, f.yo) : f.cop)...]), ",") * "]"
    phases = Dict(ph => h(phase_name(F, ph)) for ph in 0:3 if any(f->f.ph==ph, F))
    # side panels (force / energy / leg work vs time), stacked to the right of the animation
    PW = round(Int, W.width/4); PL = 52; PR = 16; PGAP = 22; PT = 40; PBOT = 30
    PH = round(Int, (W.height - PT - PBOT - 2PGAP)/3); PX = W.width + PL; TOTW = W.width + PL + PW + PR
    io = IOBuffer(); f0 = F[1]
    println(io, "<div class='anim' id='anim-$tag'>")
    println(io, "<svg xmlns='http://www.w3.org/2000/svg' width='$TOTW' height='$(W.height)' viewBox='0 0 $TOTW $(W.height)' font-family='Helvetica,Arial,sans-serif'>")
    println(io, "<rect width='100%' height='100%' fill='white'/>")
    println(io, "<line x1='0' y1='$(f2(W.Y(0)))' x2='$(W.width)' y2='$(f2(W.Y(0)))' stroke='#444' stroke-width='3'/>")
    # toggleable overlays (drawn under the body): standing line, reach circles, COM trails
    ystand = p.deck_height + OPT0.leg_max
    println(io, "<g id='$tag-ov-stand'><line x1='0' y1='$(f2(W.Y(ystand)))' x2='$(W.width)' y2='$(f2(W.Y(ystand)))' stroke='#7b2cbf' stroke-width='1.5' stroke-dasharray='8 5'/>",
        "<text x='$(W.width-24)' y='$(f2(W.Y(ystand)-5))' text-anchor='end' font-size='11' fill='#7b2cbf'>standing rider COM: deck $(p.deck_height) + ℓ_max $(OPT0.leg_max) = $(f2(ystand)) ($(si_len(ystand)))</text></g>")
    println(io, "<g id='$tag-ov-reach'><circle id='$tag-reachb' r='$(f2(W.scale*OPT0.leg_max))' fill='none' stroke='$COL_BACK' stroke-width='1' stroke-dasharray='4 4' opacity='0.6'/>",
        "<circle id='$tag-reachf' r='$(f2(W.scale*OPT0.leg_max))' fill='none' stroke='$COL_FRONT' stroke-width='1' stroke-dasharray='4 4' opacity='0.6'/></g>")
    println(io, "<g id='$tag-ov-trail'><polyline id='$tag-trailr' points='' fill='none' stroke='#d1495b' stroke-width='1.5' opacity='0.7'/>",
        "<polyline id='$tag-trailb' points='' fill='none' stroke='#202020' stroke-width='1.5' opacity='0.5'/></g>")
    println(io, "<g id='$tag-ov-arrows'>")
    for (nm, c) in (("b", COL_BACK), ("f", COL_FRONT))
        println(io, "<polyline id='$tag-arr$(nm)s' points='' stroke='$c' stroke-width='3.5' fill='none'/><polygon id='$tag-arr$(nm)h' points='' fill='$c'/>")
    end
    println(io, "</g>")
    println(io, "<line id='$tag-legb' x1='0' y1='0' x2='0' y2='0' stroke='$COL_BACK' stroke-width='4' stroke-linecap='round' opacity='0.85'/>")
    println(io, "<line id='$tag-legf' x1='0' y1='0' x2='0' y2='0' stroke='$COL_FRONT' stroke-width='4' stroke-linecap='round' opacity='0.85'/>")
    println(io, "<path id='$tag-deck' d='M 0 0' fill='none' stroke='#202020' stroke-width='10' stroke-linecap='round' stroke-linejoin='round'/>")
    for k in 1:2 println(io, "<circle id='$tag-w$k' r='$(f2(W.scale*WHEEL_R))' fill='#1261a0' stroke='#0b3d66' stroke-width='1.5'/>") end
    println(io, "<circle id='$tag-rider' r='9' fill='#d1495b' stroke='#7a1f2b' stroke-width='1.5'/>")
    println(io, "<circle id='$tag-cop' r='7' fill='#8e44ad' opacity='0.35'/>")
    println(io, "<text id='$tag-phase' x='20' y='40' font-size='15'></text>")
    bx0 = 20; bx1 = W.width-20; by = W.height-12
    println(io, "<line x1='$bx0' y1='$by' x2='$bx1' y2='$by' stroke='#bbb' stroke-width='4'/>")
    for tb in [F[findfirst(f->f.ph==ph, F)].t for ph in 1:3 if any(f->f.ph==ph, F)]
        xb = bx0 + (bx1-bx0)*tb/T; println(io, "<line x1='$(f2(xb))' y1='$(by-6)' x2='$(f2(xb))' y2='$(by+6)' stroke='#666' stroke-width='2'/>")
    end
    println(io, "<circle id='$tag-knob' cx='$bx0' cy='$by' r='6' fill='#333'/>")
    println(io, "<text id='$tag-time' x='$bx1' y='$(by-10)' text-anchor='end' font-size='13'></text>")
    println(io, "<text x='20' y='20' font-size='15' font-weight='bold'>$(h(label))</text>")
    lx = 20; ly = 62
    println(io, "<line x1='$lx' y1='$ly' x2='$(lx+30)' y2='$ly' stroke='$COL_BACK' stroke-width='3.5'/><text x='$(lx+36)' y='$(ly+4)' font-size='12'>back (tail) foot: leg and force on board</text>")
    println(io, "<line x1='$lx' y1='$(ly+18)' x2='$(lx+30)' y2='$(ly+18)' stroke='$COL_FRONT' stroke-width='3.5'/><circle cx='$(lx+15)' cy='$(ly+36)' r='7' fill='#8e44ad' opacity='0.35'/><text x='$(lx+36)' y='$(ly+40)' font-size='12'>COP of the foot forces on the deck (deck-normal components, weighted along the deck)</text><text x='$(lx+36)' y='$(ly+22)' font-size='12'>front foot: leg and force on board (arrow $(W.arrow_len) unit = $(f2(FMAX)) board weights = $(f2(FMAX*DIM.force_bw)) BW = $(round(FMAX*DIM.force)) N)</text>")
    println(io, "<line x1='$(f2(W.X(W.xmin+0.05)))' y1='$(f2(W.Y(W.ymin+0.06)))' x2='$(f2(W.X(W.xmin+0.55)))' y2='$(f2(W.Y(W.ymin+0.06)))' stroke='#666' stroke-width='3'/><text x='$(f2(W.X(W.xmin+0.3)))' y='$(f2(W.Y(W.ymin+0.06)-6))' text-anchor='middle' font-size='12'>0.5 length unit = $(si_len(0.5)) (x drifts at $(V_FORWARD)/s for visibility)</text>")
    # ---- side panels
    ts = [f.t for f in F]; tbs = [F[findfirst(f->f.ph==ph, F)].t for ph in 1:3 if any(f->f.ph==ph, F)]
    slog(x) = log10(1 + abs(x))
    Fbm = [hypot(f.Fbb...) for f in F]; Ffm = [hypot(f.Ffb...) for f in F]
    fticks = [(slog(v), string(v)) for v in (0, 1, 3, 10, 30, 100)]
    mini_panel(io, PX, PT, PW, PH, ts, [(slog.(Fbm), COL_BACK, "back foot", ""), (slog.(Ffm), COL_FRONT, "front foot", "")], T;
        title="|force on board| (board weights, log scale: log10(1+F))", yticks=fticks, vlines=tbs, ylo=0.0, cursor_id="$tag-cur1")
    KE = [f.KE for f in F]; PE = [f.PE for f in F]; Em = KE .+ PE
    nice(lo, hi) = begin r = hi-lo; st = 10.0^floor(log10(max(r,1e-9))); st = r/st < 2 ? st/5 : r/st < 5 ? st/2 : st; [(v, string(round(v, sigdigits=3))) for v in ceil(lo/st)*st:st:hi] end
    mini_panel(io, PX, PT+PH+PGAP, PW, PH, ts, [(KE, "#1261a0", "KE", ""), (PE, "#2a9d3f", "PE", ""), (Em, "#202020", "KE+PE", "")], T;
        title="mechanical energy (rider + board; board weight · length)", yticks=nice(min(minimum(KE),minimum(PE)), maximum(Em)), vlines=tbs, cursor_id="$tag-cur2")
    Wn = [f.Wnet for f in F]; Wp = [f.Wpos for f in F]; Wm = [f.Wneg for f in F]
    Wc = [f.Wnet - f.Eloss for f in F]                      # legs + collisions (impact work is negative)
    dEm = [f.KE + f.PE - (F[1].KE + F[1].PE) for f in F]    # ΔE_mech, the check: must coincide with Wc
    mini_panel(io, PX, PT+2PH+2PGAP, PW, PH, ts, [(Wp, "#2a9d3f", "positive", "4 3"), (Wm, "#d1495b", "negative", "4 3"), (Wn, "#202020", "net legs", ""), (Wc, "#8e44ad", "legs + collisions", ""), (dEm, "#e07b00", "ΔE mech", "2 2")], T;
        title="cumulative work: legs, legs + collisions, vs ΔE mech", yticks=nice(min(minimum(Wm),minimum(Wc),minimum(dEm)), max(maximum(Wp),maximum(Wc),maximum(dEm))), vlines=tbs, cursor_id="$tag-cur3")
    println(io, "<text x='$(PX+PW/2)' y='$(W.height-8)' text-anchor='middle' font-size='11' fill='#555'>time (nondimensional; dashed = phase starts; impact losses: tail strike $(f2(F[end].Eloss - (any(f->f.ph==3, F) ? F[end].wrench.KE_lost : 0.0)))$(any(f->f.ph==3, F) ? ", landing $(f2(F[end].wrench.KE_lost))" : ""))</text>")
    println(io, "</svg>")
    println(io, "<div class='ctl'><button id='$tag-play'>&#9646;&#9646; pause</button> ",
        "<input id='$tag-scrub' type='range' min='0' max='10000' value='0' style='width:60%'> ",
        "speed <select id='$tag-speed'>", join(["<option value='$v'$(v==speed ? " selected" : "")>×$v</option>" for v in (0.05, 0.1, 0.25, 0.5, 1.0)]), "</select>",
        "<span class='tog'> show: ", join(["<label><input type='checkbox' id='$tag-tg-$k'$(on ? " checked" : "")> $lbl</label>" for (k,lbl,on) in
            (("arrows","force arrows",true), ("stand","standing height",true), ("reach","leg reach ℓ_max",false), ("trail","COM trails",false))], " "), "</span></div>")
    # data + script. Frame layout: [t, ph, tail(2), kink(2), nose(2), w1(2), w2(2), front(2), rider(2), Fbb(2), Ffb(2), boardCOM(2), cop(2)]
    # (energies / leg work are drawn statically by mini_panel above; only the time cursors PX..PX+PW are driven from JS)
    println(io, "<script>(function(){")
    println(io, "const F=[", join(fr.(F), ","), "];")
    println(io, "const T=$(f2(T)), W={xoff:$(W.xoff),yoff:$(W.yoff),sc:$(W.scale),xmin:$(W.xmin),ymax:$(W.ymax),al:$(W.arrow_len),Fmax:$(FMAX),bx0:$bx0,bx1:$bx1,px:$PX,pw:$PW}, TS=$(DIM.time);")
    println(io, "const PH={", join(["$k:'$v'" for (k,v) in phases], ","), "};")
    println(io, raw"""
const g=id=>document.getElementById(id), tag=""" * "'$tag'" * raw""";
const X=x=>W.xoff+W.sc*(x-W.xmin), Y=y=>W.yoff+W.sc*(W.ymax-y);
const el={legb:g(tag+'-legb'),legf:g(tag+'-legf'),deck:g(tag+'-deck'),w1:g(tag+'-w1'),w2:g(tag+'-w2'),rider:g(tag+'-rider'),cop:g(tag+'-cop'),
  arrbs:g(tag+'-arrbs'),arrbh:g(tag+'-arrbh'),arrfs:g(tag+'-arrfs'),arrfh:g(tag+'-arrfh'),phase:g(tag+'-phase'),knob:g(tag+'-knob'),
  time:g(tag+'-time'),play:g(tag+'-play'),scrub:g(tag+'-scrub'),speed:g(tag+'-speed'),cur:[g(tag+'-cur1'),g(tag+'-cur2'),g(tag+'-cur3')],
  reachb:g(tag+'-reachb'),reachf:g(tag+'-reachf'),trailr:g(tag+'-trailr'),trailb:g(tag+'-trailb'),trail:g(tag+'-ov-trail')};
for(const k of ['arrows','stand','reach','trail']){ const cb=g(tag+'-tg-'+k), grp=g(tag+'-ov-'+k);
  const apply=()=>{ grp.style.display=cb.checked?'':'none'; if(!playing) draw(t); }; cb.onchange=apply; grp.style.display=cb.checked?'':'none'; }
const N=F.length; let i0=0; let t=0, playing=true, last=null;
function frameAt(t){ if(t<=F[0][0]) return F[0].slice(); if(t>=F[N-1][0]) return F[N-1].slice();
  if(!(F[i0][0]<=t && t<F[i0+1][0])){ i0=0; while(F[i0+1][0]<=t) i0++; }
  const a=F[i0], b=F[i0+1], u=(t-a[0])/(b[0]-a[0]); const o=new Array(a.length);
  for(let k=0;k<a.length;k++) o[k]=a[k]+u*(b[k]-a[k]); o[1]=a[1]; return o; }
function arrow(foot,Fv){ const mag=Math.hypot(Fv[0],Fv[1]); if(mag<1e-9){const x=X(foot[0]),y=Y(foot[1]);return[`${x},${y} ${x},${y}`,`${x},${y} ${x},${y} ${x},${y}`];}
  const len=W.al*mag/W.Fmax, ux=Fv[0]/mag, uy=Fv[1]/mag, tip=[foot[0]+len*ux,foot[1]+len*uy];
  const hs=Math.min(0.06,0.6*len), base=[tip[0]-hs*ux,tip[1]-hs*uy], nx=-uy, ny=ux, hw=0.5*hs;
  const P=q=>`${X(q[0])},${Y(q[1])}`;
  return [P(foot)+' '+P(base), P(tip)+' '+P([base[0]+hw*nx,base[1]+hw*ny])+' '+P([base[0]-hw*nx,base[1]-hw*ny])]; }
function draw(t){ const f=frameAt(t); const tail=[f[2],f[3]],kink=[f[4],f[5]],nose=[f[6],f[7]],w1=[f[8],f[9]],w2=[f[10],f[11]],front=[f[12],f[13]],rider=[f[14],f[15]],Fb=[f[16],f[17]],Ff=[f[18],f[19]];
  el.legb.setAttribute('x1',X(tail[0]));el.legb.setAttribute('y1',Y(tail[1]));el.legb.setAttribute('x2',X(rider[0]));el.legb.setAttribute('y2',Y(rider[1]));
  el.legf.setAttribute('x1',X(front[0]));el.legf.setAttribute('y1',Y(front[1]));el.legf.setAttribute('x2',X(rider[0]));el.legf.setAttribute('y2',Y(rider[1]));
  el.deck.setAttribute('d',`M ${X(tail[0])} ${Y(tail[1])} L ${X(kink[0])} ${Y(kink[1])} L ${X(nose[0])} ${Y(nose[1])}`);
  el.w1.setAttribute('cx',X(w1[0]));el.w1.setAttribute('cy',Y(w1[1]));el.w2.setAttribute('cx',X(w2[0]));el.w2.setAttribute('cy',Y(w2[1]));
  el.rider.setAttribute('cx',X(rider[0]));el.rider.setAttribute('cy',Y(rider[1]));
  el.cop.setAttribute('cx',X(f[22]));el.cop.setAttribute('cy',Y(f[23]));
  let a=arrow(tail,Fb); el.arrbs.setAttribute('points',a[0]); el.arrbh.setAttribute('points',a[1]);
  a=arrow(front,Ff); el.arrfs.setAttribute('points',a[0]); el.arrfh.setAttribute('points',a[1]);
  el.reachb.setAttribute('cx',X(tail[0]));el.reachb.setAttribute('cy',Y(tail[1]));el.reachf.setAttribute('cx',X(front[0]));el.reachf.setAttribute('cy',Y(front[1]));
  if(el.trail.style.display!=='none'){ let pr='',pb=''; for(let k=0;k<N && F[k][0]<=t;k++){ pr+=X(F[k][14])+','+Y(F[k][15])+' '; pb+=X(F[k][20])+','+Y(F[k][21])+' '; }
    pr+=X(rider[0])+','+Y(rider[1]); pb+=X(f[20])+','+Y(f[21]); el.trailr.setAttribute('points',pr); el.trailb.setAttribute('points',pb); }
  el.phase.textContent='phase '+f[1]+': '+(PH[f[1]]||'');
  el.knob.setAttribute('cx',W.bx0+(W.bx1-W.bx0)*t/T);
  const cx=W.px+W.pw*Math.min(Math.max(t,0),T)/T; for(const c of el.cur){ c.setAttribute('x1',cx); c.setAttribute('x2',cx); }
  el.time.textContent='t = '+t.toFixed(3)+' / '+T.toFixed(3)+'  (= '+(t*TS).toFixed(3)+' s)';
  el.scrub.value=Math.round(10000*t/T); }
function step(now){ if(playing){ if(last!==null){ t+= (now-last)/1000*parseFloat(el.speed.value)/TS; if(t>T){ t=0; } } last=now; draw(t);} else last=now; requestAnimationFrame(step); }
el.play.onclick=()=>{ playing=!playing; el.play.innerHTML=playing?'&#9646;&#9646; pause':'&#9654; play'; };
el.scrub.oninput=()=>{ playing=false; el.play.innerHTML='&#9654; play'; t=T*el.scrub.value/10000; draw(t); };
draw(0); requestAnimationFrame(step);
})();</script></div>""")
    String(take!(io))
end

"Static strip of snapshots, same scale in every cell."
function snapshots_svg(F; ncol=4, nrow=2, cellw=300, cellh=340, label="")
    ids = round.(Int, range(1, length(F), length=ncol*nrow)); width = ncol*cellw; height = nrow*cellh + 30
    io = IOBuffer(); println(io, "<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height' font-family='Helvetica,Arial,sans-serif'><rect width='100%' height='100%' fill='white'/>")
    println(io, "<text x='10' y='20' font-size='15' font-weight='bold'>$(h(label))</text>")
    # shared scale: fit the biggest per-frame extent
    ext = maximum(max(f.rider[2], f.nose[2], f.tail[2], 0.7) + 0.3 for f in F); scale = min((cellh-50)/ext, (cellw-20)/1.9)
    for (j,k) in enumerate(ids)
        f = F[k]; col = (j-1)%ncol; row = (j-1)÷ncol; left = col*cellw; top = 30 + row*cellh; ground = top+cellh-30; cx = left+cellw/2
        W = (; X = x -> cx + scale*(x-f.xo), Y = y -> ground - scale*y, scale, arrow_len=0.5)
        println(io, "<rect x='$left' y='$top' width='$cellw' height='$cellh' fill='none' stroke='#ddd'/><line x1='$left' y1='$ground' x2='$(left+cellw)' y2='$ground' stroke='#555' stroke-width='2'/>")
        draw_frame(io, W, f)
        println(io, "<text x='$(left+6)' y='$(top+16)' font-size='11'>t=$(r3(f.t)) phase $(f.ph)  θ=$(r3(f.θ))  |Fb|=$(r3(hypot(f.Fbb...)))  |Ff|=$(r3(hypot(f.Ffb...)))</text>")
    end
    println(io, "</svg>"); String(take!(io))
end

# ------------------------------------------------------------------ line plots (2-column figure)
function line_panel(io, ox, oy, xs, series; width=440, height=230, xlabel="", ylabel="", title="", vlines=Float64[])
    vals = reduce(vcat, [filter(isfinite, Float64.(ys)) for (_,ys,_) in series])
    xmin,xmax = extrema(xs); ymin,ymax = extrema(vals); ymax==ymin && (ymax=ymin+1)
    L,R,T,B = 60,15,28,42; pw = width-L-R; ph = height-T-B
    X(x) = ox+L+pw*(x-xmin)/(xmax-xmin); Y(y) = oy+T+ph*(ymax-y)/(ymax-ymin)
    colors = ("#1261a0","#d1495b","#2a9d8f","#7b2cbf","#e69f00","#56b4e9")
    println(io, "<text x='$(ox+L)' y='$(oy+16)' font-size='13' font-weight='bold'>$(h(title))</text>")
    println(io, "<line x1='$(ox+L)' y1='$(oy+T+ph)' x2='$(ox+L+pw)' y2='$(oy+T+ph)' stroke='black'/><line x1='$(ox+L)' y1='$(oy+T)' x2='$(ox+L)' y2='$(oy+T+ph)' stroke='black'/>")
    for v in vlines println(io, "<line x1='$(f2(X(v)))' y1='$(oy+T)' x2='$(f2(X(v)))' y2='$(oy+T+ph)' stroke='#999' stroke-dasharray='3 3'/>") end
    ymin < 0 < ymax && println(io, "<line x1='$(ox+L)' y1='$(f2(Y(0)))' x2='$(ox+L+pw)' y2='$(f2(Y(0)))' stroke='#ccc'/>")
    for j in 0:4
        x = xmin+j*(xmax-xmin)/4; y = ymin+j*(ymax-ymin)/4
        println(io, "<text x='$(f2(X(x)))' y='$(oy+T+ph+16)' text-anchor='middle' font-size='11'>$(r3(x))</text><text x='$(ox+L-6)' y='$(f2(Y(y)+4))' text-anchor='end' font-size='11'>$(r3(y))</text>")
    end
    for (i,(name,ys,style)) in enumerate(series)
        pts = join(["$(f2(X(x))),$(f2(Y(y)))" for (x,y) in zip(xs,ys) if isfinite(y)], " ")
        println(io, "<polyline points='$pts' fill='none' stroke='$(colors[i])' stroke-width='2' $style/>")
        lx = ox+L+8+(i-1)*(pw/length(series)); println(io, "<line x1='$lx' y1='$(oy+T+8)' x2='$(lx+18)' y2='$(oy+T+8)' stroke='$(colors[i])' stroke-width='2' $style/><text x='$(lx+22)' y='$(oy+T+12)' font-size='11'>$(h(name))</text>")
    end
    println(io, "<text x='$(ox+L+pw/2)' y='$(oy+height-6)' text-anchor='middle' font-size='12'>$(h(xlabel))</text><text transform='translate($(ox+14) $(oy+T+ph/2)) rotate(-90)' text-anchor='middle' font-size='12'>$(h(ylabel))</text>")
end

function figure_svg(F, d; label="")
    t = [f.t for f in F]; sw = [d.T0, d.T0+d.T1]; dash = "stroke-dasharray='6 3'"
    panels = [
        ("COM heights", "height", [("board COM", [f.yo for f in F], ""), ("rider COM", [f.rider[2] for f in F], dash)]),
        ("deck angle", "θ (rad)", [("θ", [f.θ for f in F], "")]),
        ("back-foot force on board", "force (board weights)", [("Fx", [f.Fbb[1] for f in F], ""), ("Fy", [f.Fbb[2] for f in F], dash), ("|F|", [hypot(f.Fbb...) for f in F], "stroke-dasharray='2 3'")]),
        ("front-foot force on board", "force (board weights)", [("Fx", [f.Ffb[1] for f in F], ""), ("Fy", [f.Ffb[2] for f in F], dash), ("|F|", [hypot(f.Ffb...) for f in F], "stroke-dasharray='2 3'")]),
        ("front-foot deck position s", "s (body x)", [("s", [f.s for f in F], "")]),
        ("lowest board point and tail tip", "height", [("lowest point", [f.lowest for f in F], ""), ("tail tip", [f.tail_y for f in F], dash)]),
        ("leg slacks ℓmax − ℓ (push allowed when > 0)", "length", [("back", [1 - f.ℓb for f in F], ""), ("front", [1 - f.ℓf for f in F], dash)]),
        ("board ground gaps (flight only)", "height", [("tail", [f.ph==2 ? f.yo+Yr(f.θ,rt) : NaN for f in F], ""), ("rear wheel pt", [f.ph==2 ? f.yo+Yr(f.θ,rc) : NaN for f in F], dash), ("front wheel pt", [f.ph==2 ? f.yo+Yr(f.θ,rf) : NaN for f in F], "stroke-dasharray='2 3'")]),
    ]
    W = 900; Hp = 230; nrow = cld(length(panels), 2); io = IOBuffer()
    println(io, "<svg xmlns='http://www.w3.org/2000/svg' width='$W' height='$(30+nrow*Hp)' viewBox='0 0 $W $(30+nrow*Hp)' font-family='Helvetica,Arial,sans-serif'><rect width='100%' height='100%' fill='white'/>")
    println(io, "<text x='10' y='20' font-size='15' font-weight='bold'>$(h(label)) — dashed verticals: phase switches</text>")
    for (i,(title,yl,ser)) in enumerate(panels)
        col = (i-1)%2; row = (i-1)÷2
        line_panel(io, 10+col*445, 30+row*Hp, t, ser; xlabel="time", ylabel=yl, title=title, vlines=sw)
    end
    println(io, "</svg>"); String(take!(io))
end

# ------------------------------------------------------------------ render
Fs = [frames(c.d) for c in cases]                 # optimized horizon: figures, snapshots
Fa = coast_frames(Fs[2], cases[2].d)              # + simulated coast: the (one) animation, solve B
wr = landing_wrench(cases[2].d); println("landing wrench B: ", wr)
files = String[]
function save(name, s) path = joinpath(REPORT_DIR, name); write(path, s); push!(files, path); name end
W = window(Fa; width=880, height=480)
labB = NAMES[2]*(cases[2].converged ? "" : "  [NOT CONVERGED: $(cases[2].status)]")
anim_html = animation_html(Fa, W; label=labB, tag="B")
for (i,c) in enumerate(cases)
    tag = TAGS[i]; lab = NAMES[i]*(c.converged ? "" : "  [NOT CONVERGED: $(c.status)]")
    save("ollie_rider_$(tag)_snapshots.svg", snapshots_svg(Fs[i]; label=lab))
    save("ollie_rider_$(tag)_figure.svg", figure_svg(Fs[i], c.d; label=lab))
end
for stale in ("ollie_rider_A_animation.svg", "ollie_rider_B_animation.svg", "ollie_rider_AB_animation.svg")
    f = joinpath(REPORT_DIR, stale); isfile(f) && rm(f)   # SMIL animations superseded
end

# ------------------------------------------------------------------ report
opt = RiderOptions()
tbl(rows; head=nothing) = "<table>" * (isnothing(head) ? "" : "<tr>"*join("<th>".*h.(head).*"</th>")*"</tr>") * join(["<tr>"*join("<td>".*h.(string.(r)).*"</td>")*"</tr>" for r in rows]) * "</table>"
param_rows = [("rider mass M", "$(opt.rider_mass) board masses", "$(SC.rider_mass_kg) kg rider → board $(round(DIM.mass_board_kg,digits=2)) kg"),
    ("leg force limit", "F ≤ $(opt.force_limit_bw) BW per leg = $(f2(FMAX)) board weights", "$(si_force(FMAX)) per leg"),
    ("leg length", "$(opt.leg_min) ≤ ℓ ≤ $(opt.leg_max)", "ℓ_max = $(SC.leg_length_m) m; push only while ℓ ≤ 1: F(ℓ−1) ≤ 0"),
    ("time unit", "√(ℓ_max/g) = 1", "$(round(DIM.time,digits=3)) s"), ("velocity unit", "√(g ℓ_max) = 1", "$(round(DIM.velocity,digits=2)) m/s"),
    ("restitution e", p.restitution, "tail strike, board only"),
    ("ground-phase force direction", "along the leg (foot → rider)", "force_dir=:leg"), ("flight force direction", "cone: Fy on board ≤ 0, Fx free, |F| ≤ $(f2(FMAX))", "flight_force_dir=:cone"),
    ("front foot", "free on the deck, s ∈ [$(opt.front_point_bounds[1]), $(opt.front_point_bounds[2])]", "starts over the front wheel s=$(opt.front_point0)"), ("back foot", "tail tip (−0.40, 0.06)", "fixed"),
    ("ground", "ice (no friction), board slides in x", "vx0 = $(opt.vx0)"), ("rider clearance", opt.rider_clearance, "COM above deck line, board frame"),
    ("board", "deck 0.80, wheelbase 0.58, deck height 0.08, tail rise 0.06, I=$(p.inertia)", "deck $(si_len(0.8)), wheelbase $(si_len(0.58))"), ("solver", "Ipopt, tol=1e-4, max_iter=10000, nodes 21/31/61", "force-rate weight 1e-4")]
obj_rows = [("A: board_apex", "board COM height y_b at the mid flight node", "v_yb(mid)=0", "—"),
    ("B: lowest_point", "z_low: lower bound on height of tail, nose, rear and front wheel points at mid node", "v_yb(mid)=0; z_low ≤ y_b + Y(θ, r) for each r", "warm-started from the board-apex optimum (stage 2 inside solve_rider_ollie)")]
act_rows = [("0 load (θ=0, y fixed)", "along leg, foot→rider; on board −F û", "0 ≤ F ≤ 50", "F(ℓ−1) ≤ 0, ℓ ≥ 0.25", "wheel reactions R ≥ 0; exit when R_front=0"),
    ("1 pop (pivot on rear wheel)", "along leg, foot→rider; on board −F û", "0 ≤ F ≤ 50", "F(ℓ−1) ≤ 0, ℓ ≥ 0.25", "R_rear ≥ 0, tail and front wheel above ground; ends with tail strike"),
    ("2 flight", "cone: on rider (G, F) with F ≥ 0; on board (−G, −F)", "G² + F² ≤ N², 0 ≤ N ≤ 50", "N(ℓ−1) ≤ 0 (magnitude variable gated), ℓ ≥ 0.25 (same foot points)", "no ground contact; level landing θ=0, ω=0, both legs ℓ ≤ 1 and COM between feet"),
    ("all", "rider COM ≥ 0.15 above deck line", "", "", "force-rate penalty on F, s, G"),
    ("flight (new)", "feet on board: both legs within reach throughout flight, ℓ ≤ 1", "", "", "feet_on_board=true"),
    ("all (new)", "rider above feet: rider COM ≥ 0.2 above both foot points in world y", "", "", "rider_above_feet=true, foot_margin=0.2")]
# essentials only: outcome, timing, the two apexes, and the four "is the physics right" numbers
audit_keys = [:status, :T0, :T1, :T2, :theta_hit, :lowest_point_apex, :board_apex_rise, :rider_rise, :theta_at_mid, :peak_force,
    :energy_residual_total, :energy_residual_rel, :max_constraint_violation, :impact_velocity_error, :angular_NE_residual,
    :horizontal_momentum_drift, :min_gap, :landing_vy_board, :landing_vy_rider]
fmt(v) = v isa Number ? (0 < abs(v) < 1e-3 ? @sprintf("%.2e", v) : string(round(v, digits=5))) : v isa Tuple ? "(" * join(string.(round.(collect(v), digits=4)), ", ") * ")" : string(v)
const SI_LEN = (:board_apex_rise, :lowest_point_apex, :tail_apex_rise, :rider_apex, :rider_rise, :min_gap)
const SI_TIME = (:T0, :T1, :T2); const SI_FORCE = (:peak_force, :initial_forces); const SI_VEL = (:landing_vy_board, :landing_vy_rider)
si(k, v) = k in SI_LEN ? si_len(v) : k in SI_TIME ? si_time(v) : k in SI_FORCE ? (v isa Tuple ? join(si_force.(v), "; ") : si_force(v)) :
           k in SI_VEL ? "$(round(v*DIM.velocity,digits=2)) m/s" : k === :impulse ? "$(round(v*DIM.force*DIM.time,digits=1)) N·s" : "—"
audit_rows = [(string(k), fmt(getfield(cases[1].a, k)), fmt(getfield(allB[1].a, k)), fmt(getfield(allB[2].a, k)), si(k, getfield(cases[2].a, k))) for k in audit_keys]
audit_html = tbl(audit_rows; head=("quantity", "A board_apex", "B lowest_point (warm from A)" * (Bsel==1 ? " [rendered]" : ""), "B lowest_point (cold start)" * (Bsel==2 ? " [rendered]" : ""), "rendered B in SI ($(SC.rider_mass_kg) kg, ℓ=$(SC.leg_length_m) m)"))
convcases = [cases[1], allB[1], allB[2]]; convnames = [NAMES[1], "B (warm start from A)", "B (cold start)"]
conv = join(["<li><b>$(convnames[i])</b>: status $(h(convcases[i].status)), objective value $(r4(convcases[i].obj_value)), lowest-point apex $(r4(convcases[i].a.lowest_point_apex)), θ(mid) $(r4(convcases[i].a.theta_at_mid)), wall time $(round(convcases[i].wall,digits=0)) s$(convcases[i].converged ? "" : " — <b>did not converge; rendered as produced</b>")</li>" for i in 1:3]) * "<li>Rendered B variant: $(Bsel==1 ? "warm start" : "cold start") (higher lowest-point apex with a level board).</li>"
aA, aB = cases[1].a, cases[2].a
interp = """
<p>Solve A lifts the board COM by $(r3(aA.board_apex_rise)) with the deck pitched to θ=$(r3(aA.theta_at_mid)) rad at the apex, so its lowest board point (the rear wheel/tail side) reaches only $(r3(aA.lowest_point_apex)); the tail-strike angle is $(r3(aA.theta_hit)) rad and the flight lasts $(r3(aA.T2)).
Solve B instead maximises the height of the lowest board point at the apex and reaches $(r3(aB.lowest_point_apex)) (board COM rise $(r3(aB.board_apex_rise)), apex pitch $(r3(aB.theta_at_mid)) rad), i.e. it trades COM height for a level board in flight, which is what a real ollie is judged on.
Solve B\'s lowest-point apex of $(r3(aB.lowest_point_apex)) is $(si_len(aB.lowest_point_apex)) above the ice with the rider at $(si_len(aB.rider_apex)); flight lasts $(si_time(aB.T2)).
The levelling is done with the flight cone: the front foot presses down on the nose (negative Fy on the board with a free horizontal component) while the back foot pulls its force to zero, so the pitch rate built by the tail strike is cancelled before the apex; on the ground both feet act along the legs and the pop is a back-foot dominated push (peak forces $(r3(aA.peak_force)) and $(r3(aB.peak_force)) board weights = $(r3(aA.peak_force*DIM.force_bw)) / $(r3(aB.peak_force*DIM.force_bw)) BW, at the $(f2(FMAX)) limit).
In both solves the reach complementarity is what shapes the timing: a leg can only push while shorter than 1, so the rider extends the legs during the pop and then the board must rise into the feet during flight; the leg-slack panel shows where each leg goes slack.
Constraint, impact, angular-NE and momentum audits sit at solver precision (max constraint violation $(fmt(aA.max_constraint_violation)) / $(fmt(aB.max_constraint_violation)), impact velocity error $(fmt(aA.impact_velocity_error)) / $(fmt(aB.impact_velocity_error)), angular NE residual $(fmt(aA.angular_NE_residual)) / $(fmt(aB.angular_NE_residual)), momentum drift $(fmt(aA.horizontal_momentum_drift)) / $(fmt(aB.horizontal_momentum_drift))), so the A/B differences are due to the objectives. <b>The energy audit is NOT small:</b> Σ(ΔE − W) = $(r3(aA.energy_residual_total)) / $(r3(aB.energy_residual_total)) ≈ $(round(100*abs(aB.energy_residual_rel)))% of the leg work, split $(r3(aA.energy_residual_support)) / $(r3(aB.energy_residual_support)) in load+pop and $(r3(aA.energy_residual_flight)) / $(r3(aB.energy_residual_flight)) in flight (where the legs do almost no work, W = $(r3(aB.leg_work_flight))). This is the first-order symplectic-Euler discretisation at 21/31/61 nodes, not a modelling error (the dynamics constraints are satisfied to 1e-8); the mesh-refinement check (residual must halve at 2× nodes) has not been run for this model yet. The horizontal drift in the animation is a visual offset of $(V_FORWARD)/s, not part of the dynamics (vx0=0).</p>"""
html = """<!DOCTYPE html><html><head><meta charset='utf-8'><title>Rider + board ollie: vector animations and objective comparison</title>
<style>body{font-family:Helvetica,Arial,sans-serif;max-width:1800px;margin:20px auto;padding:0 16px}.ctl{margin:6px 0;font-size:13px}.tog{margin-left:12px}.ctl button{font-size:13px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:3px 8px;font-size:13px;text-align:left}th{background:#eee}figure{margin:12px 0}object{max-width:100%}</style></head><body>
<h1>Rider + board ollie: SMIL-animated SVG, objectives and actuator constraints</h1>
<p>Model <code>ollie_rider.jl</code> (OllieRider). Three phases with free durations: load (flat board sliding on ice), pop (rear-wheel pivot, reduced coordinates), tail strike (Newton impact on the board only, e=$(p.restitution)), flight, level landing. Rider is a point mass on two push-only telescoping legs; back foot on the tail tip, front foot free on the deck. Arrows show the force applied <b>on the board</b> at each foot (orange back/tail foot, green front foot), one shared scale: 0.5 length unit = $(f2(FMAX)) board weights = $(opt.force_limit_bw) rider body weights. <b>Units:</b> everything is nondimensional (board mass 1, g 1, ℓ_max 1); SI values use a $(SC.rider_mass_kg) kg rider and ℓ_max = $(SC.leg_length_m) m (board $(round(DIM.mass_board_kg,digits=2)) kg, time unit $(round(DIM.time,digits=3)) s, force unit $(round(DIM.force,digits=1)) N).</p>
<h2>Solves</h2><ul>$conv</ul>
<h2>Parameters</h2>$(tbl(param_rows; head=("parameter","value","note")))
<h2>Objectives</h2>$(tbl(obj_rows; head=("objective","maximised quantity","apex-node conditions","initialisation")))
<h2>Actuator constraints per phase</h2>$(tbl(act_rows; head=("phase","force direction set","magnitude bound","reach / push-only","other")))
<p>Without the two new constraints (feet within reach throughout flight; rider COM at least 0.2 above both foot points) the optimizer throws the board vertically above the rider's head and catches it, reaching a lowest-point apex of 2.34 instead of 1.80.</p>
<h2>Animation: solve B (lowest board point), scrubbable</h2>
<p>Only solve B is animated; A is shown as snapshots and time series below. Inline SVG driven by requestAnimationFrame, interpolating linearly between the $(length(Fa)) frames in physical time; drag the slider to scrub, ▶/▮▮ to play/pause, speed ×0.25 = 4× slow motion. The three panels to the right, with a red time cursor synced to the animation and dashed phase-start lines, show (1) the magnitude of the force each foot applies to the board on a log10(1+F) axis (board weights: ≈1 in flight, tens during the pop), (2) kinetic, potential and total mechanical energy of rider + board (board translation and rotation included; board weight · length units), and (3) cumulative work: leg work (the within-phase mechanical-energy change) split into positive and negative parts plus the net, then <b>legs + collisions</b> (purple), which adds the negative work of the tail-strike and landing impacts; it must coincide with the dashed orange ΔE<sub>mech</sub> = KE+PE − (KE+PE)(0) curve — that overlay is the energy-accounting check (impact losses in the panel footer). Time reads nondimensional and seconds ($(round(DIM.time,digits=3)) s per unit). After touchdown a $(POST_LANDING)-unit <b>simulated</b> ride-away is appended that is not part of the optimization: landing is level (θ=0, ω=0), both wheels touch the ice together, the touchdown wrench is two vertical wheel impulses with e=0 vertically and e=1 horizontally (frictionless ice): vy⁻=$(r3(wr.vy_before)), ω⁻=$(r3(wr.om_before)) → J_rear=$(r3(wr.J1)), J_front=$(r3(wr.J2)), ΣJ_y=$(r3(wr.Jy)), ride-away vx=$(r3(wr.vx_before)) (KE lost $(r3(wr.KE_lost))). The rider's vertical velocity is brought to rest over $(ABSORB_T) units; arrows then show the implied absorbing leg forces M(a+g)/2 per foot, then body weight. Figures and audits cover only the optimized horizon.</p>
<figure>$(anim_html)</figure>
<h2>Snapshots</h2>
<figure><object data='ollie_rider_A_snapshots.svg' type='image/svg+xml' style='width:1200px;height:710px'></object></figure>
<figure><object data='ollie_rider_B_snapshots.svg' type='image/svg+xml' style='width:1200px;height:710px'></object></figure>
<h2>Time series</h2>
<figure><object data='ollie_rider_A_figure.svg' type='image/svg+xml' style='width:900px;height:950px'></object></figure>
<figure><object data='ollie_rider_B_figure.svg' type='image/svg+xml' style='width:900px;height:950px'></object></figure>
<h2>Audit</h2>
<p>energy_residual_total = Σ(ΔE − W_legs) over all phases (impact energy change excluded, it is accounted separately); energy_residual_rel divides by the total leg work. max_constraint_violation = largest violation of any model constraint or bound at the returned point (JuMP primal_feasibility_report, atol 0) — should be ≲ the solver tolerance 1e-4.</p>
$audit_html
<h2>Interpretation</h2>$interp
$(FORCE_LIMITS_SECTION)
$(read(joinpath(@__DIR__,"reports","ollie_rider_appendix_inertia.html"),String))
$(read(joinpath(@__DIR__,"reports","ollie_rider_appendix_fv.html"),String))
<p><b>Force–velocity headline:</b> the appendix-2 test shows the unconstrained model's ollie depends on legs pushing at up to 2 BW while extending at 9–17 m/s; with a Hill limit (v_max 1.2 ≈ 3.6 m/s) the peak leg force drops to ~0.6 BW and the lowest-point apex falls from 1.46 m to 0.20 m — a realistic ollie height. The solves in this report are still run with <code>force_velocity=false</code>.</p>
</body></html>"""
save("ollie_rider_svg_report.html", html)
println("wrote:"); foreach(println, files)
println("\nAUDIT (A | B warm | B cold)"); for r in audit_rows println(rpad(r[1],28), rpad(r[2],30), rpad(r[3],30), r[4]) end
