#!/usr/bin/env julia
# Report for the 3-D rider + board ollie / kickflip (ollie_rider_3D.jl).
# No solves here: reads the serialized stage files in simulation_reports/ and writes a
# self-contained HTML (inline SVG + JS, no external libraries):
#   stage 1  simulation_reports/ollie3d_stage1.jls      (; g, d, a)      3-D ollie, lateral locked
#   stage 2  simulation_reports/kickflip3d_stage2.jls   (; d, a, opt)    pop frozen, flight optimized (kickflip)
#   stage 3  simulation_reports/kickflip3d_stage3.jls   (; d, a, opt)    everything free (optional)
#
#   julia --project=. generate_kickflip_3d_report.jl
# Output: simulation_reports/kickflip_3d_report.html
include(joinpath(@__DIR__, "ollie_rider_3D.jl")); using .OllieRider3D; const rotmat = OllieRider3D.rotmat
using Serialization, LinearAlgebra, Printf

const REPORT_DIR = joinpath(@__DIR__, "simulation_reports"); mkpath(REPORT_DIR)
const OUT = joinpath(REPORT_DIR, "kickflip_3d_report.html")
const B = Board3D()
const HALFW = B.deck_width/2                # 0.1175
const TAIL_B = [0.0, -B.deck_length/2, B.tail_rise]   # (0, −0.4, 0.06)
const NOSE_B = [0.0,  B.deck_length/2, 0.0]           # (0, 0.4, 0)
const WHEELS_B = [[sx*B.axle_half, sy*B.wheelbase/2, -B.deck_height] for sy in (-1, 1) for sx in (-1, 1)]  # (±0.12, ±0.29, −0.08)
const KINK_Y = -0.32                        # tail region starts here (body y)
const DECK_B = [[-HALFW, KINK_Y, 0.0], [HALFW, KINK_Y, 0.0], [HALFW, B.deck_length/2, 0.0], [-HALFW, B.deck_length/2, 0.0]]
const TAILQ_B = [[-HALFW, -B.deck_length/2, B.tail_rise], [HALFW, -B.deck_length/2, B.tail_rise], [HALFW, KINK_Y, 0.0], [-HALFW, KINK_Y, 0.0]]
const ARROW_LEN = 0.5                       # world length units for FORCE_REF board weights
const FORCE_REF = 50.0
const COL_BACK = "#e07b00"; const COL_FRONT = "#2a9d3f"; const COL_RIDER = "#d1495b"

h(s) = replace(string(s), "&"=>"&amp;", "<"=>"&lt;", ">"=>"&gt;")
jnum(x) = (x isa Real && isfinite(x)) ? string(round(Float64(x), digits=5)) : "null"   # NaN/Inf → null
jarr(v) = "[" * join(jnum.(v), ",") * "]"
f2(x) = string(round(x, digits=2))
fmt(x) = x isa Real ? (isfinite(x) ? (abs(x) < 1e-3 && x != 0 ? @sprintf("%.3e", x) : string(round(x, digits=4))) : string(x)) :
         x isa Tuple ? "(" * join(fmt.(x), ", ") * ")" : string(x)

# ------------------------------------------------------------------ data
function load_case(file, label, note)
    isfile(file) || return nothing
    x = deserialize(file)
    (; label, note, d=x.d, a=x.a, opt=haskey(x, :opt) ? x.opt : nothing, file)
end
cases = filter(!isnothing, [
    load_case(joinpath(REPORT_DIR, "ollie3d_stage1.jls"),   "Stage 1: 3-D ollie, lateral locked",
              "lateral DOFs locked (σ = 0, x ≡ 0); should reproduce the 2-D ollie (lowest-point apex 1.804)"),
    load_case(joinpath(REPORT_DIR, "kickflip3d_stage2.jls"), "Stage 2: kickflip, pop frozen to the ollie, flight optimized",
              "phases 0/1 and the tail strike fixed to stage 1; flight free with roll target 2π, mu = 0 (normal pushes only)"),
    load_case(joinpath(REPORT_DIR, "kickflip3d_stage3.jls"), "Stage 3: kickflip, everything free",
              "load, pop and flight all optimized together with the roll target 2π"),
])
isempty(cases) && error("no stage files found in $(REPORT_DIR)")
println("cases: ", join([c.label for c in cases], " | "))

"Per-node frames across the three phases: world points computed with rotmat."
function build_frames(d)
    frames = NamedTuple[]
    ctrl(v, k) = v[min(k, length(v))]
    function push_frame!(t, ph, xb, R, xr, Fb_w, Ff_w, front_b)
        W(p) = xb .+ R*p
        push!(frames, (; t, ph, rider=xr, tail=W(TAIL_B), nose=W(NOSE_B), wheels=W.(WHEELS_B), front=W(front_b),
                         deck=W.(DECK_B), tailq=W.(TAILQ_B), Fb=Fb_w, Ff=Ff_w))
    end
    legforce(foot, xr, F) = begin u = xr .- foot; n = norm(u); n < 1e-9 ? zeros(3) : -F .* u ./ n end   # force ON the board
    for (ph, t, xb, yb, zb, th, xr, yr, zr, Fb, Ff, s) in (
            (0, d.t0, d.xb0, d.yb0, d.zb0, zeros(length(d.t0)), d.xr0, d.yr0, d.zr0, d.Fb0, d.Ff0, d.s0),
            (1, d.t1, d.xb1, d.yb1, d.zb1, d.th1,               d.xr1, d.yr1, d.zr1, d.Fb1, d.Ff1, d.s1))
        for k in eachindex(t)
            R = rotmat(0.0, th[k], 0.0); c = [xb[k], yb[k], zb[k]]; r = [xr[k], yr[k], zr[k]]
            front_b = [0.0, ctrl(s, k), 0.0]
            push_frame!(t[k], ph, c, R, r, legforce(c .+ R*TAIL_B, r, ctrl(Fb, k)), legforce(c .+ R*front_b, r, ctrl(Ff, k)), front_b)
        end
    end
    for k in eachindex(d.t2)
        R = rotmat(d.ps2[k], d.th2[k], d.ph2[k]); c = [d.xb2[k], d.yb2[k], d.zb2[k]]; r = [d.xr2[k], d.yr2[k], d.zr2[k]]
        front_b = [ctrl(d.sg2, k), ctrl(d.s2, k), 0.0]
        Fb_w = R*[ctrl(d.Gb1, k), ctrl(d.Gb2, k), -ctrl(d.Nb2, k)]
        Ff_w = R*[ctrl(d.Gf1, k), ctrl(d.Gf2, k), -ctrl(d.Nf2, k)]
        push_frame!(d.t2[k], 2, c, R, r, Fb_w, Ff_w, front_b)
    end
    frames
end
# frame layout (flat): [t, ph, rider3, tail3, nose3, wheels 4×3, front3, deck 4×3, tailq 4×3, Fb3, Ff3] = 56 numbers
frame_json(f) = "[" * join(jnum.(vcat(f.t, f.ph, f.rider, f.tail, f.nose, f.wheels..., f.front, f.deck..., f.tailq..., f.Fb, f.Ff)), ",") * "]"

# ------------------------------------------------------------------ static time-series panel
function nice_ticks(lo, hi)
    lo == hi && (lo -= 0.5; hi += 0.5)
    r = hi - lo; st = 10.0^floor(log10(r)); q = r/st
    st = q < 2 ? st/5 : q < 5 ? st/2 : st
    [(v, string(round(v, sigdigits=3))) for v in ceil(lo/st)*st:st:hi]
end
"""
series: vector of (ts, vals, color, label, dash). Draws axes, ticks, phase lines, legend, and a cursor line with
id `cursor_id` carrying data-x0/data-w/data-T so the JS can move it.
"""
function panel(io, x, y, w, hgt, series, T; title="", yticks=nothing, right_ticks=nothing, vlines=Float64[], ylo=nothing, yhi=nothing, cursor_id="", xlabel="")
    L = 48; Rm = right_ticks === nothing ? 12 : 44; Tm = 22; Bm = 20
    px = x + L; pw = w - L - Rm; py = y + Tm; ph = hgt - Tm - Bm
    allv = vcat([filter(isfinite, s[2]) for s in series]...)
    lo = ylo === nothing ? minimum(allv) : ylo; hi = yhi === nothing ? maximum(allv) : yhi
    lo == hi && (lo -= 0.5; hi += 0.5); pad = 0.05*(hi-lo); lo -= pad; hi += pad
    yticks === nothing && (yticks = nice_ticks(lo, hi))
    X(t) = px + pw*t/T; Y(v) = py + ph*(hi - v)/(hi - lo)
    println(io, "<rect x='$px' y='$py' width='$pw' height='$ph' fill='#fafafa' stroke='#999'/>")
    println(io, "<text x='$(x+L)' y='$(y+14)' font-size='12' font-weight='bold'>$(h(title))</text>")
    for (v, lbl) in yticks
        lo <= v <= hi || continue
        println(io, "<line x1='$px' y1='$(f2(Y(v)))' x2='$(px+pw)' y2='$(f2(Y(v)))' stroke='#e4e4e4'/><text x='$(px-4)' y='$(f2(Y(v)+4))' text-anchor='end' font-size='10'>$lbl</text>")
    end
    if right_ticks !== nothing
        for (v, lbl) in right_ticks
            lo <= v <= hi || continue
            println(io, "<text x='$(px+pw+4)' y='$(f2(Y(v)+4))' font-size='10' fill='#7b2cbf'>$lbl</text>")
        end
    end
    for tv in vlines
        println(io, "<line x1='$(f2(X(tv)))' y1='$py' x2='$(f2(X(tv)))' y2='$(py+ph)' stroke='#888' stroke-dasharray='4 3'/>")
    end
    for tv in 0:0.5:T
        println(io, "<text x='$(f2(X(tv)))' y='$(py+ph+12)' text-anchor='middle' font-size='10'>$(tv)</text>")
    end
    isempty(xlabel) || println(io, "<text x='$(px+pw)' y='$(py+ph+12)' text-anchor='end' font-size='10' fill='#555'>$(h(xlabel))</text>")
    lx = px + 6
    for (ts, vals, col, lbl, dash) in series
        pts = join(["$(f2(X(t))),$(f2(clamp(Y(v), py, py+ph)))" for (t, v) in zip(ts, vals) if isfinite(v)], " ")
        println(io, "<polyline points='$pts' fill='none' stroke='$col' stroke-width='1.6'$(isempty(dash) ? "" : " stroke-dasharray='$dash'")/>")
        println(io, "<line x1='$lx' y1='$(py+8)' x2='$(lx+14)' y2='$(py+8)' stroke='$col' stroke-width='2'$(isempty(dash) ? "" : " stroke-dasharray='$dash'")/><text x='$(lx+17)' y='$(py+11)' font-size='10'>$(h(lbl))</text>")
        lx += 24 + 6*length(lbl)
    end
    isempty(cursor_id) || println(io, "<line id='$cursor_id' class='cursor' data-x0='$px' data-w='$pw' data-tend='$T' x1='$px' y1='$py' x2='$px' y2='$(py+ph)' stroke='#d1495b' stroke-width='1.5'/>")
end

# ------------------------------------------------------------------ per-case HTML

"""Top-down (deck-frame) map of where each foot pushes on the board, all three phases.
Long axis s → right (nose), lateral σ → up. One dot per node: hue green (start) → red (end) by time,
brightness ∝ force magnitude (log scale), plus a thin trace of the front foot's path."""
function footprint_svg(d)
    L = 0.80; halfw = 0.1175; tail_s = -0.40; tail_sig = 0.0
    Wp = 640; Hp = 260; sc = 600/1.0; ox = 320; oy = 130
    X(s) = ox + sc*s; Y(sig) = oy - sc*sig
    io = IOBuffer()
    println(io, "<svg xmlns='http://www.w3.org/2000/svg' width='$Wp' height='$Hp' viewBox='0 0 $Wp $Hp' font-family='system-ui,Helvetica,Arial,sans-serif' font-size='11'>")
    println(io, "<rect width='100%' height='100%' fill='white'/>")
    println(io, "<rect x='$(X(-L/2))' y='$(Y(halfw))' width='$(sc*L)' height='$(2sc*halfw)' rx='40' fill='#e9e4d8' stroke='#777'/>")
    for sw in (-0.29, 0.29)   # axles
        println(io, "<line x1='$(X(sw))' y1='$(Y(0.12))' x2='$(X(sw))' y2='$(Y(-0.12))' stroke='#1261a0' stroke-width='6' opacity='0.5'/>")
    end
    println(io, "<line x1='$(X(-L/2))' y1='$(Y(0))' x2='$(X(L/2))' y2='$(Y(0))' stroke='#bbb' stroke-dasharray='3 3'/>")
    println(io, "<text x='$(X(L/2)+4)' y='$(Y(0)+4)'>nose</text><text x='$(X(-L/2)-30)' y='$(Y(0)+4)'>tail</text><text x='$(X(0))' y='$(Y(halfw)-6)' text-anchor='middle'>σ up = toe side (+); s → nose</text>")
    # gather (t, s, σ, F) per phase for the front foot, and (t, F) for the back foot
    ts = vcat(d.t0[1:end-1], d.t1[1:end-1], d.t2[1:end-1]); T0, T1 = ts[1], d.t2[end]
    sfr = vcat(d.s0, d.s1, d.s2); sgf = vcat(zeros(length(d.s0)), zeros(length(d.s1)), d.sg2)
    Ffr = vcat(d.Ff0, d.Ff1, d.Nf2); Fbk = vcat(d.Fb0, d.Fb1, d.Nb2)
    Fmax = max(maximum(Ffr), maximum(Fbk), 1e-9)
    bright(F) = log10(1+F)/log10(1+Fmax)          # 0..1
    col(t, F) = begin
        u = (t-T0)/(T1-T0); r = round(Int, 255*u); g = round(Int, 255*(1-u))
        m = 0.15 + 0.85*bright(F)                  # blend toward white for weak pushes
        "rgb($(round(Int, 255-(255-r)*m)),$(round(Int, 255-(255-g)*m)),$(round(Int, 255*(1-m))))"
    end
    # front-foot path
    pts = join(["$(X(sfr[i])),$(Y(sgf[i]))" for i in eachindex(sfr)], " ")
    println(io, "<polyline points='$pts' fill='none' stroke='#555' stroke-width='1' opacity='0.5'/>")
    for i in eachindex(sfr)
        println(io, "<circle cx='$(X(sfr[i]))' cy='$(Y(sgf[i]))' r='$(3+4*bright(Ffr[i]))' fill='$(col(ts[i], Ffr[i]))' opacity='0.85'/>")
    end
    # back foot: fixed at the tail; stack the time dots vertically beside it so magnitude/time read
    for i in eachindex(Fbk)
        u = (ts[i]-T0)/(T1-T0)
        println(io, "<circle cx='$(X(tail_s)-14)' cy='$(Y(-0.2)+ -120*u)' r='$(2+3*bright(Fbk[i]))' fill='$(col(ts[i], Fbk[i]))' opacity='0.85'/>")
    end
    println(io, "<text x='$(X(tail_s)-14)' y='$(Y(-0.2)+14)' text-anchor='middle' font-size='9'>back foot</text><text x='$(X(tail_s)-14)' y='$(Y(-0.2)+24)' text-anchor='middle' font-size='9'>(time ↑)</text>")
    # phase boundary markers on the front-foot trace
    for (tb, lbl) in ((d.t1[1], "pop"), (d.t2[1], "strike"))
        i = findfirst(>=(tb), ts); i === nothing && continue
        println(io, "<text x='$(X(sfr[i])+5)' y='$(Y(sgf[i])-5)' font-size='9' fill='#333'>$lbl</text>")
    end
    println(io, "<text x='10' y='$(Hp-8)' font-size='10' fill='#555'>colour: green = start → red = end; dot size/brightness ∝ log(1+F), F_max = $(round(Fmax,digits=1)) board weights; axles in blue</text>")
    println(io, "</svg>")
    String(take!(io))
end

function case_html(io, c, tag)
    d = c.d; a = c.a; F = build_frames(d); T = F[end].t
    tbs = [d.t1[1], d.t2[1]]
    # world extents for the two views
    allpts = vcat([vcat([f.rider], [f.tail], [f.nose], f.wheels, [f.front], f.deck, f.tailq) for f in F]...)
    ymin, ymax = extrema(p[2] for p in allpts); zmin, zmax = extrema(p[3] for p in allpts); xmin, xmax = extrema(p[1] for p in allpts)
    ymin -= 0.5; ymax += 0.5; zmin = min(zmin, 0.0) - 0.15; zmax += 0.3
    W = 540; Hh = 420; PWID = 2W + 20
    println(io, "<h2>$(h(c.label))</h2><p>$(h(c.note)) — file <code>$(basename(c.file))</code>, status <b>$(h(a.status))</b>.</p>")
    println(io, "<figure><div class='anim' id='anim-$tag'>")
    println(io, "<svg xmlns='http://www.w3.org/2000/svg' width='$PWID' height='$Hh' viewBox='0 0 $PWID $Hh' font-family='system-ui,Helvetica,Arial,sans-serif'>")
    println(io, "<rect width='100%' height='100%' fill='white'/>")
    for (vi, ttl) in ((0, "side view (Y forward → right, Z up)"), (1, "oblique view (cabinet projection; X lateral recedes up-right)"))
        ox = vi*(W + 20)
        println(io, "<g id='$tag-v$vi' transform='translate($ox,0)'><rect width='$W' height='$Hh' fill='none' stroke='#ddd'/>")
        println(io, "<text x='10' y='18' font-size='13' font-weight='bold'>$(h(ttl))</text>")
        println(io, "<polygon id='$tag-v$vi-ground' points='' fill='#eef3f7' stroke='#444' stroke-width='2'/>")
        println(io, "<g id='$tag-v$vi-triad'></g>")
        println(io, "<polygon id='$tag-v$vi-tailq' points='' stroke='#202020' stroke-width='1'/>")
        println(io, "<polygon id='$tag-v$vi-deck' points='' stroke='#202020' stroke-width='1'/>")
        for k in 1:4 println(io, "<circle id='$tag-v$vi-w$k' r='6' fill='#1261a0' stroke='#0b3d66' stroke-width='1.5'/>") end
        println(io, "<line id='$tag-v$vi-legb' stroke='$COL_BACK' stroke-width='4' stroke-linecap='round' opacity='0.85'/>")
        println(io, "<line id='$tag-v$vi-legf' stroke='$COL_FRONT' stroke-width='4' stroke-linecap='round' opacity='0.85'/>")
        for (nm, col) in (("b", COL_BACK), ("f", COL_FRONT))
            println(io, "<line id='$tag-v$vi-arr$(nm)10' stroke='$col' stroke-width='2' stroke-dasharray='5 4' opacity='0.7'/>")
            println(io, "<polyline id='$tag-v$vi-arr$(nm)s' points='' stroke='$col' stroke-width='3.5' fill='none'/><polygon id='$tag-v$vi-arr$(nm)h' points='' fill='$col'/>")
        end
        println(io, "<circle id='$tag-v$vi-rider' r='8' fill='$COL_RIDER' stroke='#7a1f2b' stroke-width='1.5'/>")
        println(io, "</g>")
    end
    println(io, "<text id='$tag-phase' x='10' y='$(Hh-30)' font-size='14'></text><text id='$tag-time' x='10' y='$(Hh-12)' font-size='13'></text>")
    bx0 = W + 40; bx1 = PWID - 20; by = Hh - 14
    println(io, "<line x1='$bx0' y1='$by' x2='$bx1' y2='$by' stroke='#bbb' stroke-width='4'/>")
    for tb in tbs
        xb = bx0 + (bx1-bx0)*tb/T; println(io, "<line x1='$(f2(xb))' y1='$(by-6)' x2='$(f2(xb))' y2='$(by+6)' stroke='#666' stroke-width='2'/>")
    end
    println(io, "<circle id='$tag-knob' cx='$bx0' cy='$by' r='6' fill='#333'/>")
    ly = 36
    println(io, "<g font-size='11'><line x1='10' y1='$ly' x2='34' y2='$ly' stroke='$COL_BACK' stroke-width='3.5'/><text x='38' y='$(ly+4)'>back (tail) leg + force on board</text>",
        "<line x1='10' y1='$(ly+16)' x2='34' y2='$(ly+16)' stroke='$COL_FRONT' stroke-width='3.5'/><text x='38' y='$(ly+20)'>front leg + force on board (solid: $ARROW_LEN unit = $(round(Int, FORCE_REF)) board weights; dashed: ×10, flight only)</text>",
        "<rect x='10' y='$(ly+26)' width='24' height='8' fill='#2b2b2b'/><text x='38' y='$(ly+34)'>deck top (griptape)</text>",
        "<rect x='10' y='$(ly+40)' width='24' height='8' fill='#d9a066'/><text x='38' y='$(ly+48)'>deck underside (wood); tail region lighter</text></g>")
    println(io, "</svg>")
    println(io, "<div class='ctl'><button id='$tag-play'>&#9646;&#9646; pause</button> ",
        "<input id='$tag-scrub' type='range' min='0' max='10000' value='0' style='width:60%'> ",
        "speed <select id='$tag-speed'>", join(["<option value='$v'$(v == 0.25 ? " selected" : "")>×$v</option>" for v in (0.05, 0.1, 0.25, 0.5, 1.0)]), "</select>",
        " <label><input type='checkbox' id='$tag-tg-arrows' checked> force arrows</label></div>")
    println(io, "</div></figure>")

    # ---- time-series panels (static, cursor driven by JS)
    n0, n1, n2 = length(d.t0), length(d.t1), length(d.t2)
    tc0, tc1, tc2 = d.t0[1:n0-1], d.t1[1:n1-1], d.t2[1:n2-1]
    tall = vcat(d.t0, d.t1, d.t2)
    th_all = vcat(zeros(n0), d.th1, d.th2); ph_all = vcat(zeros(n0+n1), d.ph2); ps_all = vcat(zeros(n0+n1), d.ps2)
    w1_all = vcat(zeros(n0), d.om1, d.w1); w2_all = vcat(zeros(n0+n1), d.w2); w3_all = vcat(zeros(n0+n1), d.w3)
    slog(x) = log10(1 + abs(x))
    tF = vcat(tc0, tc1, tc2); Fb_all = vcat(d.Fb0, d.Fb1, d.Nb2); Ff_all = vcat(d.Ff0, d.Ff1, d.Nf2)
    s_all = vcat(d.s0, d.s1, d.s2); sg_all = vcat(zeros(n0-1 + n1-1), d.sg2)
    zb_all = vcat(d.zb0, d.zb1, d.zb2); zr_all = vcat(d.zr0, d.zr1, d.zr2); xb_all = vcat(d.xb0, d.xb1, d.xb2); xr_all = vcat(d.xr0, d.xr1, d.xr2)
    PW = 540; PH = 190; GAP = 20; rows = 3
    println(io, "<figure><svg xmlns='http://www.w3.org/2000/svg' width='$PWID' height='$(rows*(PH+GAP))' viewBox='0 0 $PWID $(rows*(PH+GAP))' font-family='system-ui,Helvetica,Arial,sans-serif'>")
    println(io, "<rect width='100%' height='100%' fill='white'/>")
    angmax = maximum(abs, vcat(th_all, ph_all, ps_all))
    panel(io, 0, 0, PW, PH, [(tall, th_all, "#1261a0", "pitch θ", ""), (tall, ph_all, "#7b2cbf", "roll φ", ""), (tall, ps_all, "#2a9d3f", "yaw ψ", "")], T;
        title="(a) orientation (rad; right axis: roll in turns)", vlines=tbs, cursor_id="$tag-cur1",
        right_ticks=[(2π*k, "$(k) turn") for k in -2:2], xlabel="t")
    panel(io, PW+20, 0, PW, PH, [(tall, w1_all, "#1261a0", "w1 (pitch rate)", ""), (tall, w2_all, "#7b2cbf", "w2 (roll rate)", ""), (tall, w3_all, "#2a9d3f", "w3 (yaw rate)", "")], T;
        title="(b) body angular velocity (rad / time unit); ground phases: w1 = om1", vlines=tbs, cursor_id="$tag-cur2", xlabel="t")
    fticks = [(slog(v), string(v)) for v in (0, 1, 3, 10, 30, 100)]
    panel(io, 0, PH+GAP, PW, PH, [(tF, slog.(Fb_all), COL_BACK, "back foot Fb / Nb", ""), (tF, slog.(Ff_all), COL_FRONT, "front foot Ff / Nf", "")], T;
        title="(c) leg force on the board (board weights, log10(1+F)); flight: normal push N", yticks=fticks, ylo=0.0, vlines=tbs, cursor_id="$tag-cur3", xlabel="t")
    panel(io, PW+20, PH+GAP, PW, PH, [(tF, s_all, "#1261a0", "s (along deck)", ""), (tF, sg_all, "#7b2cbf", "σ (lateral)", "")], T;
        title="(d) front-foot position on the deck (body frame); deck half-width ±$(HALFW)", vlines=tbs, cursor_id="$tag-cur4", xlabel="t")
    panel(io, 0, 2(PH+GAP), PW, PH, [(tall, zb_all, "#202020", "board Z", ""), (tall, zr_all, COL_RIDER, "rider Z", ""), (tall, xb_all, "#202020", "board X (lateral)", "4 3"), (tall, xr_all, COL_RIDER, "rider X (lateral)", "4 3")], T;
        title="(e) heights Z (solid) and lateral positions X (dashed)", vlines=tbs, cursor_id="$tag-cur5", xlabel="t")
    println(io, "<text x='$(PW+40)' y='$(2(PH+GAP)+40)' font-size='12' fill='#555'>dashed vertical lines: pop start (t = $(f2(d.t1[1]))) and flight start / tail strike (t = $(f2(d.t2[1]))); T = $(f2(T)).</text>")
    println(io, "<text x='$(PW+40)' y='$(2(PH+GAP)+60)' font-size='12' fill='#555'>Controls (forces, s, σ) are piecewise constant per interval and plotted at the interval start.</text>")
    println(io, "</svg></figure>")

    # ---- audit table
    rows_ = [("status", "status"), ("T0 (load)", "T0"), ("T1 (pop)", "T1"), ("T2 (flight)", "T2"), ("θ at tail strike", "theta_hit"),
        ("board apex rise", "board_apex_rise"), ("lowest-point apex", "lowest_point_apex"), ("roll at landing (turns)", "roll_turns"),
        ("roll rate at landing w2(end)", "roll_rate_end"), ("max |roll rate|", "max_roll_rate"), ("yaw range ψ", "yaw_range"),
        ("peak ground force (board weights)", "peak_force_ground"), ("peak normal push in flight (back, front)", "peak_normal_flight"),
        ("σ range", "sigma_range"), ("s range", "s_range"), ("lateral board X range", "lateral_board_range"), ("lateral rider X range", "lateral_rider_range"),
        ("landing v board (x, y, z)", "landing_v_board"), ("landing v rider (x, y, z)", "landing_v_rider"), ("landing COM rel. board (x, y)", "landing_com_rel"),
        ("flight leg work", "flight_leg_work"), ("flight ΔE", "flight_energy_change"), ("flight energy residual (work − ΔE)", "flight_energy_residual"),
        ("semi-implicit gravity bias (explains residual)", "semi_implicit_gravity_bias"), ("angular-momentum residual max", "angular_momentum_residual_max")]
    println(io, "<h3>Foot-force map on the deck (top-down, deck frame)</h3><p class='small'>Where each foot pushes, over the whole trick. A human kickflip sweeps the front foot forward and off the toe-side edge; check whether the optimizer does the same.</p><figure>", footprint_svg(d), "</figure>")
    println(io, "<h3>Audit summary</h3><table><tr><th>quantity</th><th>value</th></tr>")
    for (lbl, key) in rows_
        k = Symbol(key); val = haskey(a, k) ? fmt(a[k]) : "—"
        println(io, "<tr><td>$(h(lbl))</td><td>$(h(val))</td></tr>")
    end
    println(io, "</table>")
    if c.opt !== nothing
        o = c.opt
        println(io, "<p class='small'>Options: rider mass $(o.rider_mass), leg $(o.leg_min)–$(o.leg_max), F ≤ $(o.force_limit_bw) BW, nodes $(o.n_load)/$(o.n_support)/$(o.n_flight), mu = $(o.mu), objective $(o.objective), flip_turns $(o.flip_turns), land_roll_rate_zero $(o.land_roll_rate_zero), lateral_free $(o.lateral_free), yaw_free $(o.yaw_free), tol $(o.tol).</p>")
    end
    # ---- data + JS hookup
    println(io, "<script>setupAnim('$tag', [", join(frame_json.(F), ","), "], {T:$(jnum(T)), view:[{ymin:$(jnum(ymin)),ymax:$(jnum(ymax)),zmin:$(jnum(zmin)),zmax:$(jnum(zmax)),xmin:$(jnum(xmin)),xmax:$(jnum(xmax))}], W:$W, H:$Hh, bx0:$bx0, bx1:$bx1, al:$ARROW_LEN, Fref:$FORCE_REF, ncur:5});</script>")
end

# ------------------------------------------------------------------ shared JS
const JS = raw"""
<script>
function setupAnim(tag, F, cfg){
  const g=id=>document.getElementById(id);
  const PH={0:'phase 0: load (flat board sliding)',1:'phase 1: pop (pivot on the rear axle)',2:'phase 2: flight (3-D rigid body)'};
  const N=F.length, T=cfg.T, W=cfg.W, H=cfg.H;
  const v=cfg.view[0];
  // projections: side view (u=y, v=z); oblique cabinet (u = y + kx·x, v = z + ky·x)
  const kx=0.5*Math.cos(Math.PI/6), ky=0.5*Math.sin(Math.PI/6);
  const proj=[p=>[p[1],p[2]], p=>[p[1]+kx*p[0], p[2]+ky*p[0]]];
  // fit each view: compute screen scale from extents
  const fits=[];
  for(let vi=0;vi<2;vi++){
    const corners=[[v.xmin,v.ymin,v.zmin],[v.xmax,v.ymin,v.zmin],[v.xmin,v.ymax,v.zmin],[v.xmax,v.ymax,v.zmin],[v.xmin,v.ymin,v.zmax],[v.xmax,v.ymin,v.zmax],[v.xmin,v.ymax,v.zmax],[v.xmax,v.ymax,v.zmax]].map(proj[vi]);
    const umin=Math.min(...corners.map(c=>c[0])), umax=Math.max(...corners.map(c=>c[0])), vmin=Math.min(...corners.map(c=>c[1])), vmax=Math.max(...corners.map(c=>c[1]));
    const sc=Math.min((W-20)/(umax-umin),(H-70)/(vmax-vmin));
    const xoff=10+((W-20)-sc*(umax-umin))/2, yoff=60+((H-70)-sc*(vmax-vmin))/2;
    fits.push({sc, X:u=>xoff+sc*(u-umin), Y:w=>yoff+sc*(vmax-w)});
  }
  const S=(vi,p)=>{const q=proj[vi](p); return [fits[vi].X(q[0]), fits[vi].Y(q[1])];};
  const pts=(vi,arr)=>arr.map(p=>S(vi,p).map(c=>c.toFixed(1)).join(',')).join(' ');
  // ground: side view = line at z=0; oblique = parallelogram of the x–y plane
  const gpts=[[[v.xmin,v.ymin,0],[v.xmax,v.ymin,0],[v.xmax,v.ymax,0],[v.xmin,v.ymax,0]], [[v.xmin,v.ymin,0],[v.xmin,v.ymax,0],[v.xmax,v.ymax,0],[v.xmax,v.ymin,0]]];
  g(tag+'-v0-ground').setAttribute('points', pts(0,[[0,v.ymin,0],[0,v.ymax,0],[0,v.ymax,-0.001]]));
  g(tag+'-v1-ground').setAttribute('points', pts(1,gpts[1]));
  // axis triads
  for(let vi=0;vi<2;vi++){
    const o=[v.xmin+0.05, v.ymin+0.05, 0.02], L=0.35, tri=g(tag+'-v'+vi+'-triad'); let s='';
    const ax=[[[L,0,0],'X','#7b2cbf'],[[0,L,0],'Y','#1261a0'],[[0,0,L],'Z','#2a9d3f']];
    for(const [d,lbl,col] of ax){ const a=S(vi,o), b=S(vi,[o[0]+d[0],o[1]+d[1],o[2]+d[2]]);
      s+=`<line x1='${a[0]}' y1='${a[1]}' x2='${b[0]}' y2='${b[1]}' stroke='${col}' stroke-width='2'/><text x='${b[0]+3}' y='${b[1]-3}' font-size='11' fill='${col}'>${lbl}</text>`; }
    tri.innerHTML=s;
  }
  const el={}; const ids=['ground','tailq','deck','w1','w2','w3','w4','legb','legf','arrb10','arrf10','arrbs','arrbh','arrfs','arrfh','rider'];
  for(let vi=0;vi<2;vi++){ el[vi]={}; for(const k of ids) el[vi][k]=g(tag+'-v'+vi+'-'+k); }
  const phase=g(tag+'-phase'), time=g(tag+'-time'), knob=g(tag+'-knob'), play=g(tag+'-play'), scrub=g(tag+'-scrub'), speed=g(tag+'-speed'), tgArrows=g(tag+'-tg-arrows');
  const curs=[]; for(let i=1;i<=cfg.ncur;i++){ const c=g(tag+'-cur'+i); if(c) curs.push(c); }
  let i0=0, t=0, playing=true, last=null;
  const P3=(f,k)=>[f[k]||0,f[k+1]||0,f[k+2]||0];
  function frameAt(t){ if(t<=F[0][0]) return F[0].slice(); if(t>=F[N-1][0]) return F[N-1].slice();
    if(!(F[i0][0]<=t && t<F[i0+1][0])){ i0=0; while(F[i0+1][0]<=t) i0++; }
    const a=F[i0], b=F[i0+1], u=(t-a[0])/(b[0]-a[0]); const o=new Array(a.length);
    for(let k=0;k<a.length;k++){ const av=a[k]??0, bv=b[k]??0; o[k]=av+u*(bv-av); } o[1]=a[1]; return o; }
  function signedArea(q){ let s=0; for(let i=0;i<q.length;i++){ const a=q[i], b=q[(i+1)%q.length]; s+=a[0]*b[1]-b[0]*a[1]; } return s; }
  function quad(vi, elq, P, colTop, colBot){ const q=P.map(p=>S(vi,p)); elq.setAttribute('points', q.map(c=>c[0].toFixed(1)+','+c[1].toFixed(1)).join(' '));
    // corners are CCW seen from the top face (+e3); screen y is down, so CCW-in-world ⇒ negative signed area on screen
    const top = signedArea(q) < 0; elq.setAttribute('fill', top?colTop:colBot); }
  function arrow(vi, foot, Fv, scale, els){ const mag=Math.hypot(Fv[0],Fv[1],Fv[2]);
    if(mag<1e-9||!tgArrows.checked){ els.s.setAttribute('points',''); els.h.setAttribute('points',''); return; }
    const len=scale*cfg.al*mag/cfg.Fref, u=Fv.map(c=>c/mag), tip=[foot[0]+len*u[0],foot[1]+len*u[1],foot[2]+len*u[2]];
    const a=S(vi,foot), b=S(vi,tip); els.s.setAttribute('points',`${a[0]},${a[1]} ${b[0]},${b[1]}`);
    const dx=b[0]-a[0], dy=b[1]-a[1], L=Math.hypot(dx,dy); if(L<1){ els.h.setAttribute('points',''); return; }
    const hs=Math.min(9,0.6*L), ux=dx/L, uy=dy/L, bx=b[0]-hs*ux, by=b[1]-hs*uy, nx=-uy, ny=ux, hw=0.5*hs;
    els.h.setAttribute('points',`${b[0]},${b[1]} ${bx+hw*nx},${by+hw*ny} ${bx-hw*nx},${by-hw*ny}`); }
  function dashed(vi, foot, Fv, scale, e){ const mag=Math.hypot(Fv[0],Fv[1],Fv[2]);
    if(mag<1e-9||!tgArrows.checked){ e.setAttribute('x1',0);e.setAttribute('y1',0);e.setAttribute('x2',0);e.setAttribute('y2',0); return; }
    const len=scale*cfg.al*mag/cfg.Fref, tip=[foot[0]+len*Fv[0]/mag,foot[1]+len*Fv[1]/mag,foot[2]+len*Fv[2]/mag];
    const a=S(vi,foot), b=S(vi,tip); e.setAttribute('x1',a[0]);e.setAttribute('y1',a[1]);e.setAttribute('x2',b[0]);e.setAttribute('y2',b[1]); }
  function draw(t){ const f=frameAt(t); const ph=f[1];
    const rider=P3(f,2), tail=P3(f,5), nose=P3(f,8), wheels=[0,1,2,3].map(k=>P3(f,11+3*k)), front=P3(f,23), deck=[0,1,2,3].map(k=>P3(f,26+3*k)), tailq=[0,1,2,3].map(k=>P3(f,38+3*k)), Fb=P3(f,50), Ff=P3(f,53);
    for(let vi=0;vi<2;vi++){ const e=el[vi];
      quad(vi, e.deck, deck, '#2b2b2b', '#d9a066'); quad(vi, e.tailq, tailq, '#555', '#efc48f');
      for(let k=0;k<4;k++){ const c=S(vi,wheels[k]); e['w'+(k+1)].setAttribute('cx',c[0]); e['w'+(k+1)].setAttribute('cy',c[1]); }
      const line=(L,a,b)=>{const p=S(vi,a),q=S(vi,b); L.setAttribute('x1',p[0]);L.setAttribute('y1',p[1]);L.setAttribute('x2',q[0]);L.setAttribute('y2',q[1]);};
      line(e.legb, tail, rider); line(e.legf, front, rider);
      const r=S(vi,rider); e.rider.setAttribute('cx',r[0]); e.rider.setAttribute('cy',r[1]);
      arrow(vi, tail, Fb, 1, {s:e.arrbs,h:e.arrbh}); arrow(vi, front, Ff, 1, {s:e.arrfs,h:e.arrfh});
      if(ph==2){ dashed(vi, tail, Fb, 10, e.arrb10); dashed(vi, front, Ff, 10, e.arrf10); } else { dashed(vi, tail, [0,0,0], 1, e.arrb10); dashed(vi, front, [0,0,0], 1, e.arrf10); }
    }
    phase.textContent=PH[ph]||('phase '+ph);
    time.textContent='t = '+t.toFixed(2)+' / '+T.toFixed(2)+'  (× 0.303 s)';
    knob.setAttribute('cx', cfg.bx0+(cfg.bx1-cfg.bx0)*t/T);
    for(const c of curs){ const x0=+c.dataset.x0, w=+c.dataset.w, Tc=+c.dataset.tend; const x=x0+w*t/Tc; c.setAttribute('x1',x); c.setAttribute('x2',x); }
    if(document.activeElement!==scrub) scrub.value=Math.round(10000*t/T); }
  function step(ts){ if(last!==null && playing){ t+= (ts-last)/1000*parseFloat(speed.value)/0.303*1.0; if(t>T) t=0; draw(t); } last=ts; requestAnimationFrame(step); }
  play.onclick=()=>{ playing=!playing; play.innerHTML=playing?'&#9646;&#9646; pause':'&#9654; play'; };
  scrub.oninput=()=>{ playing=false; play.innerHTML='&#9654; play'; t=T*scrub.value/10000; draw(t); };
  tgArrows.onchange=()=>draw(t);
  draw(0); requestAnimationFrame(step);
}
</script>
"""

# ------------------------------------------------------------------ assemble
io = IOBuffer()
println(io, "<!DOCTYPE html><html><head><meta charset='utf-8'><title>3-D ollie and kickflip: rider + board</title>")
println(io, "<style>body{font-family:system-ui,Helvetica,Arial,sans-serif;max-width:1100px;margin:20px auto;padding:0 16px;line-height:1.4}.ctl{margin:6px 0;font-size:13px}.ctl button{font-size:13px}table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #bbb;padding:3px 8px;font-size:13px;text-align:left}th{background:#eee}figure{margin:12px 0;overflow-x:auto}svg{max-width:100%;height:auto}.small{font-size:12px;color:#555}code{background:#f3f3f3;padding:0 3px}</style>")
println(io, JS, "</head><body>")
println(io, "<h1>3-D ollie and kickflip: rider + board (<code>ollie_rider_3D.jl</code>)</h1>")
println(io, """
<p><b>Coordinate convention</b> (world, right-handed, Z up): X lateral (perpendicular to the plane of the 2-D model), Y forward (direction of travel), Z vertical, gravity −Z.
Board body frame at the board COM on the deck mid-plane: e1 lateral across the deck (parallel to the axles), e2 along the deck (nose positive), e3 deck normal.
Orientation R = R<sub>z</sub>(ψ)·R<sub>x</sub>(θ)·R<sub>y</sub>(φ), world = R·body: ψ yaw about world Z (kept small; free in flight), θ pitch about the axle axis (nose-up positive, the ollie rotation),
φ roll about the board long axis (the kickflip rotation; +2π = one flip). Body angular velocity ω = (w1, w2, w3) about (e1, e2, e3); Euler's equations with I1 = pitch inertia $(B.I_pitch), I2 = roll inertia $(B.I_roll), I3 = yaw inertia $(B.I_yaw).
Front foot on the deck at body point (σ, s, 0): s along the deck, σ lateral — the lateral offset is what produces roll torque, τ<sub>roll</sub> = σ·N for a normal push N. Back foot at the tail tip (0, −0.4, 0.06).
Body points drawn: tail (0, −0.4, 0.06), nose (0, 0.4, 0), wheels (±0.12, ±0.29, −0.08), deck half-width $(HALFW), deck length $(B.deck_length).</p>
<p><b>Modelling assumptions.</b> Nondimensional units as in <code>ollie_rider.jl</code>: board mass 1, g 1, length ℓ<sub>max</sub> = 0.9 m (time unit 0.303 s). Rider is a point mass on two push-only telescoping legs.
Phases: 0 load (flat board sliding on ice, planar + lateral slide), 1 pop (pivot on the rear axle; roll and yaw locked by the two rear wheels), tail-strike impact (vertical impulse at the tail centre, e = $(B.restitution)), 2 flight (full 3-D rigid body).
In phases 0/1 roll = yaw = 0 and the front foot is on the centreline (σ = 0). In flight with mu = 0 the feet can only push normal to the deck (N ≥ 0), so roll torque comes exclusively from a normal push applied at a lateral offset σ.</p>
<p><b>Staging.</b> Stage 1 = the 3-D model run as an ollie with the lateral DOFs locked; it must match the 2-D ollie (lowest-point apex 1.804). Stage 2 = the pop (phases 0, 1 and the tail strike) frozen to stage 1 and only the flight optimized, with the roll target 2π and mu = 0 — only normal pushes at lateral offset σ produce roll torque. Stage 3 = everything free (load, pop and flight optimized together; section present only if <code>kickflip3d_stage3.jls</code> exists).</p>
<p><b>Reading the animations.</b> Two orthographic views: side view (Y → right, Z up) and an oblique cabinet projection with an axis triad. The deck is drawn as two quads (main deck and tail region) filled dark when the griptape top face is toward the viewer and tan when the underside shows, so a 2π roll reads as dark → tan → dark. Blue circles are the four wheels, the red dot the rider COM, orange/green lines the back and front legs, and the arrows the force each foot applies to the board (solid arrows: $ARROW_LEN length unit = $(round(Int, FORCE_REF)) board weights; dashed arrows in flight: the same force ×10, since flight pushes are only ~1–20 board weights). The five panels below each animation carry a red time cursor synced to the animation.</p>
<!-- FINDINGS -->

<h2>Appendix: taming the spastic board contact — candidate force penalties</h2>
<p>The flight forces are impulse-like (they sharpen under mesh refinement), because the only smoothing is a
tie-breaker penalty <code>force_rate_weight · Σ(ΔF)²</code> with weight 1e-4, and that sum is not even
mesh-normalized (Σ(ΔF)² halves when h halves, so refinement <i>weakens</i> the penalty). Candidate fixes, in
increasing order of physics content — the ReBULA walker (<code>infinite_rebula_simpler.jl</code>) already
implements the third:</p>
<table>
<tr><th>penalty / constraint</th><th>mechanism</th><th>pros</th><th>cons</th></tr>
<tr><td><b>1. Mesh-normalized rate penalty</b> w·Σ(ΔF)²/h ≈ w·∫Ḟ²dt</td><td>tune w up from 1e-4</td>
<td>one-line change; sweep below shows it works</td><td>w is arbitrary; trades apex for smoothness with no physical anchor</td></tr>
<tr><td><b>2. Hard rate limit</b> |Ḟ| ≤ r_max</td><td>like the foot-slide limit; human force-rate ≈ 10 BW / 0.1 s</td>
<td>physical units, no objective distortion</td><td>bang-bang solutions ride the bound; needs continuation like the slide limit</td></tr>
<tr><td><b>3. ReBULA-style: force as a C¹ state, penalize ∫F̈²</b> — F, Ḟ become states, F̈ the control (split F̈ = F̈⁺−F̈⁻)</td>
<td>exactly the walker's <code>c_fr2</code> machinery; forces come out smooth by construction</td>
<td>proven in this repo; smoothness independent of mesh; c_fr2 already calibrated in the walker (work ≈ fr2 at 5e-5-ish scales)</td>
<td>2 extra state arrays per leg per phase; boundary conditions for Ḟ at phase joins and the impact need care</td></tr>
<tr><td><b>4. Activation dynamics</b> τ·Ḟ = u − F (first-order lag, τ ≈ 0.3 nondim ≈ 90 ms)</td>
<td>muscle-like: the control u is bounded, F must lag it</td><td>physiological; kills impulses harder than 3</td><td>adds stiffness; τ is another parameter</td></tr>
<tr><td><b>5. Hill force–velocity limit</b> (Appendix 2 of the ollie report, already implemented in 2-D)</td>
<td>F ≤ F_Hill(dℓ/dt)</td><td>the most physical: fast pushes are weak pushes; also fixes the absolute heights</td>
<td>collapses apex to realistic ≈0.2 m — changes the trick, not just its smoothness; port to 3-D pending</td></tr>
</table>
<p><b>Pilot sweep (option 1)</b>, stage-2 kickflip (pop frozen), <code>force_rate_weight</code> w on the existing
un-normalized Σ(ΔF)²: </p>
<table><tr><th>w</th><th>status</th><th>apex</th><th>peak N (back, front)</th><th>∫Ṅ² (both feet)</th></tr>
<tr><td>1e-4 (default)</td><td>solved</td><td>2.513</td><td>1.34, 21.3</td><td>7436</td></tr>
<tr><td>1e-2</td><td>solved</td><td>2.191</td><td>0.64, 20.6</td><td>4083</td></tr>
<tr><td>1e-1</td><td>solved</td><td>2.195</td><td>0.68, 20.5</td><td>4041</td></tr>
<tr><td>1</td><td>solved</td><td>2.195</td><td>0.71, 20.5</td><td>4041</td></tr>
<tr><td>10</td><td>solved</td><td>2.195</td><td>0.70, 20.5</td><td>4041</td></tr></table>
<p><b>Pilot verdict: the rate penalty saturates.</b> Three orders of magnitude of extra weight change neither the
front-foot spike (≈20.5 BW) nor ∫Ṅ² (≈4040): the spike is <i>constraint-driven</i>, not preference-driven — with
the pop frozen, the impact hands the flight a state that requires an immediate hard catch push (and costs ≈0.3 of
apex to smooth what can be smoothed). This is the same lesson as the 2026-07-05 walker log ("the force-rate weight
is not the culprit"): when a spike survives a 1000× weight increase, look for the constraint that demands it.
So option 1 alone cannot deliver a realistic contact; the spike must be removed structurally — free the pop
timing together with the flight, bound the rate (option 2), make force a C¹ state with F(strike⁺) continuity
(option 3), or cap what a leg can do at speed (option 5).
<p><b>Recommendation:</b> option 3 for the report solves — with F and Ḟ continuous across the strike so the catch cannot be an impulse — (it is the house style — the walker's force-rate states —
and makes mesh refinement honest), option 5 when the goal shifts from "does a kickflip exist" to "does a
<i>human</i> kickflip exist". Options 1–2 are good quick knobs for the Pluto playground.</p>

<h2>Findings</h2>
<p class='small'>Current settings: per-leg force cap 3 BW (running-peak scale; was 2 BW), front-foot slide speed ≤ 1.5 (≈4.5 m/s), landing |ψ| ≤ 0.1, μ = 0 (normal pushes only).</p>
<ul>
<li><b>3-D model reproduces the 2-D ollie.</b> With lateral motion locked (stage 1) the 3-D solve matches <code>ollie_rider.jl</code> (at 2 BW: 1.802 vs 1.804; at 3 BW: 2.52 vs 2.46–2.55 across the 2-D solve variants); the board angular-momentum balance closes to machine precision. The flight energy residual is the semi-implicit Euler gravity bias −½(m+M)g²h² per step (reported as <code>semi_implicit_gravity_bias</code>), the same scheme as the 2-D model — not a physics error.</li>
<li><b>A kickflip is feasible with normal foot pushes only (μ = 0), and it is almost free.</b> Stage 2 (pop frozen to the ollie, flight optimized): roll exactly 2π, roll rate zero at touchdown, square landing, lowest-point apex 2.19. Stage 3 (everything free): apex <b>2.47 vs 2.52 for the plain ollie (−2 %)</b>, identical pop. The roll torque is entirely σ·N from the front foot's lateral offset; with I_roll = 0.008 the angular impulse needed is ≈0.03, i.e. a few board-weight·time units of push at the deck edge. The result was the same at 2 BW (1.765 vs 1.802).</li>
<li><b>Force cap matters a lot for height, not for the flip.</b> Raising the leg cap from 2 to 3 BW lifts the ollie from 1.80 to 2.52 (≈2.3 m) — the model has no force–velocity limit, so height simply scales with the cap (see Appendix 2 of the ollie report: with a Hill limit the ollie collapses to ≈0.2 m). The kickflip's cost relative to the ollie stays ≈2 % either way.</li>
<li><b>Foot-force map.</b> Without a slide limit the front foot teleported around the deck between pushes (its position is unconstrained while the force is zero). With |ds/dt|, |dσ/dt| ≤ 1.5 the load/pop phase shows the human-like <b>forward sweep</b>: the front foot slides from mid-deck to the nose and arrives there at the tail strike. In flight the flick is a continuous lateral traverse — nose → heel-side edge (catch side) → toe-side edge — rather than a forward flick off the toe edge; that is what a normal-only (μ = 0) foot can do. A toe-drag flick needs μ &gt; 0 and would be the next experiment.</li>
<li><b>Timing.</b> Stage 3 delays the flick until the board is nearly level, then completes 2π in ≈1 time unit (≈0.3 s) at a peak roll rate of 4.1 rad/unit (≈14 rad/s, ≈2.2 rev/s), inside the range of real kickflips.</li>
<li><b>Yaw coupling is real and had to be constrained.</b> Rolling a pitched board yaws it through the (I₂−I₁)w₁w₂ Euler term; unconstrained, the all-free solve landed 34° off-axis. |ψ(end)| ≤ 0.1 fixes it for ≈0.01 of apex; yaw still reaches 0.5 rad mid-flight and is steered back by the catch.</li>
<li><b>Mesh.</b> Refining the flight from 61 to 121 nodes keeps the stage-2 solution (apex 2.19 → 2.17). The post-strike front-foot push sharpens with the mesh (21 → 41 BW peak): it is impulse-like; a force-rate weight or the force–velocity limit would regularize it.</li>
<li><b>What was needed to converge.</b> (i) Roll-target continuation (0.1 … 0.75 turns; the intermediate targets were locally infeasible but served as a warm-start path); (ii) flight-only with the pop frozen first, then everything free with a full warm start; (iii) locking the flat lateral DOFs of the ground phases (Ipopt's step computation failed otherwise); (iv) continuation in the foot-slide limit (solve at 3.0, tighten to 1.5) — the tight limit from the analytic guess gives NUMERICAL_ERROR; (v) a cold barrier restart (μ₀ = 0.1, adaptive) as a fallback.</li>
<li><b>Assumptions to revisit.</b> Roll/yaw locked during the pop without checking the wheel-reaction split; tail strike as one vertical impulse at the tail centre; μ = 0 (no toe drag); the feet are attached to body points during the flip; the flight-time bound (4.0) is close to active in the 3 BW ollie (T2 = 3.9).</li>
</ul>
""")
for (i, c) in enumerate(cases)
    case_html(io, c, "C$i")
end
println(io, "<p class='small'>Generated by <code>generate_kickflip_3d_report.jl</code> from the serialized stage files; no solves were run.</p></body></html>")
write(OUT, take!(io))
println("wrote $(OUT) ($(round(filesize(OUT)/1024, digits=1)) kB)")
