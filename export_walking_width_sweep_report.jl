# Step-width sweep of the 3D model with c_fr2 calibrated so work ≈ fr2 at width 0.10.
# Run: julia --project=. export_walking_width_sweep_report.jl
ENV["GKSwstype"]="nul"
include(joinpath(@__DIR__,"infinite_3d_rebula_simpler.jl"))
const DEST=joinpath(@__DIR__,"simulation_reports"); mkpath(DEST)
const C_FR2=5e-5           # calibrated: width 0.10 → work 0.417, fr2 0.428
widths=[0.0,0.05,0.10,0.15,0.20,0.25,0.30]
rows=[]; figs=String[]
for w in widths
    m=point_mass_walker_3d(step_width=w,c_fr2=C_FR2); o=object_dictionary(m)
    st=termination_status(m); ok=st in (MOI.LOCALLY_SOLVED,MOI.ALMOST_LOCALLY_SOLVED)
    work=ok ? value(o[:cost_work]) : NaN; fr2=ok ? value(o[:cost_fr2]) : NaN
    # lateral vs sagittal decomposition of the mechanical work (|F·v| by component)
    t=value(o[:τ])*value(o[:t_f]); dt=diff(t)
    Fx=value(o[:Ftot_x]); Fy=value(o[:Ftot_y]); Fz=value(o[:Ftot_z])
    vx=value(o[:vx]); vy=value(o[:vy]); vz=value(o[:vz])
    absint(p)=sum((abs.(p[1:end-1]).+abs.(p[2:end]))./2 .*dt)
    Wx=absint(Fx.*vx); Wy=absint(Fy.*vy); Wz=absint(Fz.*vz)
    vx0=vx[1]; pxamp=maximum(abs.(value(o[:px])))
    e=ok ? check_energy_balance(m).balance_error : NaN
    push!(rows,(;w,st,work,fr2,obj=work+fr2,Wx,Wy,Wz,vx0,pxamp,e))
    println("w=$w $st work=$(round(work,digits=4)) fr2=$(round(fr2,digits=4)) |Fx vx|=$(round(Wx,digits=4)) |Fy vy|=$(round(Wy,digits=4)) |Fz vz|=$(round(Wz,digits=4)) vx0=$(round(vx0,digits=3))")
    tag="walking_width_"*lpad(round(Int,w*100),3,'0')
    f=plot_results_3d(m); plot!(f,plot_title="step width $w  c_fr2=$C_FR2",plot_titlefontsize=10)
    savefig(f,joinpath(DEST,tag*"_plot_results.svg")); push!(figs,tag*"_plot_results.svg")
end
ws=[r.w for r in rows]
fs=plot(ws,[r.work for r in rows],marker=:circle,label="work",xlabel="step width",ylabel="cost",title="Width sweep, c_fr2=$C_FR2",size=(800,600),layout=(2,1),subplot=1)
plot!(fs,ws,[r.fr2 for r in rows],marker=:circle,label="fr2",subplot=1)
plot!(fs,ws,[r.obj for r in rows],marker=:circle,label="objective",subplot=1)
plot!(fs,ws,[r.Wx for r in rows],marker=:circle,label="∫|Fx·vx|",subplot=2,xlabel="step width",ylabel="abs work by axis")
plot!(fs,ws,[r.Wy for r in rows],marker=:circle,label="∫|Fy·vy|",subplot=2)
plot!(fs,ws,[r.Wz for r in rows],marker=:circle,label="∫|Fz·vz|",subplot=2)
savefig(fs,joinpath(DEST,"walking_width_sweep_summary.svg"))
r3(x)=round(x,digits=4)
tbl=join(["<tr><td>$(r.w)</td><td>$(r.st)</td><td>$(r3(r.work))</td><td>$(r3(r.fr2))</td><td>$(r3(r.obj))</td><td>$(r3(r.Wx))</td><td>$(r3(r.Wy))</td><td>$(r3(r.Wz))</td><td>$(r3(r.vx0))</td><td>$(r3(r.pxamp))</td><td>$(r3(r.e))</td></tr>" for r in rows],"\n")
figsec=join(["<h3>Step width $(rows[i].w)</h3><figure><object data='$(figs[i])' type='image/svg+xml'></object></figure>" for i in eachindex(figs)],"\n")
html="""<!doctype html><meta charset='utf-8'><title>ReBULA 3D step-width sweep</title>
<style>body{font-family:system-ui;max-width:1000px;margin:2em auto;line-height:1.45}figure{border:1px solid #ccc;padding:.6em}object{width:100%;height:820px}code{background:#eee;padding:.1em .25em}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.2em .5em;text-align:right}</style>
<h1>ReBULA 3D step-width sweep</h1>
<p><code>point_mass_walker_3d(step_width=w, c_fr2=$C_FR2)</code>, all other kwargs default (step_speed 0.4, z0 0.98, vy endpoints pinned). <code>c_fr2</code> was calibrated at width 0.10 so that work and fr2 are the same size (0.417 vs 0.428); the same scaling is used at every width. Note that fr2 scales exactly linearly in c_fr2 (∫F̈² ≈ 7400 for any weight 1e-5…3e-2), i.e. the force-rate term is not shaping the solution — the force profile is essentially fixed by the boundary conditions.</p>
<p>Work uses the full 3-D leg-length rate <code>(leg·v)/|leg|</code>, so lateral (x) work is in the objective. The columns ∫|F·v| by axis decompose the total absolute power by direction to show where the work actually goes.</p>
<table><tr><th>width</th><th>status</th><th>work</th><th>fr2</th><th>objective</th><th>∫|Fx vx|</th><th>∫|Fy vy|</th><th>∫|Fz vz|</th><th>vx(0)</th><th>max|px|</th><th>energy err</th></tr>$tbl</table>
<figure><object data='walking_width_sweep_summary.svg' type='image/svg+xml' style='height:620px'></object></figure>
<h2>Per-width plot_results_3d</h2>$figsec</body>"""
write(joinpath(DEST,"walking_width_sweep_report.html"),html)
println("wrote ",joinpath(DEST,"walking_width_sweep_report.html"))
