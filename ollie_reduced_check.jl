module OllieReducedCheck

using LinearAlgebra

export ReducedBoardParams, body_point, rotated_point, reduced_configuration,
       reduced_velocity, point_generalized_force, reduced_acceleration,
       expand_preimpact_state, tail_impact_reset, check_reduced_model

"Parameters for the independent reduced-coordinate fingerboard check."
Base.@kwdef struct ReducedBoardParams
    mass::Float64 = 1.0
    inertia::Float64 = 0.085
    gravity::Float64 = 1.0
    deck_length::Float64 = 0.80
    wheelbase::Float64 = 0.58
    deck_height::Float64 = 0.08
    tail_rise::Float64 = 0.06
    restitution::Float64 = 0.18
end

function body_point(p::ReducedBoardParams, point::Symbol; s=0.0)
    point === :tail && return [-p.deck_length/2, p.tail_rise]
    point === :rear_slide && return [-p.wheelbase/2, -p.deck_height]
    point === :front_slide && return [p.wheelbase/2, -p.deck_height]
    point === :deck && return [s, 0.0]
    throw(ArgumentError("unknown board point $point"))
end

"World-oriented vector R(theta)*rho, still relative to the board COM."
function rotated_point(theta, rho)
    c, s = cos(theta), sin(theta)
    return [c*rho[1]-s*rho[2], s*rho[1]+c*rho[2]]
end

"Full q=[x,y,theta] implied by reduced qr=[x,theta] and an active slide."
function reduced_configuration(qr, p::ReducedBoardParams; contact=:rear_slide)
    rc = rotated_point(qr[2], body_point(p,contact))
    return [qr[1], -rc[2], qr[2]]
end

"Full v=[xdot,ydot,omega] implied by vr=[xdot,omega]."
function reduced_velocity(qr, vr, p::ReducedBoardParams; contact=:rear_slide)
    rc = rotated_point(qr[2], body_point(p,contact))
    return [vr[1], -rc[1]*vr[2], vr[2]]
end

"Generalized force in qr=[x,theta] from a world force at a body point."
function point_generalized_force(theta, force, rho, rho_contact)
    r = rotated_point(theta,rho)
    rc = rotated_point(theta,rho_contact)
    # delta p = [delta x - r_y delta theta,
    #            (r_x-r_cx) delta theta]
    return [force[1], -r[2]*force[1] + (r[1]-rc[1])*force[2]]
end

"""
Reduced equations while one frictionless underside point slides on y=0.

`loads` is an iterable of `(force, body_point)` pairs. Returns
`[xddot,thetaddot]`. The slide reaction is eliminated by admissible virtual work.
"""
function reduced_acceleration(qr, vr, loads, p::ReducedBoardParams;
                              contact=:rear_slide)
    theta, omega = qr[2], vr[2]
    rho_c = body_point(p,contact)
    rc = rotated_point(theta,rho_c)
    Q = zeros(promote_type(typeof(theta),typeof(omega),Float64),2)
    for (force,rho) in loads
        Q .+= point_generalized_force(theta,force,rho,rho_c)
    end
    xddot = Q[1]/p.mass
    effective_inertia = p.inertia + p.mass*rc[1]^2
    # M(theta)*alpha - m*Xc*Yc*omega^2 = Qtheta + m*g*Xc
    alphaddot = (Q[2] + p.mass*p.gravity*rc[1] +
                 p.mass*rc[1]*rc[2]*omega^2)/effective_inertia
    return [xddot,alphaddot]
end

expand_preimpact_state(qr,vr,p::ReducedBoardParams;contact=:rear_slide) =
    (reduced_configuration(qr,p;contact), reduced_velocity(qr,vr,p;contact))

"Frictionless Newton tail impact after the old slide constraint is released."
function tail_impact_reset(qminus, vminus, p::ReducedBoardParams;
                           restitution=p.restitution)
    rt = rotated_point(qminus[3],body_point(p,:tail))
    J = [0.0, 1.0, rt[1]]
    Minv = Diagonal([1/p.mass,1/p.mass,1/p.inertia])
    vnminus = dot(J,vminus)
    impulse = max(zero(vnminus),-(1+restitution)*vnminus/dot(J,Minv*J))
    vplus = vminus + Minv*J*impulse
    return (; qplus=copy(qminus),vplus,impulse,J,vnminus,
            vnplus=dot(J,vplus))
end

"Numerical identities for the reduced constraint, virtual work, and impact."
function check_reduced_model(; p=ReducedBoardParams(), atol=1e-11)
    qr=[0.12,0.35]; vr=[0.4,0.8]
    q,v=expand_preimpact_state(qr,vr,p)
    rc=rotated_point(qr[2],body_point(p,:rear_slide))
    gap=q[2]+rc[2]
    gap_velocity=v[2]+rc[1]*v[3]

    rho=body_point(p,:tail); f=[0.7,-1.3]; dq=[0.02,-0.03]
    Q=point_generalized_force(qr[2],f,rho,body_point(p,:rear_slide))
    eps=1e-7
    p0=q[1:2]+rotated_point(qr[2],rho)
    qre=qr+eps*dq
    qe=reduced_configuration(qre,p)
    pe=qe[1:2]+rotated_point(qre[2],rho)
    virtual_work_fd=dot(f,(pe-p0)/eps)
    virtual_work_q=dot(Q,dq)

    # Choose a closing preimpact velocity solely to test the reset identities.
    vtest=copy(v); vtest[2]-=1.0
    impact=tail_impact_reset(q,vtest,p)
    M=Diagonal([p.mass,p.mass,p.inertia])
    momentum_residual=norm(M*(impact.vplus-vtest)-impact.J*impact.impulse)
    restitution_residual=abs(impact.vnplus+p.restitution*impact.vnminus)
    passed=abs(gap)<atol && abs(gap_velocity)<atol &&
           abs(virtual_work_fd-virtual_work_q)<1e-6 &&
           momentum_residual<atol && restitution_residual<atol
    return (;passed,gap,gap_velocity,virtual_work_fd,virtual_work_q,
            momentum_residual,restitution_residual,impact)
end

end # module

