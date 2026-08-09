"""Run the validated inclined/eccentric lunar J2 demonstration."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from lunar_astrodynamics import (GRGM1200A_J2, ClassicalElements, PropagationSettings,
    analytical_j2_secular_rates, element_history, linear_rate, orbital_period_s,
    propagate, state_from_elements)
DEG=np.pi/180.0; DAY=86400.0

def initial_elements():
    return ClassicalElements(GRGM1200A_J2.collision_radius_m+150_000.0,0.02,45*DEG,30*DEG,40*DEG,0.0)

def run(orbits=40):
    e=initial_elements(); y0=state_from_elements(e,GRGM1200A_J2.mu_m3_s2)
    period=orbital_period_s(e.semi_major_axis_m,GRGM1200A_J2.mu_m3_s2); duration=orbits*period
    times=np.linspace(0,duration,orbits*80+1); sol=propagate(y0,duration,sample_times_s=times)
    if not sol.success or sol.t[-1] < duration*(1-1e-12): raise RuntimeError(sol.message)
    hist=element_history(sol.t,sol.y,GRGM1200A_J2.mu_m3_s2)
    nr=linear_rate(hist.time_s,hist.raan_rad_unwrapped); na=linear_rate(hist.time_s,hist.argument_of_periapsis_rad_unwrapped)
    ar,aa=analytical_j2_secular_rates(e,GRGM1200A_J2.mu_m3_s2,GRGM1200A_J2.reference_radius_m,GRGM1200A_J2.j2)
    hz=np.cross(sol.y[:3].T,sol.y[3:].T)[:,2]
    tight=propagate(y0,5*period,sample_times_s=[0,5*period],settings=PropagationSettings(rtol=1e-13,position_atol_m=1e-9,velocity_atol_m_s=1e-12))
    default=propagate(y0,5*period,sample_times_s=[0,5*period])
    return {"model":GRGM1200A_J2.name,"orbits":orbits,"duration_days":duration/DAY,
      "initial_semi_major_axis_km":e.semi_major_axis_m/1000,"initial_eccentricity":e.eccentricity,"initial_inclination_deg":e.inclination_rad/DEG,
      "analytical_raan_deg_day":ar/DEG*DAY,"numerical_raan_deg_day":nr/DEG*DAY,"raan_relative_error":abs((nr-ar)/ar),
      "analytical_argp_deg_day":aa/DEG*DAY,"numerical_argp_deg_day":na/DEG*DAY,"argp_relative_error":abs((na-aa)/aa),
      "relative_hz_span":float((hz.max()-hz.min())/abs(hz[0])),
      "five_orbit_tight_reference_position_difference_m":float(np.linalg.norm(default.y[:3,-1]-tight.y[:3,-1])),
      "surface_impact":bool(sol.t_events and sol.t_events[0].size)}

def svg(metrics):
    vals=[abs(metrics["analytical_raan_deg_day"]),abs(metrics["numerical_raan_deg_day"]),metrics["analytical_argp_deg_day"],metrics["numerical_argp_deg_day"]]
    labels=["RAAN theory","RAAN numerical","Periapsis theory","Periapsis numerical"]
    m=max(vals); rows=[]
    for i,(label,v) in enumerate(zip(labels,vals)):
        y=70+i*58; w=520*v/m
        rows.append(f'<text x="20" y="{y}" font-size="14">{label}</text><rect x="170" y="{y-16}" width="{w:.1f}" height="20" fill="#555"/><text x="{180+w:.1f}" y="{y}" font-size="13">{v:.6f} deg/day</text>')
    return '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="330" viewBox="0 0 900 330"><rect width="100%" height="100%" fill="white"/><text x="20" y="30" font-size="20" font-weight="600">Lunar J2 secular-rate validation</text>'+''.join(rows)+'</svg>\n'

def write_outputs(metrics_path=Path("results/j2_validation.json"),figure_path=Path("assets/results/j2_precession.svg"),orbits=40):
    metrics=run(orbits); metrics_path.parent.mkdir(parents=True,exist_ok=True); figure_path.parent.mkdir(parents=True,exist_ok=True)
    metrics_path.write_text(json.dumps(metrics,indent=2,sort_keys=True)+"\n"); figure_path.write_text(svg(metrics)); return metrics

def main():
    p=argparse.ArgumentParser(); p.add_argument("--orbits",type=int,default=40); p.add_argument("--metrics-path",type=Path,default=Path("results/j2_validation.json")); p.add_argument("--figure-path",type=Path,default=Path("assets/results/j2_precession.svg")); a=p.parse_args(); print(json.dumps(write_outputs(a.metrics_path,a.figure_path,a.orbits),indent=2,sort_keys=True))
if __name__=="__main__": main()
