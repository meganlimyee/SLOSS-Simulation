"""
SLOSS Simulator — Streamlit GUI Iteration 2: timestep scrubber


Iiteration 1: end-to-end scaffold

Implements minimal slider change (number of reserves)-> cached simulation -> landscape plot.
Established the architecture we could build on:
  - run_counter in session_state forces fresh runs on the Run button
    while letting slider drags reuse cached (seed=42) results
  - simulate_cached() is keyed on parameters + run_counter, so identical
    inputs return

Iteration 2: timestep scrubber
 - timestep scrubber slider lets  user replay the simulation history
 - landscape shows pop_history[t] instead of always the final state
 - scrubber defaults to the final timestep when parameters change or Run
    is clicked

Future iterations could implement:
- time-series plots with synchronized playhead
- advanced parameter panels
- presets to simulate realistic conditions (to learn important aspects of SLOSS)
    - currently it is free parameter exploration, it may be hard to grasp SLOSS effects

"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sim_streamlit import create_landscape, run_simulation



############ Cached simulation ############

# when the user clicks Run, we increment run_counter, which busts cache and produces
# a fresh stochastic run. When the user just drags a slider, run_counter is unchanged
# so identical parameters return the cached result instantly. Slider drags use seed=42
# (clean exploration); Run button uses no seed (true stochasticity).


@st.cache_data(show_spinner=False)
def simulate_cached(num_reserves: int, r: float, K: float,
                    m: float, traveldist: float,
                    disturbance_rate: float, disturbance_extent: float,
                    disturbance_severity: float,
                    run_counter: int, use_seed: bool):
    #Run a full simulation cached on its arguments
    seed = 42 if use_seed else None
    # create_landscape also uses np.random so seeding here makes the whole pipeline reproducible
    if seed is not None:
        np.random.seed(seed)
    landscape = create_landscape(num_reserves=num_reserves)
    pop_history, history = run_simulation(
        landscape,
        r=r, K=K, m=m,
        traveldist=traveldist,
        disturbance_rate=disturbance_rate,
        disturbance_extent=disturbance_extent,
        disturbance_severity=disturbance_severity,
        seed=seed,
    )
    return landscape, pop_history, history


############# Page setup ########################

st.set_page_config(page_title="SLOSS Simulator", layout="wide")
st.title("SLOSS Simulator")
st.caption("Single Large Or Several Small reserves — explore the tradeoff.")

# Initialize session state
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0
    st.session_state.fresh_run = False  # True only on the run after Run is clicked
    st.session_state.last_params = None  # for detecting parameter changes



############ Layout: vizual on left, controls on right ########################

viz_col, ctrl_col = st.columns([2, 1])

with ctrl_col:
    st.subheader("Controls")

    num_reserves = st.slider(
        "Number of reserves",
        min_value=1, max_value=20, value=1, step=1,
        help="TODO: DESCRIPTION",
    )
    
    st.markdown("**Logistic Growth Parameters**")

    r = st.slider(
        "Growth Rate",
        min_value=0.05, max_value=3.0, value=0.05, step=0.05,
        help="TODO: DESCRIPTION",
    )

    K = st.slider(
        "Carrying Capacity per Cell",
        min_value=5, max_value=100, value=10, step=1,
        help="TODO: DESCRIPTION",
    )
    
    st.markdown("**Migration Parameters**")

    m = st.slider(
        "Percent of individuals migrating per cell per timestep",
        min_value=0.0, max_value=1.0, value=0.05, step=0.05,
        help="TODO: DESCRIPTION",
    )
    
    traveldist = st.slider(
        "Dispersal distance σ",
        min_value=1.0, max_value=20.0, value=10.0, step=0.5,
        help="TODO: DESCRIPTION",
    )

    st.markdown("**Disturbance regime**")

    disturbance_rate = st.slider(
        "Disturbance rate",
        min_value=0.0, max_value=1.0, value=0.01, step=0.05,
        help="TODO: DESCRIPTION",
    )

    disturbance_extent = st.slider(
        "Disturbance extent",
        min_value=0.0, max_value=1.0, value=0.1, step=0.05,
        help="TODO: DESCRIPTION",
    )

    disturbance_severity = st.slider(
        "Disturbance severity",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="TODO: DESCRIPTION",
    )

    st.markdown("---")

    if st.button("Run (fresh randomness)", use_container_width=True):
        st.session_state.run_counter += 1
        st.session_state.fresh_run = True
        # Force the timestep slider to snap to the final frame on a fresh run
        st.session_state.timestep = None
        st.rerun()

    st.caption(
        "Drag sliders to explore parameter effects with stochasticity held "
        "fixed (same random seed). Click **Run** to draw a fresh random "
        "realization with the current parameters."
    )

# Decide which mode we're in for this render
use_seed = not st.session_state.fresh_run
# Reset the flag so the *next* slider change reverts to seeded mode
st.session_state.fresh_run = False

# detect parameter changes; when the user changes a parameter, we want the
# scrubber to snap to the final timestep of the new run rather than stay
# wherever it was in the previous run's history
current_params = (num_reserves, r, K, m, traveldist, disturbance_rate,
                  disturbance_extent, disturbance_severity,
                  st.session_state.run_counter)
if st.session_state.last_params != current_params:
    st.session_state.timestep = None  # signal to default to final
    st.session_state.last_params = current_params

landscape, pop_history, history = simulate_cached(
    num_reserves=num_reserves, r=r, K=K, m=m,
    traveldist=traveldist,
    disturbance_rate=disturbance_rate,
    disturbance_extent=disturbance_extent,
    disturbance_severity=disturbance_severity,
    run_counter=st.session_state.run_counter,
    use_seed=use_seed,
)

T = len(pop_history)


############ Visualization ########################

with viz_col:
    st.subheader("Population")

    # Timestep scrubber defaults to the final timestep on first run whenever parameters change
    default_t = T - 1 if st.session_state.timestep is None else st.session_state.timestep
    timestep = st.slider(
        "Timestep",
        min_value=0, max_value=T - 1,
        value=default_t,
        key="timestep_slider",
        help="Drag to replay the simulation. Defaults to the final timestep.",
    )
    st.session_state.timestep = timestep

    pop_at_t = pop_history[timestep]

    fig = go.Figure(
        data=go.Heatmap(
            z=pop_at_t,
            zmin=0, zmax=K,
            colorscale="Viridis",
            colorbar=dict(title="Population"),
        )
    )
    fig.update_layout(
        width=600, height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        yaxis=dict(scaleanchor="x", autorange="reversed"),  # match imshow
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"Timestep **{timestep}** of {T - 1}  •  "
        f"Total population: **{pop_at_t.sum():.0f}**  •  "
        f"Reserves occupied: **{history['num_occupied_reserves'][timestep]}**"
    )
