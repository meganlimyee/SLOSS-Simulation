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
- primary/advanced parameter panels
- presets to simulate realistic conditions (to learn important aspects of SLOSS)

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
def simulate_cached(num_reserves: int, run_counter: int, use_seed: bool):
    #Run a full simulation cached on its arguments
    seed = 42 if use_seed else None
    # create_landscape also uses np.random so seeding here makes the whole pipeline reproducible
    if seed is not None:
        np.random.seed(seed)
    landscape = create_landscape(num_reserves=num_reserves)
    pop_history, history = run_simulation(landscape, seed=seed)
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
        min_value=1, max_value=25, value=1, step=1,
        help="Total habitat area is held constant. Toggle to change its distribution.",
    )

    if st.button("Run (fresh randomness)", use_container_width=True):
        st.session_state.run_counter += 1
        st.session_state.fresh_run = True
        # Force the timestep slider to snap to the final frame on a fresh run
        st.session_state.timestep = None
        st.rerun()

    st.markdown("---")
    st.caption(
        "Drag the slider to explore parameter effects with stochasticity held "
        "fixed. Click **Run** to draw a fresh random "
        "realization with the current parameters."
    )

# Decide which mode we're in for this render
use_seed = not st.session_state.fresh_run
# Reset the flag so the *next* slider change reverts to seeded mode
st.session_state.fresh_run = False

# detect parameter changes; when the user changes a parameter, we want the
# scrubber to snap to the final timestep of the new run rather than stay
# wherever it was in the previous run's history
current_params = (num_reserves, st.session_state.run_counter)
if st.session_state.last_params != current_params:
    st.session_state.timestep = None  # signal to default to final
    st.session_state.last_params = current_params

landscape, pop_history, history = simulate_cached(
    num_reserves=num_reserves,
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
    K = 50  # carrying capacity, matches simulation default

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
