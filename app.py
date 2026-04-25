"""
SLOSS Simulator — Streamlit GUI Iiteration 1: end-to-end scaffold

Implements minimal slider change (number of reserves)-> cached simulation -> landscape plot.

Established the architecture we could build on:
  - run_counter in session_state forces fresh runs on the Run button
    while letting slider drags reuse cached (seed=42) results
  - simulate_cached() is keyed on parameters + run_counter, so identical
    inputs return

Future iterations could implement:
- timestep scrubber
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

landscape, pop_history, history = simulate_cached(
    num_reserves=num_reserves,
    run_counter=st.session_state.run_counter,
    use_seed=use_seed,
)



############ Visualization ########################

with viz_col:
    st.subheader("Final population")

    final_pop = pop_history[-1]
    K = 50  # carrying capacity, matches simulation default

    fig = go.Figure(
        data=go.Heatmap(
            z=final_pop,
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
        f"Total population at final timestep: **{final_pop.sum():.0f}**  •  "
        f"Reserves still occupied: **{history['num_occupied_reserves'][-1]}**"
    )
