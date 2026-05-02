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

from sloss import create_landscape, run_simulation



############ Cached simulation ############

# when the user clicks Run, we increment run_counter, which busts cache and produces
# a fresh stochastic run. When the user just drags a slider, run_counter is unchanged
# so identical parameters return the cached result instantly. Slider drags use seed=42
# (clean exploration); Run button uses no seed (true stochasticity).


@st.cache_data(show_spinner=False)
def simulate_cached(num_reserves: int, r: float, K: float,
                    m: float, traveldist: float,
                    disturbance_rate: float, disturbance_extent: float,
                    disturbance_severity: float, edge_effect: float,
                    patchiness: float,
                    run_counter: int, use_seed: bool):
    #Run a full simulation cached on its arguments
    seed = 42 if use_seed else None
    # create_landscape also uses np.random so seeding here makes the whole pipeline reproducible
    if seed is not None:
        np.random.seed(seed)
    landscape = create_landscape(num_reserves=num_reserves, patchiness=patchiness)
    pop_history, history = run_simulation(
        landscape,
        r=r, K=K, m=m,
        traveldist=traveldist,
        disturbance_rate=disturbance_rate,
        disturbance_extent=disturbance_extent,
        disturbance_severity=disturbance_severity,
        edge_effect=edge_effect,
        seed=seed,
    )
    return landscape, pop_history, history


############# Page setup ########################

st.set_page_config(page_title="SLOSS Simulator", layout="wide")
st.title("SLOSS Simulator")
st.caption("Explore the tradeoff between single large or several small reserves.")

# Initialize session state
if "run_counter" not in st.session_state:
    st.session_state.run_counter = 0
    st.session_state.fresh_run = False  # True only on the run after Run is clicked
    st.session_state.last_params = None  # for detecting parameter changes



############ Layout: vizual on left, controls on right ########################

#preset scenarios that show specific effects we want to showcase
presets = {
"default": {
    "L": 50, "num_reserves": 1, "num_reserves2": 16,
    "r": 0.5, "K": 50, "m": 0.05, "traveldist": 5.0, "disturbance_rate": 0.01,
    "disturbance_severity": 0.5, "disturbance_extent": 5.0, "edge_effect": 1.0, "patchiness": 0.0,
},
"rescue_effect": {
    "L": 50, "num_reserves": 1, "num_reserves2": 16,
    "r": 2.0, "K": 50, "m": 0.05, "traveldist": 5.0, "disturbance_rate": 0.3,
    "disturbance_severity": 1.0, "disturbance_extent": 11.0, "edge_effect": 1.0, "patchiness": 0.0,
},
"edge_effect": {
    "L": 50, "num_reserves": 1, "num_reserves2": 16,
    "r": 0.2, "K": 50, "m": 0.25, "traveldist": 5.0, "disturbance_rate": 0.01,
    "disturbance_severity": 0.5, "disturbance_extent": 5.0, "edge_effect": 0.3, "patchiness": 0.0,
}
}

st.subheader("Controls")
ctrl_col, ctrl_col2, ctrl_col3 = st.columns(3)

with ctrl_col:
    with st.expander("**Reserve Configuration**", expanded=True):
        num_reserves = st.slider(
            "Number of reserves (left)",
            min_value=1, max_value=20, value=1, step=1,
            help="Determine how the total habitat is divided (total area is held constant).",
        )

        num_reserves2 = st.slider(
            "Number of reserves (right)",
            min_value=1, max_value=20, value=10, step=1,
            help="Determine how the total habitat is divided (total area is held constant).",
        )

        patchiness = st.slider(
            "Patchiness",
            min_value=0.0, max_value=0.5, value=0.0, step=0.05,
            help="Fraction of interior cells redistributed to the perimeter.",
        )

    with st.expander("**Preset Scenarios**", expanded=True):
        #create a version counter which will reset sliders every time a preset is used
        if 'preset_version' not in st.session_state:
            st.session_state.preset_version = 0
        if st.button("Default", width='stretch'):
            st.session_state.preset = "default"
            st.session_state.preset_version += 1
            st.rerun()
        st.text("High disturbance, high growth rate:")
        if st.button("Rescue Effect (Favors SS)", width='stretch'):
            st.session_state.preset = "rescue_effect"
            st.session_state.preset_version += 1
            st.rerun()
        st.text("High migration, low growth rate:")
        if st.button("High Roaming (Favors SL)", width='stretch'):
            st.session_state.preset = "edge_effect"
            st.session_state.preset_version += 1
            st.rerun()
        

with ctrl_col2:
    if 'preset' not in st.session_state:
        st.session_state.preset = "default"

    current_preset = presets[st.session_state.preset]
    #updates keys when presets pressed, resetting slider
    v = st.session_state.preset_version

    with st.expander("**Logistic Growth Parameters**", expanded=True):
        r = st.slider(
            "Growth Rate (r)",
            min_value=0.05, max_value=3.0, value=current_preset["r"], step=0.05,
            help="Intrinsic per-capita growth rate (r) in the logistic equation.", key=f"slider_r{v}"
        )

        K = st.slider(
            "Carrying capacity per cell (K)",
            min_value=5, max_value=100, value=current_preset["K"], step=1,
            help="Maximum population a single reserve cell can sustain (K).", key=f"slider_K{v}"
        )

        edge_effect = st.slider(
            "Edge effect",
            min_value=0.1, max_value=3.0, value=current_preset["edge_effect"], step=0.1,
            help="Multiplier on carrying capacity for cells on the edge of a reserve. 1.0 is neutral; <1 means edges support smaller populations than the interior (interior-loving species like deep-forest birds); >1 means edges support larger populations (edge-adapted species like deer or many songbirds). Several Small reserves are mostly edge cells, so this slider strongly affects them.",
            key=f"slider_edge_effect{v}"
        )

    with st.expander("**Migration Parameters**", expanded=True):
        m = st.slider(
            "Percent of individuals migrating per cell per timestep",
            min_value=0.0, max_value=0.25, value=current_preset['m'], step=0.01,
            help="Higher migration moves more individuals out of source cells. Whether that helps or hurts depends on dispersal distance and reserve geometry.", key=f"slider_m{v}"
        )

        traveldist = st.slider(
            "Dispersal distance (σ)",
            min_value=1.0, max_value=20.0, value=current_preset['traveldist'], step=0.5,
            help="Standard deviation σ of the Gaussian dispersal kernel, in cells.", key=f"slider_traveldist{v}"
        )

with ctrl_col3:
    with st.expander("**Disturbance Parameters**", expanded=True):
        disturbance_rate = st.slider(
            "Disturbance rate",
            min_value=0.0, max_value=1.0, value=current_preset['disturbance_rate'], step=0.05,
            help="Probability that a disturbance event occurs per timestep.", key=f"slider_distrate{v}"
        )

        disturbance_extent = st.slider(
            "Disturbance radius",
            min_value=0.0, max_value=20.0, value=current_preset['disturbance_extent'], step=1.0,
            help="Radius of the disturbance in cells. A disturbance hits a circular area centered on a random reserve cell.", key=f"slider_distextent{v}"
        )

        disturbance_severity = st.slider(
            "Disturbance severity",
            min_value=0.0, max_value=1.0, value=current_preset['disturbance_severity'], step=0.05,
            help="Fraction of the population removed in disturbed cells.", key=f"slider_distseverity{v}"
        )

    st.markdown("---")

    if st.button("Run (Fresh Randomness)", width='stretch'):
        st.session_state.run_counter += 1
        st.session_state.fresh_run = True
        # Force the timestep slider to snap to the final frame on a fresh run
        st.session_state.timestep = None
        st.rerun()

    st.caption(
        "**Hint:** Drag sliders to explore parameter effects with stochasticity held "
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
                  disturbance_extent, disturbance_severity, edge_effect, patchiness,
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
    edge_effect=edge_effect,
    patchiness=patchiness,
    run_counter=st.session_state.run_counter,
    use_seed=use_seed,
)

landscape2, pop_history2, history2 = simulate_cached(
    num_reserves=num_reserves2, r=r, K=K, m=m,
    traveldist=traveldist,
    disturbance_rate=disturbance_rate,
    disturbance_extent=disturbance_extent,
    disturbance_severity=disturbance_severity,
    edge_effect=edge_effect,
    patchiness=patchiness,
    run_counter=st.session_state.run_counter,
    use_seed=use_seed,
)


T = len(pop_history)


############ Visualization ########################

# how many frames a disturbance border lingers on the landscape after the disturbance
disturb_persistence = 5


def overlay_disturbances(fig, disturbance_events, ts_current):
    """
    Draw a red circle on figures for each disturbance that fired, sustained by
    the number of timeframe set above via disturb_persistence.

    disturbance_events: list of tuples from history['disturbance_events']
                    (t, center_y, center_x, radius)
    ts_current: the timestep currently shown on scrubber
    """
    for (event_t, cy, cx, radius) in disturbance_events:
        sustain = ts_current - event_t
        if 0 <= sustain < disturb_persistence:
            fig.add_shape(
                type="circle",
                xref="x", yref="y",
                x0=cx - radius, y0=cy - radius,
                x1=cx + radius, y1=cy + radius,
                line=dict(color="red", width=2),
                fillcolor="rgba(0,0,0,0)", 
            )


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
st.subheader("Population")

viz_col, viz_col2 = st.columns(2)

with viz_col:

    pop_at_t = pop_history[timestep]

    fig = go.Figure(
        data=go.Heatmap(
            z=pop_at_t,
            zmin=0, zmax=max(K, K * edge_effect),
            colorscale="Viridis",
            colorbar=dict(title="Population"),
        )
    )
    
    landscape_float = landscape.astype(float)

    fig.add_trace(go.Contour(
        z=landscape_float,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            showlabels=False,
            coloring='none' # just want outline around reserve locations
        ),
        line=dict(
            color='black',
            width=2
        ),
        showscale=False,
        hoverinfo='skip',
        name="Reserves"
    ))

    overlay_disturbances(fig, history['disturbance_events'], timestep)


    bound = pop_at_t.shape[0]
    fig.update_layout(
        width=600, height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-0.5, bound - 0.5], constrain="domain"),
        yaxis=dict(scaleanchor="x", range=[bound - 0.5, -0.5], constrain="domain"),  # reversed for imshow
    )
    st.plotly_chart(fig, width='stretch', key='colormap')

    st.caption(
        f"Timestep **{timestep}** of {T - 1}  •  "
        f"Total population: **{pop_at_t.sum():.0f}**  •  "
        f"Reserves occupied: **{history['num_occupied_reserves'][timestep]}**"
    )
    
    st.subheader("Statistics Over Time")
   
    # Total Population
    fig_pop = go.Figure() 
    fig_pop.add_trace(go.Scatter(
        y=history['total_pop'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Total Population'
    ))
    # add line which shows which timestep we are at on the plot
    fig_pop.add_vline(x=timestep, line_dash="dash", line_color="red", opacity=0.7)
    fig_pop.update_layout(
        title="Total Population over Time",
        xaxis_title="Timestep",
        yaxis_title="Population",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_pop, width='stretch', key='pop')
    
    # Occupancy
    fig_occ = go.Figure()
    fig_occ.add_trace(go.Scatter(
        y=history['occupancy'],
        mode='lines',
        line=dict(color='green', width=2),
        name='Occupancy'
    ))
    fig_occ.add_vline(x=timestep, line_dash="dash", line_color="red", opacity=0.7)
    fig_occ.update_layout(
        title="Reserve Cell Occupancy over Time (more than 1 individual)",
        xaxis_title="Timestep",
        yaxis_title="Occupancy (%)",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_occ, width='stretch', key='occ')
    
    # Number of Occupied Reserves
    fig_reserves = go.Figure()
    fig_reserves.add_trace(go.Scatter(
        y=history['num_occupied_reserves'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='Occupied Reserves'
    ))
    fig_reserves.add_vline(x=timestep, line_dash="dash", line_color="red", opacity=0.7)
    fig_reserves.update_layout(
        title="Number of Reserves Occupied over Time (more than 10 individuals)",
        xaxis_title="Timestep",
        yaxis_title="Number of Reserves",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_reserves, width='stretch', key='reserves')
    
with viz_col2:

    st.session_state.timestep = timestep

    pop_at_t2 = pop_history2[timestep]

    fig2 = go.Figure(
        data=go.Heatmap(
            z=pop_at_t2,
            zmin=0, zmax=max(K, K * edge_effect),
            colorscale="Viridis",
            colorbar=dict(title="Population"),
        )
    )
    
    landscape_float2 = landscape2.astype(float)

    fig2.add_trace(go.Contour(
        z=landscape_float2,
        contours=dict(
            start=0.5,
            end=0.5,
            size=1,
            showlabels=False,
            coloring='none' # just want outline around reserve locations
        ),
        line=dict(
            color='black',
            width=2
        ),
        showscale=False,
        hoverinfo='skip',
        name="Reserves"
    ))

    overlay_disturbances(fig2, history2['disturbance_events'], timestep)

    L = pop_at_t2.shape[0]
    fig2.update_layout(
        width=600, height=600,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(range=[-0.5, L - 0.5], constrain="domain"),
        yaxis=dict(scaleanchor="x", range=[L - 0.5, -0.5], constrain="domain"),
    )
    st.plotly_chart(fig2, width='stretch', key='colormap2')

    st.caption(
        f"Timestep **{timestep}** of {T - 1}  •  "
        f"Total population: **{pop_at_t2.sum():.0f}**  •  "
        f"Reserves occupied: **{history2['num_occupied_reserves'][timestep]}**"
    )
    
    st.subheader("Statistics Over Time")
   
    # Total Population
    fig_pop2 = go.Figure() 
    fig_pop2.add_trace(go.Scatter(
        y=history2['total_pop'],
        mode='lines',
        line=dict(color='blue', width=2),
        name='Total Population'
    ))
    # add line which shows which timestep we are at on the plot
    fig_pop2.add_vline(x=timestep, line_dash="dash", line_color="red", opacity=0.7)
    fig_pop2.update_layout(
        title="Total Population over Time",
        xaxis_title="Timestep",
        yaxis_title="Population",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_pop2, width='stretch', key='pop2')
    
    # Occupancy
    fig_occ2 = go.Figure()
    fig_occ2.add_trace(go.Scatter(
        y=history2['occupancy'],
        mode='lines',
        line=dict(color='green', width=2),
        name='Occupancy'
    ))
    fig_occ2.add_vline(x=timestep, line_dash="dash", line_color="red", opacity=0.7)
    fig_occ2.update_layout(
        title="Reserve Cell Occupancy over Time (more than 1 individual)",
        xaxis_title="Timestep",
        yaxis_title="Occupancy (%)",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_occ2, width='stretch', key='occ2')
    
    # Number of Occupied Reserves
    fig_reserves2 = go.Figure()
    fig_reserves2.add_trace(go.Scatter(
        y=history2['num_occupied_reserves'],
        mode='lines',
        line=dict(color='orange', width=2),
        name='Occupied Reserves'
    ))
    fig_reserves2.add_vline(x=timestep, line_dash="dash", line_color="red", opacity=0.7)
    fig_reserves2.update_layout(
        title="Number of Reserves Occupied over Time (more than 10 individuals)",
        xaxis_title="Timestep",
        yaxis_title="Number of Reserves",
        height=250,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig_reserves2, width='stretch', key='reserves2')
