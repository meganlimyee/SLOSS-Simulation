import numpy as np
from scipy import ndimage
from scipy.signal import fftconvolve

"""
Notes on the program:
Ecology simulation exploring the classic SLOSS ( "Single Large Or Several Small" reserves) debate.
If total protected habitat area is the same, is it better to protect one large area or several smaller areas?
Build a landscape, place habiat reserves, and simulate polulations living in those reserves over time based on
population growth, emigration, and disturbances.

Simulation core used by the Streamlit GUI (app.py)
Two changes from the original sloss.py:
  1. run_simulation accepts an optional `seed` argument so the GUI can produce
     reproducible runs while a slider is being dragged (clean parameter
     exploration), then switch to fresh randomness when the Run button is hit
  2. run_simulation now records a snapshot of the full population grid at
     every timestep (pop_history), so the GUI can scrub backward/forward
     through the simulation without re-running it
"""


"""
Create a landscape, marking reserve habitat cells as True and non-reserve cells as
False. 
If multiple habitat areas, he landscape includes a few well-separated habitat patches of roughly equal size.

Params:
total_area (int): total area of reserve habitat
num_reserves (int) : distinct number of reserves

"""


def create_landscape(L=50, total_area=100, num_reserves=1):
    # create empty np array for landscape and reserves
    landscape = np.zeros((L, L), dtype=bool)

    # calculate approximately how large each reserve should be
    reserveArea = total_area / num_reserves
    reserveSide = int(np.sqrt(reserveArea))

    numReservesPlaced = 0
    attempts = 0

    while numReservesPlaced < num_reserves and attempts < 1000:

        # randomly place top left corner of each reserve in the landscape
        top = np.random.randint(0, L - reserveSide)
        left = np.random.randint(0, L - reserveSide)

        # place reserve if there isn't already another reserve there or adjacent
        # use 0 or L if we are at the edge so we don't get indexing errors
        # add extra cells to bottom
        checktop = max(0, top - 1)
        checkbottom = min(L, top + reserveSide + 1)
        checkleft = max(0, left - 1)
        checkright = min(L, left + reserveSide + 1)

        if not landscape[checktop:checkbottom, checkleft:checkright].any():
            landscape[top:top + reserveSide, left:left + reserveSide] = True
            numReservesPlaced += 1
        attempts += 1

    # extra space that was not able to fit in the squares
    extraSpace = total_area - num_reserves * reserveSide**2
    addedExtra = 0
    attempts = 0
    while addedExtra < extraSpace and attempts < 1000:
        reserve_coords = np.argwhere(landscape)  # current reserve cells

        # set to find edges
        edgecoords = []

        for x, y in reserve_coords:
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            nextneighbors = [(x - 2, y), (x + 2, y), (x, y - 2), (x, y + 2)]

            for nx, ny in neighbors:
                # Check if in bounds and  empty
                if 0 <= nx < L and 0 <= ny < L and not landscape[nx, ny]:

                    # check that next neighbors are also empty
                    # this prevents connection of two adjacent reserves
                    for nnx, nny in nextneighbors:
                        if 0 <= nnx < L and 0 <= nny < L:
                            # if in bounds, check if currently reserve
                            if not landscape[nnx, nny]:
                                edgecoords.append((nx, ny))
                        else:  # not in bounds, don't worry about it
                            edgecoords.append((nx, ny))

        # no more possible edges, can't add any more habitat to existing reserves
        if len(edgecoords) == 0:
            break

        # Pick random edge cell and make it a reserve
        randEdgex, randEdgey = edgecoords[np.random.randint(len(edgecoords))]
        landscape[randEdgex, randEdgey] = True
        addedExtra += 1
        attempts += 1

    return landscape


"""
Run population dynamics.

Reserve geometry matters because small-scattered reserves lose mroe dispersers to non-habitat, 
but also might receive rescue immigrants from neighbors.


r: polutation growth rate
K: carrying population capacity per reserve cell
m: migration rate (# individuals leaving each cell per timestep)
disturbance_rate: % chance of disturbance per timestep 
disturbance_severity: % indivuduals removed in affected cells
disturbance_extent: % of reserve cells affected
traveldist: distance that individuls disperse
seed: optional int, seeds numpy's RNG so the run is reproducible. The GUI
      uses a fixed seed while the user is dragging sliders (so changes show
      clean parameter effects, not stochastic noise) and None when the user
      clicks Run (to draw a fresh random realization)

Returns:
pop_history (list of (L, L) ndarrays): snapshot of the population grid after
    each timestep. pop_history[-1] is the final state. The GUI uses this to
    scrub backward/forward through time without re-running
history (dict): aggregate per-timestep stats (total_pop, occupancy,
    num_occupied_reserves), same as the original

"""


def run_simulation(landscape, timesteps=100, r=0.5, K=50, m=0.05,
                   disturbance_rate=0.01, disturbance_severity=0.5,
                   disturbance_extent=0.1, traveldist=10, seed=None):

    # seed the RNG for reproducible runs (used by the GUI's slider-drag mode)
    if seed is not None:
        np.random.seed(seed)

    L = landscape.shape[0]
    pop = np.zeros((L, L))  # create a population array matching landscape
    pop[landscape] = K * 0.2  # start at a fraction of carrying capacity

    # find reserves using ndimage
    # ndimage identifies connected reserve patches to track how many separate reserves are occupied (>10)
    reserveArrays, numReserves = ndimage.label(landscape)

    history = {
        'total_pop': [],  # total population
        'occupancy': [],  # fraction of reserve cells with pop > 1
        'num_occupied_reserves': []
    }

    # per-timestep snapshots of the full population grid, for GUI scrubbing
    pop_history = []

    # kernel for immigration, with decreasing likelihood (Gaussian distrib) the further out we go
    # scale the kernel side with traveldist so larger dispersal distances aren't truncated
    kernelSize = int(6 * traveldist)
    # ensure odd kernel side length so our original cell is at the center
    if kernelSize % 2 == 0:
        kernelSize += 1
    center = kernelSize // 2  # find the center

    # Gaussian kernel
    y, x = np.ogrid[-center:center + 1, -center:center + 1]
    dispersal_kernel = np.exp(-(x**2 + y**2) / (2 * traveldist**2))
    # normalize so sum of probabilities equals 1
    dispersal_kernel = dispersal_kernel / dispersal_kernel.sum()

    # repeat process of growth, immigration, disturbance for timesteps
    for t in range(timesteps):

        # 1. Logistic growth of  ulation in each cell (standard-density growth equation)
        # population grows towareeds carrying capacity
        pop[landscape] = pop[landscape] + r * \
            pop[landscape] * (1 - pop[landscape] / K)
        pop = np.clip(pop, 0, None)  # disallow negative populations

        # 2. Dispersal
        # a fraction of each reserve's population emigrates, getting redistributed across grid via convolution
        # with Gaussian kernel.
        emigrants = pop * m  # how many individuals are leaving from each cell
        # each cell's base population decreases as some leave
        pop = pop * (1 - m)

        # immigrants move into nearby cells according to Gaussian kernel
        dispersed = fftconvolve(emigrants, dispersal_kernel, mode='same')

        pop += dispersed  # add immigrants to their new home cells
        pop[~landscape] = 0  # species can only survive in reserve

        # 3. Disturbance
        # with some chance (per timestep), random fraction of reserve cells are impacted.
        # this models disturbances such as fires, stormes, diseases, etc.

        # randomly check if there will be a disturbance based on our rate (0-1)
        if np.random.rand() < disturbance_rate:
            # check for reserve cells (marked True)
            reserve_coords = np.argwhere(landscape)

            # disturbance applies to a certain percent of the reserve cells
            numDisturbed = int(disturbance_extent * len(reserve_coords))
            if numDisturbed > 0:
                disturbed_idx = np.random.choice(
                    len(reserve_coords), numDisturbed, replace=False)
                disturbed = reserve_coords[disturbed_idx]

                #pop[disturbed] *= (1 - disturbance_severity)
                pop[disturbed[:, 0], disturbed[:, 1]
                    ] *= (1 - disturbance_severity)

        # record this timestep's full grid (.copy() so future timesteps don't overwrite it)
        pop_history.append(pop.copy())

        history['total_pop'].append(float(pop.sum()))
        history['occupancy'].append(
            float((pop[landscape] > 1).sum() / landscape.sum()))

        occupiedReserve = 0
        for i in range(1, numReserves + 1):
            if pop[reserveArrays == i].sum() > 10:
                occupiedReserve += 1
        history['num_occupied_reserves'].append(occupiedReserve)

    return pop_history, history
