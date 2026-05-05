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


def create_landscape(L=50, total_area=200, num_reserves=1, patchiness=0.0):
    """
    Create a landscape, marking reserve habitat cells as True and non-reserve 
    cells False. If multiple habitat areas, the landscape includes a few 
    well-separated habitat patches of roughly equal size.

    Parameters
    ----------
    L : int, optional
        Side length of total square landscape area. The default is 50.
    total_area : int, optional
        Total number of habitat cells. The default is 200.
    num_reserves : int, optional
        Number of habitat reserves. The default is 1.
    patchiness : float, optional
        Fraction of interior cells redistributed to edges for heterogeneity 
        and roughness. Reasonable range from 0 to 0.5. The default is 0.0.

    Returns
    -------
    landscape : (L, L) ndarray
        Numpy array of booleans where True is inside a reserve and False is
        outside a reserve.

    """
    # error handling
    if not isinstance(L, int) or L <= 0:
        raise ValueError(f"L must be a positive integer, got {L}")
    if not isinstance(num_reserves, int) or num_reserves < 1:
        raise ValueError(
            f"num_reserves must be a positive integer, got {num_reserves}")
    if total_area <= 0 or total_area > L * L:
        raise ValueError(
            f"total_area must be in (0, L*L={L * L}], got {total_area}")
    if not 0.0 <= patchiness <= 1.0:
        raise ValueError(
            f"patchiness must be between 0 and 1, got {patchiness}")

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

    # remove and redistribute cells to patch perimeter to incorporate patchiness.
    # introduces edge-to-area ratio toggle (reserve shape) in addition to reserve count
    if patchiness > 0:
        # find reserve cells whose 4 neighbors are reserve cells (interior cells)
        interior_coords = []
        reserve_coords = np.argwhere(landscape)
        for x, y in reserve_coords:
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
            allInside = True
            for nx, ny in neighbors:
                if not (0 <= nx < L and 0 <= ny < L) or not landscape[nx, ny]:
                    allInside = False
                    break
            if allInside:
                interior_coords.append((x, y))

        # convert to a set for fast 2x2 validity checks
        interior_set = set(interior_coords)

        # find candidate 2x2 clumps which are valid if all four of its cells are fully interior
        candidate_clumps = []
        for (x, y) in interior_coords:
            if (x, y + 1) in interior_set and (x + 1, y) in interior_set and (x + 1, y + 1) in interior_set:
                candidate_clumps.append((x, y))

        # how many cells to remove from interior and re-add at the perimeter
        totalCells = int(landscape.sum())
        numHoles = int(patchiness * totalCells)
        numClumps = numHoles // 4  # each 2x2 clump removes 4 cells

        # remove non-overlapping clumps
        cellsRemoved = 0
        if numClumps > 0 and len(candidate_clumps) > 0:
            np.random.shuffle(candidate_clumps)
            removed_set = set()
            for (cx, cy) in candidate_clumps:
                if cellsRemoved >= numClumps * 4:
                    break
                clump_cells = [(cx, cy), (cx, cy + 1),
                               (cx + 1, cy), (cx + 1, cy + 1)]
                if any(c in removed_set for c in clump_cells):
                    continue  # overlaps with an already-removed clump
                for (rx, ry) in clump_cells:
                    landscape[rx, ry] = False
                    removed_set.add((rx, ry))
                cellsRemoved += 4

        # add the same number of cells back to the perimeter
        if cellsRemoved > 0:
            cellsAdded = 0
            attempts = 0
            while cellsAdded < cellsRemoved and attempts < 5000:
                reserve_coords = np.argwhere(landscape)
                edgecoords = []
                for x, y in reserve_coords:
                    neighbors = [(x - 1, y), (x + 1, y),
                                 (x, y - 1), (x, y + 1)]
                    nextneighbors = [(x - 2, y), (x + 2, y),
                                     (x, y - 2), (x, y + 2)]
                    for nx, ny in neighbors:
                        if 0 <= nx < L and 0 <= ny < L and not landscape[nx, ny]:
                            for nnx, nny in nextneighbors:
                                if 0 <= nnx < L and 0 <= nny < L:
                                    if not landscape[nnx, nny]:
                                        edgecoords.append((nx, ny))
                                else:
                                    edgecoords.append((nx, ny))

                if len(edgecoords) == 0:
                    break

                randEdgex, randEdgey = edgecoords[np.random.randint(
                    len(edgecoords))]
                landscape[randEdgex, randEdgey] = True
                cellsAdded += 1
                attempts += 1

    return landscape


def run_simulation(landscape, timesteps=100, r=0.5, K=50, m=0.05,
                   disturbance_rate=0.01, disturbance_severity=0.5,
                   disturbance_extent=0.1, traveldist=10,
                   edge_effect=1.0, seed=None):
    """
    Simulates population in the landscape over time. 

    Parameters
    ----------
    landscape : (L, L) ndarray
        Array of booleans indicating where reserves are as True.
    timesteps : int, optional
        Number of timesteps to run the simulation over. The default is 100.
    r : float, optional
        Growth rate of the logistic population model. The default is 0.5.
    K : int or float, optional
        Carrying capacity of the logistic population model. The default is 50.
    m : float, optional
        Fraction of individuals that migrate each timestep. Possible values
        from 0 to 1. The default is 0.05.
    disturbance_rate : float, optional
        Chance that a disturbance occurs each timestep. Possible values from
        0 to 1. The default is 0.01.
    disturbance_severity : float, optional
        Fraction of individuals that are killed when a disturbance hits.
        Possible values from 0 to 1. The default is 0.5.
    disturbance_extent : int or float, optional
        Radius of disturbances. The default is 0.1.
    traveldist : int or float, optional
        The standard deviation of the Gaussian kernel used to spread migrants
        out. The default is 10.
    edge_effect : float, optional
        Whether a species prefers or dislikes edge habitat. Acts as a 
        multiplier on edge habitat carrying capacity. Values from 0 to 1 
        mean edge habitat supports less individuals than the interior, and 
        values geater than 1 mean the edge habitat supports more individuals
        than the interior. The default is 1.0, which represents no species
        preference for edge or interior habitat.
    seed : int, optional
        Seed for random number generation enables reproducibility. This
        controls random events such as disturbances. The default is None.

    Returns
    -------
    pop_history : list of (L, L) ndarrays
        Snapshot of the population grid after each timestep. pop_history[-1] 
        is the final state. The GUI uses this to scrub backward/forward 
        through time without re-running
    history : dict
        Dictionary with keys total_pop, occupancy, num_occupied_reserves, and
        disturbance_events. THe values are lists that contain the values of 
        these aggregated statistics for each timestep.

    """

    # error handling
    if not isinstance(landscape, np.ndarray) or landscape.ndim != 2:
        raise TypeError("landscape must be a 2D numpy array")
    if landscape.dtype != bool:
        raise TypeError(
            f"landscape must be a boolean array: {landscape.dtype}")
    if not isinstance(timesteps, int) or timesteps < 1:
        raise ValueError(
            f"timesteps must be a positive integer:{timesteps}")
    if r < 0:
        raise ValueError(f"r (growth rate) must be non-negative:{r}")
    if K <= 0:
        raise ValueError(f"K (carrying capacity) must be positive:{K}")
    if not 0.0 <= m <= 1.0:
        raise ValueError(
            f"m (migration fraction) must be between 0 and 1:{m}")
    if not 0.0 <= disturbance_rate <= 1.0:
        raise ValueError(
            f"disturbance_rate must be between 0 and 1: {disturbance_rate}")
    if not 0.0 <= disturbance_severity <= 1.0:
        raise ValueError(
            f"disturbance_severity must be between 0 and 1: {disturbance_severity}")
    if disturbance_extent < 0:
        raise ValueError(
            f"disturbance_extent must be non-negative: {disturbance_extent}")
    if traveldist <= 0:
        raise ValueError(f"traveldist must be positive: {traveldist}")
    if edge_effect < 0:
        raise ValueError(f"edge_effect must be non-negative: {edge_effect}")

    # seed the RNG for reproducible runs (used by the GUI's slider-drag mode)
    if seed is not None:
        np.random.seed(seed)

    L = landscape.shape[0]
    pop = np.zeros((L, L))  # create a population array matching landscape
    pop[landscape] = K * 0.2  # start at a fraction of carrying capacity

    # find reserves using ndimage
    # ndimage identifies connected reserve patches to track how many separate reserves are occupied (>10)
    reserveArrays, numReserves = ndimage.label(landscape)

    # edge-effect multiplier applied to the first-layer edge cells
    non_reserve = ~landscape
    expanded_non_reserve = ndimage.binary_dilation(non_reserve)
    edge_cells = landscape & expanded_non_reserve

    K_grid = np.full((L, L), K, dtype=float)
    K_grid[edge_cells] *= edge_effect

    history = {
        'total_pop': [],  # total population
        'occupancy': [],  # fraction of reserve cells with pop > 1
        'num_occupied_reserves': [],
        'disturbance_events': []  # disturbance event tuples with timestep, radius, center data
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
        # r is uniform across cells. K varies per cell via K_grid so that edge cells equilibrate at edge_effect * K
        pop[landscape] = pop[landscape] + r * \
            pop[landscape] * (1 - pop[landscape] / K_grid[landscape])
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
        pop[~landscape] = 0  # species can't survive outside habitat

        # 3. Disturbance
        # with some chance (per timestep), random fraction of reserve cells are impacted.
        # this models disturbances such as fires, stormes, diseases, etc.

        # randomly check if there will be a disturbance based on our rate (0-1)
        if np.random.rand() < disturbance_rate:
            # check for reserve cells (marked True)
            reserve_coords = np.argwhere(landscape)

            # randomly pick a reserve coord to be the center of the disturbance
            random_index = np.random.randint(len(reserve_coords))
            disturbCenterY, disturbCenterX = reserve_coords[random_index]

            y_grid, x_grid = np.ogrid[0:L, 0:L]
            distances = np.sqrt((y_grid - disturbCenterY) **
                                2 + (x_grid - disturbCenterX)**2)

            # reduce population if distance from center is less than disturbance_extent
            disturbed = distances <= disturbance_extent
            pop[disturbed] *= (1 - disturbance_severity)

            # record event so the GUI can draw a red circle on the landscape
            history['disturbance_events'].append(
                (t, int(disturbCenterY), int(
                    disturbCenterX), float(disturbance_extent))
            )

        # record this timestep's full grid (.copy() so future timesteps don't overwrite it)
        pop_history.append(pop.copy())

        history['total_pop'].append(float(pop.sum()))
        history['occupancy'].append(
            float((pop[landscape] > 1).sum() / landscape.sum()) * 100)

        occupiedReserve = 0
        for i in range(1, numReserves + 1):
            if pop[reserveArrays == i].sum() > 10:
                occupiedReserve += 1
        history['num_occupied_reserves'].append(occupiedReserve)

    return pop_history, history
