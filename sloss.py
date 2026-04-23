import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.signal import fftconvolve


def create_landscape(L=50, total_area=100, num_reserves=1):
    #create empty np array for landscape and reserves
    landscape = np.zeros((L, L), dtype=bool)
    
    #calculate approximately how large each reserve should be
    reserveArea = total_area/num_reserves
    reserveSide = int(np.sqrt(reserveArea))
    
    numReservesPlaced = 0
    attempts = 0
    
    while numReservesPlaced < num_reserves and attempts < 1000:
        
        #randomly place top left corner of each reserve in the landscape
        top = np.random.randint(0, L-reserveSide)
        left = np.random.randint(0, L-reserveSide)
        
        
        #place reserve if there isn't already another reserve there or adjacent
        #use 0 or L if we are at the edge so we don't get indexing errors
        #add extra cells to bottom        
        checktop = max(0,top-1)
        checkbottom = min(L,top+reserveSide+1)
        checkleft =  max(0, left-1)
        checkright = min(L, left+reserveSide+1)
        
        if not landscape[checktop:checkbottom, checkleft:checkright].any():
            landscape[top:top+reserveSide, left:left+reserveSide] = True
            numReservesPlaced += 1        
        attempts += 1
    
    #extra space that was not able to fit in the squares
    extraSpace = total_area-num_reserves*reserveSide**2
    addedExtra = 0
    attempts = 0
    while addedExtra < extraSpace and attempts < 1000:
        reserve_coords = np.argwhere(landscape) #current reserve cells
        
        # set to find edges
        edgecoords = []
        
        for x, y in reserve_coords:
            neighbors = [(x-1, y), (x+1, y), (x, y-1),(x, y+1)]
            nextneighbors = [(x-2, y), (x+2, y), (x, y-2),(x, y+2)]
            
            for nx, ny in neighbors:
                # Check if in bounds and  empty
                if 0 <= nx < L and 0 <= ny < L and not landscape[nx, ny]:
                    
                    #check that next neighbors are also empty
                    #this prevents connection of two adjacent reserves
                    for nnx, nny in nextneighbors:
                        if 0 <= nnx < L and 0 <= nny < L :
                            #if in bounds, check if currently reserve
                            if not landscape[nnx, nny]:
                                edgecoords.append((nx, ny))
                        else: #not in bounds, don't worry about it 
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

def run_simulation(landscape, timesteps=100, r=0.5, K=50, m=0.05, 
                   disturbance_rate=0.01, disturbance_severity=0.5, 
                   disturbance_extent=0.1, traveldist=10, ploteachtimestep=False):
    L = landscape.shape[0]
    pop = np.zeros((L, L)) # create a population array matching landscape
    pop[landscape] = K*0.2 #start at a fraction of carrying capacity
    
    #find reserves using ndimage
    reserveArrays, numReserves = ndimage.label(landscape)
    
    history = {
        'total_pop': [], #total population
        'occupancy': [],  # fraction of reserve cells with pop > 1
        'num_occupied_reserves': []
    }
    
    #kernel for immigration, with decreasing likelihood (Gaussian distrib) the further out we go
    kernelSize = int(6 * 10)
    #ensure odd kernel side length so our original cell is at the center
    if kernelSize % 2 == 0:
        kernelSize += 1
    center = kernelSize // 2 #find the center
    
    # Gaussian kernel
    y, x = np.ogrid[-center:center+1, -center:center+1]
    dispersal_kernel = np.exp(-(x**2 + y**2) / (2 * traveldist**2))
    #normalize so sum of probabilities equals 1
    dispersal_kernel = dispersal_kernel / dispersal_kernel.sum()
    
    #repeat process of growth, immigration, disturbance for timesteps
    for t in range(timesteps):
        #logistic growth of population in each cell
        pop[landscape] = pop[landscape] + r*pop[landscape] * (1-pop[landscape]/K)    
        pop = np.clip(pop, 0, None) #disallow negative populations
        
        #individuals moving
        emigrants = pop * m #how many individuals are leaving from each cell
        pop = pop * (1 - m) #each cell's base population decreases as some leave
        
        #immigrants move according to Gaussian kernel
        dispersed = fftconvolve(emigrants, dispersal_kernel, mode='same')
        
        pop += dispersed #add immigrants to their new home cells
        pop[~landscape] = 0 #species can only survive in reserve
        
        
        #randomly check if there will be a disturbance based on our rate (0-1)
        if np.random.rand() < disturbance_rate:
            reserve_coords = np.argwhere(landscape) #check for reserve cells
            #disturbance applies to a certain percent of the reserve cells
            numDisturbed = int(disturbance_extent * len(reserve_coords)) 
            disturbed_idx = np.random.choice(len(reserve_coords), numDisturbed, replace = False)
            disturbed = reserve_coords[disturbed_idx]

            pop[disturbed] *= (1 - disturbance_severity)
        
        
        history['total_pop'].append(float(pop.sum()))
        history['occupancy'].append(float((pop[landscape]>1).sum() / landscape.sum()))
        
        occupiedReserve = 0
        for i in range(1, numReserves + 1):
            if pop[reserveArrays == i].sum() > 10:
                occupiedReserve += 1
        history['num_occupied_reserves'].append(occupiedReserve)
        
        if ploteachtimestep:
            plot_landscape(pop, K)
        
    return pop, history
    

def plot_landscape(landscape, K):
    plt.imshow(landscape, vmin=0, vmax=K, cmap='viridis') #cmap max is carrying capacity
    plt.colorbar(label='Population')
    plt.show()
    
def plot_statistics_over_time(history):
    plt.plot(history['total_pop'])
    plt.title("Total Population over Time")
    plt.xlabel("Timestep")
    plt.ylabel("Population")
    plt.show()
    
    plt.plot(history['occupancy'])
    plt.title("Total reserve Cell Occupancy over Time")
    plt.xlabel("Timestep")
    plt.show()
    
    plt.plot(history['num_occupied_reserves'])
    plt.title("Total Num reserves Occupied over Time")
    plt.xlabel("Timestep")
    plt.show()

if __name__ == "__main__":
    K=50
    #single large
    landscape = create_landscape()
    pop, history = run_simulation(landscape, ploteachtimestep=False)
    plot_landscape(pop, K)   
    plot_statistics_over_time(history)
    
    #several small
    landscape = create_landscape(num_reserves = 10)
    pop, history = run_simulation(landscape, ploteachtimestep=False)
    plot_landscape(pop, K)   
    plot_statistics_over_time(history)
    
    #several medium
    landscape = create_landscape(num_reserves = 3)
    pop, history = run_simulation(landscape, ploteachtimestep=False, disturbance_rate=0.9, disturbance_extent=0.9)
    plot_landscape(pop, K)   
    plot_statistics_over_time(history)
