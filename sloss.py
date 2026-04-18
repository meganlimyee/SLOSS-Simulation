import numpy as np
import matplotlib.pyplot as plt


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

def plot_landscape(landscape):
    plt.imshow(landscape)
    plt.show()

if __name__ == "__main__":
    #single large
    landscape = create_landscape()
    plot_landscape(landscape)
    
    #several small
    landscape = create_landscape(num_reserves = 10)
    plot_landscape(landscape)
    
    #several medium
    landscape = create_landscape(num_reserves = 3)
    plot_landscape(landscape)