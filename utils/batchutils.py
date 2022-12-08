import numpy as np

def get_neighbor_location(items, s2l):
    items = items.cpu().numpy()
    isl = []
    for i in range(items.shape[0]):
        # randomly select from the list of neighbors
        il = [s2l[j] for j in items[i]]
        isl.append(il)
    return np.array(isl).astype(np.int32)

