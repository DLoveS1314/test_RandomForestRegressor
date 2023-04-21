import pickle
def save_grid_search(grid_search,name):
    with open(name, 'wb') as f:
        pickle.dump(grid_search, f)
def get_grid_search(name):
    with open(name, 'rb') as f:
        grid_search = pickle.load(f)
    return grid_search