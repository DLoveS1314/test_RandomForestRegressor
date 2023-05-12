import matplotlib.pyplot as plt

def update_projection(ax, axi, projection='3d', fig=None):
    if fig is None:
        fig = plt.gcf()
    rows, cols, start, stop = axi.get_subplotspec().get_geometry()
    ax.flat[start].remove()
    ax.flat[start] = fig.add_subplot(rows, cols, start+1, projection=projection)
import matplotlib.projections
import numpy as np

# test data
x = np.linspace(-np.pi, np.pi, 10)

# plot all projections available
projections = matplotlib.projections.get_projection_names()

fig, ax = plt.subplots(nrows=1, ncols=len(projections), figsize=[3.5*len(projections), 4], squeeze=False)
for i, pro_i in enumerate(projections):
    update_projection(ax, ax.flat[i], pro_i)
    ax.flat[i].set_title(pro_i)
    try:
        ax.flat[i].grid(True)
        ax.flat[i].plot(x, x)
    except Exception as a:
        print(pro_i, a)
    
plt.tight_layout(pad=.5)