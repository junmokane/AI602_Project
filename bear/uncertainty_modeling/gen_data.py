import torch
import matplotlib.pyplot as plt
import numpy as np
import random


def gen_datagrid(start=-1, end=1, step=101, B=50, type='linear', size='big',
                 xrange=None, function=None, plot=False, verbose=False):
    '''Generate training data, [B, 2].
    '''
    extent = [start, end, start, end]

    x = np.linspace(start, end, step)
    y = np.linspace(start, end, step)
    xv, yv = np.meshgrid(x, y)
    meshgrid_data = torch.from_numpy(np.dstack([xv, yv]))
    
    if verbose:
        print(f"meshgrid size is : {meshgrid_data.size()}")

    # gen training data
    if function is None:
        if type is 'linear':
            def function(a):
                return int(-2 * a + 100)
        elif type is 'square':
            def function(a):
                return int(0.5 * (a - 30) ** 2 - 1.3 * a + 20)
        elif type is 'random':
            def function(a):
                return random.randint(0, step)
    
    def inside(y):
        return int(y) < step and int(y) >= 0
    if xrange is None:
        if size is 'big':
            z = random.randint(0, step//10)
            assert z+step//10 <= z+step-step//10, f"to small step, z:{z}, step:{step}"
            ind_x = list(range(z+step//10, z+step-step//10, 1))
        elif size is 'small':
            for exp in range(5):
                z = random.randint(0, 3 * step//10)
                try:
                    assert z+3*step//10 <= z+step-3*step//10
                    ind_x = list(range(z+3*step//10, z+step-3*step//10, 1))
                except:
                    print("find new function")
                    assert exp == 4, "Error, go to Seungjae"
    else:
        ind_x = xrange
    
    ind_y = [function(a) for a in ind_x if inside(function(a))]
    index = [a + b*step for (a, b) in zip(ind_x, ind_y)]
    
    meshgrid_data_lin = meshgrid_data.reshape((step*step, 2))
    train_data = meshgrid_data_lin[index, :].float()

    if plot:
        fig, (ax1) = plt.subplots(1, 1)

        mag = xv ** 2 + yv ** 2
        ax1.imshow(mag, extent=extent)
        ax1.set_title("plot")

        plot_meshgrid(ax1, meshgrid_data_lin, index)
        for ind in index:
            location = meshgrid_data_lin[ind]
            ax1.scatter(location[0], location[1], c="red")
        
        ind_x = [10, 91, 10, 91]
        ind_y = [10, 10, 91, 91]
        ind_index = [a + b*step for (a, b) in zip(ind_x, ind_y)]

        indicator = meshgrid_data_lin[ind_index, :]
        text = ["(10, 10)", "(91, 10)", "(10, 91)", "(91, 91)"]
        plot_meshgrid(ax1, meshgrid_data_lin, ind_index, text)
        
        plt.show()

    return train_data, index, meshgrid_data_lin


def plot_meshgrid(ax, meshgrid_data_lin, index, texts=None):
    '''plot from linearl viewed meshgrid data and index
    '''
    if texts is not None:
        index_texts = zip(index, texts)
    else:
        index_texts = zip(index)
    for ind_tex in index_texts:
        if len(ind_tex) == 1:
            ind = ind_tex
            location = meshgrid_data_lin[ind]
            ax.scatter(location[0], location[1], c="red")
        if len(ind_tex) == 2:
            (ind, tex) = ind_tex
            location = meshgrid_data_lin[ind]
            ax.text(location[0], location[1], tex, ha='center', va='bottom')
            ax.scatter(location[0], location[1], c="blue")


if __name__ == "__main__":
    gen_datagrid(plot=True, type='linear')
    gen_datagrid(plot=True, type='square')
    gen_datagrid(plot=True, type='random', size='small')
    def dummy(a):
        return int(-0.1 * (a-10)**2 + 3 * a +10)
    # We can assign function, which input x and output y!
    gen_datagrid(plot=True, function=dummy)
    # We can assign xrange!
    gen_datagrid(plot=True, xrange=list(range(1, 100, 2)), function=dummy)

