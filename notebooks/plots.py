#%pylab inline
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyiga'))
import pyiga

import scipy
import itertools

from pyiga import bspline, assemble, vform, geometry, vis, solvers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def grid_eval(f, grid):
    """Evaluate function `f` over the tensor grid `grid`."""
    if hasattr(f, 'grid_eval'):
        return f.grid_eval(grid)
    else:
        mesh = np.meshgrid(*grid, sparse=True, indexing='ijk')
        mesh.reverse() # convert order ZYX into XYZ
        values = f(*mesh)
        return _ensure_grid_shape(values, grid)

def plot_grid(x, y, ax=None, **kwargs):
    """Plot a grid over a geometry"""
    ax = ax or plt.gca()
    segs1 = np.stack((x, y), axis=2)
    segs2 = segs1.transpose(1, 0, 2)
    ax.add_collection(LineCollection(segs1, **kwargs))
    ax.add_collection(LineCollection(segs2, **kwargs))
    ax.autoscale()
    
    
    
#split the solution into its components (displacement in x- and y- direction)
# visualization per patch
def get_defplot(u, patches_u, kvs_u, MP_u, n_el):
    """Split solution vector into displacement components."""
    u1_funcs, u2_funcs = split_u(u, MP_u, kvs_u, patches_u)
    
    # evaluate displacement (and "pressure") over a grid in the parameter domain
    # grid variables
    x_el = n_el[0]
    y_el = n_el[1]
    #
    #x_el = n_el[1]
    #y_el = n_el[0]
    
    
    xgrid=np.linspace(0, 1, x_el)
    ygrid=np.linspace(0, 1, y_el)
    #xgrid = linspace(0, 1, ref)
    xygrid = (xgrid, ygrid)
    #len_xgrid= len(xgrid)
    vrange= None
   
    fig, ax = plt.subplots(figsize= (10,10))
    
    # visualization per patch
    for (u1_func, u2_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, patches_u): #u_funcs 
        dis1 = u1_func.grid_eval(xygrid) #x-value
        dis2 = u2_func.grid_eval(xygrid) #y-value
        dis = np.stack((dis1,dis2), axis=-1)
        G = geo.grid_eval(xygrid)
        plot_grid(G[..., 0], G[..., 1], ax=ax, color="lightgrey")
        plot_grid(G[..., 0] + dis[..., 0], G[..., 1] + dis[..., 1], ax=ax, color="black")
        
        C = np.sqrt(np.power(dis[..., 0], 2) + np.power(dis[..., 1], 2))
        if vrange is None:
            vrange = (C.min(), C.max())
            
        plt.pcolormesh(G[..., 0] + dis[..., 0], G[..., 1] + dis[..., 1], C, shading='gouraud', cmap='viridis', 
                                    vmin=vrange[0], vmax=vrange[1])
        #vis.plot_deformation(dis, ref, geo, ax, vmin=0.0, vmax=1.5e-3)
    plt.colorbar();
    plt.axis('equal')

    
#### for point evaluation & visualization ### 
def get_defplotPP(u, patches_u, kvs_u, MP_u, n_el, r_in, r_out):
    """Split solution vector into displacement components."""
    u1_funcs, u2_funcs = split_u(u, MP_u, kvs_u, patches_u)
    # evaluate displacement (and "pressure") over a grid in the parameter domain
    # grid variables, swap 
    x_el = n_el[0]
    y_el = n_el[1]
    
    xgrid=np.linspace(0, 1, x_el)
    ygrid=np.linspace(0, 1, y_el)
    xygrid = (xgrid, ygrid)
    #len_xgrid= len(xgrid)
    vrange= None
 
    fig, ax = plt.subplots(figsize= (7,7))
    ###
    
    max_xval_i = 0
    max_yval_i = 0
    max_rad_i = 0
    max_rad_o = 0
    idx_ri = 0
    idx_ro = 0
    count=0
    # visualization per patch
    for (u1_func, u2_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, patches_u): #u_funcs 
        print('\n patch:', count)
        dis1 = u1_func.grid_eval(xygrid) #x-value
        dis2 = u2_func.grid_eval(xygrid) #y-value
        dis = np.stack((dis1,dis2), axis=-1)
       
        G = geo.grid_eval(xygrid)
        x_val_i=[]
        y_val_i=[]
        x_val_o=[]
        y_val_o=[]
        
        i= 0
    
        radius_inner = np.sqrt( (G[i,0, 0]+dis1[i, 0])**2 + (G[i,0, 1]+dis2[i,0])**2)
        radius_outer = np.sqrt((G[i,y_el-1, 0]+dis1[i, y_el-1])**2 + (G[i,y_el-1, 1]+dis2[i,y_el-1])**2)
        print('inner radius_{}= {}'.format(i, radius_inner))
        print('outer radius_{}= {}'.format(i, radius_outer))
        print('displacement_inner= {}'.format(dis[i, 0,...]))
        print('displacement_outer= {}'.format(dis[i, y_el-1,...]))
        print( 'dis_inner_x/dis_outer_x: ', dis1[i, 0]/dis1[i,y_el-1])
        print( 'dis_inner_y/dis_outer_y: ', dis2[i, 0]/dis2[i, y_el-1])
        
        #print (' ratio:', radius_inner/radius_outer) # with respect to max. inner radius coord.
        
        print (' ratio_inner:', radius_inner/r_in) # with respect to inner radius
        print (' ratio_outer:', radius_outer/r_out) # with respect to outer radius
        
        plt.plot(G[i, 0, 0] + dis1[i, 0], G[i, 0, 1] + dis2[i, 0], 'ro') # inner radius
        plt.plot(G[i, y_el-1, 0] + dis1[i, y_el-1], G[i, y_el-1, 1] + dis2[i, y_el-1], 'bo') # outer radius
        #print(G[ y_el-1, 0, 0] , G[ y_el-1, 0, 1]) # (x, y) patch orientation!
        #plt.plot(G[y_el-1, y_el-1, 0], G[y_el-1, y_el-1, 1] , 'co') # outer radius
        plt.plot(G[0, y_el-1, 0], G[0, y_el-1, 1] , 'co') # outer radius
        plt.plot(G[0, 0, 0], G[0, 0, 1] , 'mo') # inner radius


        plot_grid(G[..., 0], G[..., 1], ax=ax, color="lightgrey")
        #plot(G[i, 0, 0] + dis1[i, 0], G[i, 0, 1] + dis2[i, 0], 'mo') # inner radius 
        #plot(G[i, y_el-1, 0] + dis1[i, y_el-1], G[i, y_el-1, 1] + dis2[i, y_el-1], 'co') # outer radius
        
        count+=1
       
        C = np.sqrt(np.power(dis[..., 0], 2) + np.power(dis[..., 1], 2))
        if vrange is None:
            vrange = (C.min(), C.max())
            
        plt.pcolormesh(G[..., 0] + dis[..., 0], G[..., 1] + dis[..., 1], C, shading='gouraud', cmap='viridis', 
                                    vmin=vrange[0], vmax=vrange[1])
        #vis.plot_deformation(dis, ref, geo, ax, vmin=0.0, vmax=1.5e-3)
    plt.colorbar();
    plt.axis('equal')

    
  ###################
## animation

"""Visualization functions."""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection


def animate_field(fields, patches_u, kvs_u, MP_u, vrange=None, res=(50,50),cmap=None, interval=50, progress=False, s_sol=3):
    """Animate a sequence of scalar fields over a geometry."""
    fields = list(fields)
 
    fig, ax = plt.subplots(figsize= (8,8))
    ax.set_xlim(left=-7.5, right=6)
    ax.set_ylim(bottom=-7, top=7.5)
    ax.set_aspect('equal')
    
    
    ar= np.linspace(1,3, s_sol)
    factor = ar[0]
    C = None
    vrange= None

    if np.isscalar(res):
        res = (res, res)

    # first solution
    #u= LS.complete(fields[0])
    u = fields[0]
    
    #Split solution vector into displacement components
    u1_funcs, u2_funcs = split_u(u, MP_u, kvs_u, patches_u)

    # evaluate displacement (and "pressure") over a grid in the parameter domain
    # visualization per patch
    for (u1_func, u2_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, patches_u): #u_funcs 
        grd = tuple(np.linspace(s[0], s[1], r) for (s, r) in zip(geo.support, res))
        XY = geo.grid_eval(grd)
        dis1 = u1_func.grid_eval(grd) #x-value
        dis2 = u2_func.grid_eval(grd) #y-value
        dis = np.stack((dis1,dis2), axis=-1)
        if C is None:
            C = np.sqrt(np.power(dis[..., 0], 2) + np.power(dis[..., 1], 2))
        if vrange is None:
            #vrange = (0.0, 1.5e-3)
            vrange = (C.min(), C.max())
        #vis.plot_grid(XY[..., 0] + dis[..., 0], XY[..., 1] + dis[..., 1], ax=ax, color="black")
        quadmesh = plt.pcolormesh(XY[..., 0] + factor* dis[..., 0], XY[..., 1] + factor*dis[..., 1], C, shading='gouraud', cmap='viridis',
                                    vmin=vrange[0], vmax=vrange[1], axes=ax)
        
   
    fig.colorbar(quadmesh, ax=ax);
    tqdm = vis.utils.progress_bar(progress)
    pbar = tqdm(total=len(fields))
    
    def anim_func(i):
        plt.cla()
        ax.set_xlim(left=-7.5, right=6)
        ax.set_ylim(bottom=-7, top=7.5)
        ax.set_aspect('equal')
        #factor = ar[i] # choose factor for deformation plot
        #u = LS.complete(fields[i])
        vrange= None
        
        u = fields[i]
        u1_funcs, u2_funcs = split_u(u, MP_u, kvs_u, patches_u)

    # visualization per patch
        for (u1_func, u2_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, patches_u): #u_funcs 
            grd = tuple(np.linspace(s[0], s[1], r) for (s, r) in zip(geo.support, res))
            XY = geo.grid_eval(grd)
            dis1 = u1_func.grid_eval(grd) #x-value
            dis2 = u2_func.grid_eval(grd) #y-value
            dis = np.stack((dis1,dis2), axis=-1)
            C = np.sqrt(np.power(dis[..., 0], 2) + np.power(dis[..., 1], 2))
            if vrange is None:
                #vrange = (0.0, 1.5e-3)
                vrange = (C.min(), C.max())
            quadmesh = plt.pcolormesh(XY[..., 0] + factor*dis[..., 0], XY[..., 1] + factor*dis[..., 1], C, shading='gouraud', cmap='viridis', 
                                      vmin=vrange[0], vmax=vrange[1], axes=ax)
        
            #quadmesh.set_array(C.ravel())
        pbar.update()
        if i == len(u) - 1:
            pbar.close()
  
    return animation.FuncAnimation(plt.gcf(), anim_func, frames=len(fields), interval=interval, repeat=False)

#############
##Split solution vector into displacement components
def split_u(u, MP_u, kvs_u, patches_u):
    #Split solution vector into displacement components
    u1 = u[:MP_u.numdofs] 
    u2 = u[MP_u.numdofs:]
    # restrict solution to each individual patch - BSpline functions
    u1_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u1)
           for p in range(len(patches_u))]
    u2_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u2)
           for p in range(len(patches_u))]
    return u1_funcs, u2_funcs


####
##Split solution vector into displacement components
def split_u3d(u, MP_u, kvs_u, patches_u):
    """Split solution vector into displacement components."""
    u1 = u[:MP_u.numdofs] 
    u2 = u[MP_u.numdofs:2*MP_u.numdofs]
    u3 = u[2*MP_u.numdofs:3*MP_u.numdofs]
   
    # restrict solution to each individual patch - BSpline functions
    u1_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u1)
           for p in range(len(patches_u))]
    u2_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u2)
           for p in range(len(patches_u))]
    u3_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u3)
           for p in range(len(patches_u))]
    return u1_funcs, u2_funcs, u3_funcs


### 3D ###

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
from mpl_toolkits import mplot3d


def plot_geo(geo, grid=10, gridx=None, gridy=None, gridz= None,
             res=30,
             linewidth=None, color='black',  **kwargs):
    """Plot a wireframe representation of a 2D geometry."""
    #fig = plt.figure()
    
    if geo.sdim == 1 and geo.dim == 2:
        print("plot_curve")
        return plot_curve(geo, res=res, linewidth=linewidth, color=color)
    
    #print("geo.sdim:", geo.sdim)
    #print("geo.dim:", geo.dim)
    
    if geo.dim == geo.sdim == 2:
        #assert geo.dim == geo.sdim == 2, 'Can only plot 2D geometries'
        if gridx is None: gridx = grid
        if gridy is None: gridy = grid
        supp = geo.support

        # if gridx/gridy is not an array, build an array with given number of ticks
        if np.isscalar(gridx):
            gridx = np.linspace(supp[0][0], supp[0][1], gridx)
        if np.isscalar(gridy):
            gridy = np.linspace(supp[1][0], supp[1][1], gridy)

        meshx = np.linspace(supp[0][0], supp[0][1], res)
        meshy = np.linspace(supp[1][0], supp[1][1], res)

        def plotline(pts, capstyle='butt'):
            plt.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth,
                     solid_joinstyle='round', solid_capstyle=capstyle)

        pts = grid_eval(geo, (gridx, meshy))
        plotline(pts[0, :, :], capstyle='round')
        for i in range(1, pts.shape[0] - 1):
            plotline(pts[i, :, :])
        plotline(pts[-1, :, :], capstyle='round')

        pts = grid_eval(geo, (meshx, gridy))
        plotline(pts[:, 0, :], capstyle='round')
        for j in range(1, pts.shape[1] - 1):
            plotline(pts[:, j, :])
        plotline(pts[:, -1, :], capstyle='round')

    
    if geo.sdim == geo.dim == 3:
        #print("3D geometry")
        # syntax for 3-D projection
        #ax = plt.axes(projection ='3d')
        #assert geo.dim == geo.sdim == 3, 'Can only plot 3D geometries'
        
        if gridx is None: gridx = grid
        if gridy is None: gridy = grid
        if gridz is None: gridz = grid
        supp = geo.support
        
        # if gridx/gridy is not an array, build an array with given number of ticks
        if np.isscalar(gridx):
            gridx = np.linspace(supp[0][0], supp[0][1], gridx)
            #print(gridx)
        if np.isscalar(gridy):
            gridy = np.linspace(supp[1][0], supp[1][1], gridy)
        if np.isscalar(gridz):
            gridz = np.linspace(supp[2][0], supp[2][1], gridz)

        meshx = np.linspace(supp[0][0], supp[0][1], res)
        #print(meshx)
        meshy = np.linspace(supp[1][0], supp[1][1], res)
        meshz = np.linspace(supp[2][0], supp[2][1], res)

        def plotline(pts, capstyle='butt'):
            ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], cmap=cm.coolwarm, linewidth=linewidth,  **kwargs)

        pts = grid_eval(geo, (gridx, meshy, meshz))# meshy
        plotline(pts[0, :, :], capstyle='round')# beginning
        for i in range(1, pts.shape[0] - 1): 
            plotline(pts[i, :, :]) # inbetween
        plotline(pts[-1, :, :], capstyle='round') # end

        pts = grid_eval(geo, (meshx, gridy, meshz))
        plotline(pts[:, 0, :], capstyle='round')
        for j in range(1, pts.shape[1] - 1):
            plotline(pts[:, j, :])
        plotline(pts[:, -1, :], capstyle='round')

        pts = grid_eval(geo, (meshx, meshy, gridz))
        plotline(pts[:, :, 0], capstyle='round')
        for k in range(1, pts.shape[2] - 1): 
            plotline(pts[:, :, k])
        plotline(pts[:, :, -1], capstyle='round')

        # plotting
        #ax.set_title('3D line plot')
        #plt.show()

        
        ###
        
    
### 3d surface
import matplotlib as mpl
from matplotlib.ticker import LinearLocator
from matplotlib import cm

def get_defplot3d(u, patches_u, kvs_u,  MP_u, n_el, grid=2, gridx=None, gridy=None, gridz= None,
             res=20, linewidth=None, color='black',  **kwargs):
    
    #u= LS.complete(u)
    """Split solution vector into displacement components."""
    u1 = u[:MP_u.numdofs] 
    u2 = u[MP_u.numdofs:2*MP_u.numdofs]
    u3 = u[2*MP_u.numdofs:3*MP_u.numdofs]
   
    # restrict solution to each individual patch - BSpline functions
    u1_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u1)
           for p in range(len(patches_u))]
    u2_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u2)
           for p in range(len(patches_u))]
    u3_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u3)
           for p in range(len(patches_u))]
    
    fig, ax = plt.subplots(figsize= (10,10))
    ax = plt.axes(projection ='3d')
    vrange = None
    
     # visualization per patch
    for (u1_func, u2_func, u3_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, u3_funcs, patches_u): #u_funcs 

        if gridx is None: gridx = grid
        if gridy is None: gridy = grid
        if gridz is None: gridz = grid
        supp = geo.support
        
        # if gridx/gridy is not an array, build an array with given number of ticks
        if np.isscalar(gridx):
            gridx = np.linspace(supp[0][0], supp[0][1], gridx) # e.g. linespace(0, 1, 10)
        if np.isscalar(gridy):
            gridy = np.linspace(supp[1][0], supp[1][1], gridy)
        if np.isscalar(gridz):
            gridz = np.linspace(supp[2][0], supp[2][1], gridz)

        meshx = np.linspace(supp[0][0], supp[0][1], res) # e.g. linespace(0, 1, 50)
        meshy = np.linspace(supp[1][0], supp[1][1], res)
        meshz = np.linspace(supp[2][0], supp[2][1], res)

    ## checkerboad pattern:
    # Create an empty array of strings with the same shape as the meshgrid, and
    # populate it with two colors in a checkerboard pattern.
        xlen = len(meshx)
        ylen = len(meshy)
        
        X, Y = np.meshgrid(meshx, meshy)
        colortuple = ('y', 'b')
        colors = np.empty(X.shape, dtype=str)
        for y in range(ylen):
            for x in range(xlen):
                 colors[y, x] = colortuple[(x + y) % len(colortuple)]
    
                
        def plotline(pts, capstyle='butt'):
            C = np.sqrt(np.power(pts[..., 1], 2) + np.power(pts[..., 2],2))
            #print(shape(C))
            #print('c',C)
            vrange = (C.min(), C.max())
            norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
            ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], linewidth=0, antialiased=False, shade = False, alpha = 0.3, 
                            facecolors=cm.viridis(C), vmin=vrange[0], vmax=vrange[1])
            #ax.plot_wireframe(pts[..., 0], pts[..., 1], pts[..., 2], linewidth=1, color='grey') # grid
            #ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], facecolors=colors, linewidth=0) # checkerboard
        
        #xgridyz = (gridx , meshy, meshz)
        xgridyz = (gridx,gridx, gridx)
        ygridxz = (meshx, gridy, meshz)
        zgridxy = (meshx, meshy, gridz)
        
        # y-grid 
        dis1 = u1_func.grid_eval(ygridxz) #x-value evaluated on 3-dim grid
        dis2 = u2_func.grid_eval(ygridxz) #y-value
        dis3 = u3_func.grid_eval(ygridxz) #z-value
        dis = np.stack((dis1,dis2, dis3), axis=-1) # displacement evaluated on 3-dim grid
    
        pts = grid_eval(geo, ygridxz) + dis  # + displacement y-coord
    
        C = np.sqrt(np.power(dis[..., 0],2) + np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
        #vrange = (C.min(), C.max())
        #norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
      
        plotline(pts[:, 0, :], capstyle='round')
        for j in range(1, pts.shape[1] - 1):
            plotline(pts[:, j, :])
        plotline(pts[:, -1, :], capstyle='round')

        # z-grid
        dis1 = u1_func.grid_eval(zgridxy) #x-value evaluated on 3-dim grid
        dis2 = u2_func.grid_eval(zgridxy) #y-value
        dis3 = u3_func.grid_eval(zgridxy) #z-value
        dis = np.stack((dis1,dis2, dis3), axis=-1) # displacement evaluated on 3-dim grid   
        
        pts = grid_eval(geo, zgridxy) + dis # # + displacement z-coord
        plotline(pts[:, :, 0], capstyle='round')
        for k in range(1, pts.shape[2] - 1): 
            plotline(pts[:, :, k])
        plotline(pts[:, :, -1], capstyle='round')
        
    #C = np.sqrt(np.power(dis[..., 0],2) + np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
    C = np.sqrt(np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
    #if vrange= None:
    vrange = (C.min(), C.max())
    norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
 
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, pad=0.2)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    plt.show()
    
####################
#split the solution into its components (displacement in x- and y- direction)
# visualization per patch , 2D only!
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from matplotlib.ticker import LinearLocator
from matplotlib import cm
### 3d surface

def get_defplotC(u, patches_u, kvs_u, MP_u, grid=2, gridx=None, gridy=None, gridz= None,
             res=20, linewidth=None, **kwargs):
    
    #u= LS.complete(u)
    """Split solution vector into displacement components."""
    u1 = u[:MP_u.numdofs] 
    u2 = u[MP_u.numdofs:2*MP_u.numdofs]
    u3 = u[2*MP_u.numdofs:3*MP_u.numdofs]
   
    # restrict solution to each individual patch - BSpline functions
    u1_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u1)
           for p in range(len(patches_u))]
    u2_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u2)
           for p in range(len(patches_u))]
    u3_funcs = [geometry.BSplineFunc(kvs_u, MP_u.global_to_patch(p) @ u3)
           for p in range(len(patches_u))]
    
    fig, ax = plt.subplots(figsize= (10,10))
    ax = plt.axes(projection ='3d')
    vrange = None
    
     # visualization per patch
    for (u1_func, u2_func, u3_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, u3_funcs, patches_u): #u_funcs 

        if gridx is None: gridx = grid
        if gridy is None: gridy = grid
        if gridz is None: gridz = grid
        supp = geo.support
        
        # if gridx/gridy is not an array, build an array with given number of ticks
        if np.isscalar(gridx):
            gridx = np.linspace(supp[0][0], supp[0][1], gridx) # e.g. linespace(0, 1, 10)
        if np.isscalar(gridy):
            gridy = np.linspace(supp[1][0], supp[1][1], gridy)
        if np.isscalar(gridz):
            gridz = np.linspace(supp[2][0], supp[2][1], gridz)

        meshx = np.linspace(supp[0][0], supp[0][1], res) # e.g. linespace(0, 1, 50)
        meshy = np.linspace(supp[1][0], supp[1][1], res)
        meshz = np.linspace(supp[2][0], supp[2][1], res)

        def plotlineC(pts, C, capstyle='butt'):
            #C = np.sqrt(np.power(pts[..., 1], 2) + np.power(pts[..., 2],2))
            #print(shape(C))
            vrange = (C.min(), C.max())
            #vrange = (0, 1.5e-5)
            
            norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
            ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], linewidth=0, antialiased=False, shade = False, alpha = 0.3,facecolors=cm.viridis(C), vmin=vrange[0], vmax=vrange[1])
            #ax.plot_wireframe(pts[..., 0], pts[..., 1], pts[..., 2], linewidth=1, color='grey') # grid
            #ax.plot_surface(pts[..., 0], pts[..., 1], pts[..., 2], facecolors=colors, linewidth=0) # checkerboard
        
        #xgridyz = (gridx , meshy, meshz)
        xgridyz = (gridx,gridx, gridx)
        ygridxz = (meshx, gridy, meshz)
        zgridxy = (meshx, meshy, gridz)
        
        # y-grid 
        dis1 = u1_func.grid_eval(ygridxz) #x-value evaluated on 3-dim grid
        dis2 = u2_func.grid_eval(ygridxz) #y-value
        dis3 = u3_func.grid_eval(ygridxz) #z-value
        dis = np.stack((dis1,dis2, dis3), axis=-1) # displacement evaluated on 3-dim grid
    
        pts = grid_eval(geo, ygridxz) + dis  # + displacement y-coord
    
        #C = np.sqrt( np.power(dis[..., 0], 2)+np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
        C = np.sqrt(np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
        #vrange = (C.min(), C.max())
        #norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
        #C = C[:, 0, :]
      
        plotlineC(pts[:, 0, :], C[:, 0, :])
        for j in range(1, pts.shape[1] - 1):
            plotlineC(pts[:, j, :], C[:, j, :])
        plotlineC(pts[:, -1, :], C[:, -1, :])

        # z-grid
        dis1 = u1_func.grid_eval(zgridxy) #x-value evaluated on 3-dim grid
        dis2 = u2_func.grid_eval(zgridxy) #y-value
        dis3 = u3_func.grid_eval(zgridxy) #z-value
        dis = np.stack((dis1,dis2, dis3), axis=-1) # displacement evaluated on 3-dim grid   
        
        pts = grid_eval(geo, zgridxy) + dis # # + displacement z-coord
        C = np.sqrt(np.power(dis[..., 0], 2)+ np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
        #vrange = (C.min(), C.max())
        #norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
        #C = C[:, :, 0]
        
        plotlineC(pts[:, :, 0],C[:, :, 0])
        for k in range(1, pts.shape[2] - 1): 
            plotlineC(pts[:, :, k], C[:, :, k])
        plotlineC(pts[:, :, -1], C[:, :, -1])
        
    #C = np.sqrt( np.power(dis[..., 0], 2)+ np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
    #C = np.sqrt( np.power(dis[..., 1], 2) + np.power(dis[..., 2],2))
    vrange = (C.min(), C.max())
    #vrange = (0, 1.5e-5)
    norm = mpl.colors.Normalize(vmin=vrange[0], vmax=vrange[1])
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax, pad=0.2)
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    
    plt.show()