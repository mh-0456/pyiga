import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pyiga'))
import pyiga
#
import scipy
import itertools

from pyiga import bspline, assemble, vform, geometry, vis, solvers

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# assemble matrix
from scipy.sparse import coo_matrix, block_diag, bmat
from scipy.sparse import bsr_matrix

from geo_annulus import *

######## material parameter ###########
# human artery,
# Mu= 0.0314 # N/mm²
# Lam = 1.81879 # N/mm²

# coronary arteries (CSA)
Mu = 7.56 * 1e-3  # N/mm²
##Lam = 440.307 * 1e-3  # N/mm² --> Model reduction uses a different lambda for 2D!


# rubber, Lamé-coeff.
# 0.003 GPa = 3MPa
# 1 MPa = 1e⁶ Pa
#######################################

### 2D ###########


class multipatch_block_handler:

    # Takes array of multi-patch objects
    def __init__(self, multi_patches):
        self.multi_patches = multi_patches
        self.numpatches = multi_patches[0].numpatches
        for multi_patch in multi_patches:
            if multi_patch.numpatches != self.numpatches:
                print("Inconsistent numbers of patches")

    def patch_to_global(self, p):
        first = True
        for multi_patch in self.multi_patches:
            if first:
                X = multi_patch.patch_to_global(p)
                first = False
            else:
                X = scipy.sparse.block_diag((X, multi_patch.patch_to_global(p)))
        return X

    def compute_dirichlet_bcs(self, data):
        first = True
        p = 0
        offset = 0
        for multi_patch in self.multi_patches:
            data_p = []
            for item in data:
                # print(item[3])
                if len(item) < 4 or p in item[3]:
                    data_p.append((item[0], item[1], lambda *x: item[2](*x)[p]))
            if len(data_p) > 0:
                bcs_p = multi_patch.compute_dirichlet_bcs(data_p)
                # print('bcs_p', bcs_p)
                if first:
                    bcs = list(bcs_p)
                    first = False
                else:
                    # Indices need offset
                    bcs[0] = np.concatenate((bcs[0], bcs_p[0] + offset))
                    # Values are kept as-is
                    bcs[1] = np.concatenate((bcs[1], bcs_p[1]))
            offset += multi_patch.numdofs
            p += 1
        return tuple(bcs)

    def compute_local_offset_for_component(self, p, c):
        offset = 0
        for cc in range(c):
            dim = 1
            kvs, geo = self.multi_patches[cc].patches[p]
            for kv in kvs:
                dim *= kv.numdofs
            offset += dim
        return offset


##########################################


####################################################
# split the solution into its components (displacement in x- and y- direction) and convert to BSpline function
def get_components(u, kvs_u):
    """Split solution vector into displacement components."""
    N = np.prod(tuple(kv.numdofs for kv in kvs_u))
    assert u.shape[0] == 2 * N
    m_u = tuple(kv.numdofs for kv in kvs_u)
    u1 = u[:N].reshape(m_u)
    u2 = u[N:].reshape(m_u)
    U = np.stack((u1, u2), axis=-1)
    return bspline.BSplineFunc(kvs_u, U)


### maybe not needed
def get_components_u(u, kvs_u, m_u):
    """Split solution vector into displacement components."""
    N = np.prod(tuple(kv.numdofs for kv in kvs_u))
    m_u = tuple(kv.numdofs for kv in kvs_u)
    u1 = u[:N].reshape(m_u)
    u2 = u[N:].reshape(m_u)
    return np.stack((u1, u2), axis=-1)


###############################################

# Energy functional
def energy(u_p, kvs_ux, geo_ux, X):
    # Plane Stress assumption:
    Lam = 14.618 * 1e-3  # N/mm²
    dis = get_components(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).dot(grad(dis).T) ))'
    return sum(assemble.assemble(f'( Lam/2*tr({e_term})*tr({e_term}) + Mu*tr({e_term}.dot({e_term})) ) * aux * dx',
                                 kvs_ux, bfuns=[('aux', 1)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu,
                                 idmat=np.identity(2)).ravel())


# Nonlinear variational form
def nonlinear_form(u_p, kvs_ux, geo_ux, X):
    # Plane Stress assumption:
    Lam = 14.618 * 1e-3  # N/mm²
    dis = get_components(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).dot(grad(dis).T) ))'
    f_term = '( idmat + grad(dis) )'
    return assemble.assemble(f'inner( ( Lam*tr({e_term})*idmat + 2*Mu* {e_term} ).dot({f_term}), grad(v) ) * dx',
                             kvs_ux, bfuns=[('v', 2)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu,
                             idmat=np.identity(2)).ravel()


# Linearized variational form
def linearized_form(u_p, kvs_ux, geo_ux, X):
    # Plane Stress assumption:
    Lam = 14.618 * 1e-3  # N/mm²
    dis = get_components(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).T.dot(grad(dis)) ))'
    e_deriv = '(.5*( grad(u).T + grad(u) + grad(u).T.dot(grad(dis)) + grad(dis).T.dot(grad(u)) ))'
    f_term = '( idmat + grad(dis) )'
    f_deriv = 'grad(u)'
    return assemble.assemble(
        f'inner( ( Lam * tr({e_term}) * idmat + 2*Mu*{e_term} ).dot({f_deriv}) + ( Lam * tr({e_deriv}) * idmat + 2*Mu*{e_deriv}).dot({f_term}), grad(v) ) * dx',
        kvs_ux, bfuns=[('u', 2), ('v', 2)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu, idmat=np.identity(2))


############ assemble strategies ############

# derive energy functional (with Neumann + Robin)
def ass_energy(u, MP_block, kvs_j, neu_data=None, robin_data=None):
    j = 0
    for p in range(MP_block.numpatches):  # go through each patch
        X = MP_block.patch_to_global(p)
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]
        j_p = energy(u, kvs_ux, geo_ux, X)
        dis = get_components(X.T @ u, kvs_ux)

        j_R = 0  ###!!
        j_N = 0  ###!!

        # Robin-Data
        if robin_data != None:
            for item in robin_data:
                if item[0] == p:
                    # Rg =assemble.assemble('inner(g,aux)*ds', kvs_j, bfuns=[('aux',2)], geo=geo_ux, g=item[2], boundary=item[1])
                    # print(Rg.sum())
                    R_p = item[3] / 2 * assemble.assemble('inner(dis,dis) *g* aux * ds', kvs_j, bfuns=[('aux', 1)],
                                                          dis=dis, geo=geo_ux, g=item[2], boundary=item[1],
                                                          layout='packed')  # first attempt
                    j_R += R_p.sum()
        if neu_data != None:
            for item in neu_data:
                if item[0] == p:
                    N_e = -item[3] * assemble.assemble('inner(n, dis) * aux * ds', kvs_j, bfuns=[('aux', 1)],
                                                       geo=geo_ux, boundary=item[1], dis=dis)
                    # N_e  = assemble.assemble('inner(g,dis) *v *ds', kvs_j, bfuns=[('v',1)], geo=geo_ux, g=item[2], boundary=item[1], dis=dis)
                    # print('N_e=', N_e)
                    j_N += N_e.sum()

        j += (j_p + j_R - j_N)

    return j


####################################
## inefficient, should not be used ###
# assemble linear system and right-hand side (with Neumann + Robin)
def ass_nonlinsystem_RN_old(u, MP_block, robin_data=None, neu_data=None):
    first = True
    firstR = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy

        # The whole patch-local stiffness matrix
        A_p = linearized_form(u, kvs_ux, geo_ux, X)  # take linearized variatonal form

        # The patch-local Neumann boundary data, right-hand side
        # b_p = assemble.inner_products(kvs_ux, f, f_physical=True, geo=geo_ux).ravel() - nonlinear_form(u, kvs_ux, geo_ux, X)  # for arbitrary rhs
        b_p = -nonlinear_form(u, kvs_ux, geo_ux, X)

        # Robin-Data
        if robin_data != None:
            for item in robin_data:
                if item[0] == p:
                    AR_u = item[3] * assemble.assemble('u * v * ds', kvs_ux, bfuns=[('u', 1), ('v', 1)], geo=geo_ux,
                                                       # Here, only scalar!
                                                       format='csr', layout='blocked', boundary=item[1])

                    bdofs_R = assemble.boundary_dofs(kvs_ux, item[1], ravel=True)
                    for c in range(2):  # x and y
                        offset = MP_block.compute_local_offset_for_component(p, c)
                        if firstR:
                            AR = X.tocsr()[:, bdofs_R + offset] @ AR_u @ X.tocsr()[:, bdofs_R + offset].T
                            firstR = False
                        else:
                            AR += X.tocsr()[:, bdofs_R + offset] @ AR_u @ X.tocsr()[:, bdofs_R + offset].T

        if neu_data != None:
            for item in neu_data:
                if item[0] == p:
                    # N_en = assemble.assemble('inner(g,v)*ds', kvs_ux, bfuns=[('v',2)], geo=geo_ux, g=item[2], boundary=item[1], layout='packed')

                    # normal vector with loading
                    N_en = -item[3] * assemble.assemble('inner(n,v)*ds', kvs_ux, bfuns=[('v', 2)], geo=geo_ux,
                                                        boundary=item[1], layout='packed')
                    bdofs = assemble.boundary_dofs(kvs_ux, item[1], ravel=True)
                    for c in range(2):  # x and y
                        offset = MP_block.compute_local_offset_for_component(p, c)
                        for i in range(len(bdofs)):
                            b_p[bdofs[i] + offset] += N_en[i, 0, c]  # (43,1,2)

        if first:
            A = X @ A_p @ X.T
            b = X @ b_p
            first = False
        else:
            A += X @ A_p @ X.T
            b += X @ b_p

            # b is a residual of the form l-A(u)
    if robin_data == None:
        return A, b

    return A + AR, b - AR @ u


####################################

########################################
# assemble linear system and right-hand side (with Neumann)
def ass_nonlinsystem(u, MP_block, neu_data):
    first = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy

        # The whole patch-local stiffness matrix
        A_p = linearized_form(u, kvs_ux, geo_ux, X)  # take linearized variatonal form

        # The patch-local Neumann boundary data, right-hand side
        b_p = -nonlinear_form(u, kvs_ux, geo_ux, X)

        # Neumann data
        N_en, bdofs = ass_Neumann(p, MP_block, neu_data)
        for c in range(2):  # x and y
            offset = MP_block.compute_local_offset_for_component(p, c)
            for i in range(len(bdofs)):
                b_p[bdofs[i] + offset] += N_en[i, 0, c]  # (43,1,2)

        if first:
            A = X @ A_p @ X.T
            b = X @ b_p
            first = False
        else:
            A += X @ A_p @ X.T
            b += X @ b_p

    # b is a residual of the form l-A(u)
    return A, b


#######################

# assemble linear system and right-hand side (with Neumann)
def ass_nonlinsystem_RN(u, MP_block, neu_data, AR):
    A, b = ass_nonlinsystem(u, MP_block, neu_data)
    # b is a residual of the form l-A(u)
    return A + AR, b - AR @ u


#######################
#### rhs + Neumann BC 
def ass_rhs(u, MP_block, neu_data):
    first = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy

        # The patch-local Neumann boundary data, right-hand side
        b_p = -nonlinear_form(u, kvs_ux, geo_ux, X)

        N_en, bdofs = ass_Neumann(p, MP_block, neu_data)
        for c in range(2):  # x and y
            offset = MP_block.compute_local_offset_for_component(p, c)
            for i in range(len(bdofs)):
                b_p[bdofs[i] + offset] += N_en[i, 0, c]  # (43,1,2)

        if first:
            b = X @ b_p
            first = False
        else:
            b += X @ b_p

    # b is a residual of the form l-A(u)
    return b

#########################################

# for Robin bdc only
##################################
#### rhs + Neumann BC + Robin BC ####
def ass_rhs_RN(u, MP_block, neu_data, AR):
    b = ass_rhs(u, MP_block, neu_data)
    return b - AR @ u


################################################

### assemble Neumann bdc ####
def ass_Neumann(p, MP_block, neu_data):
    kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy
    for item in neu_data:
        if item[0] == p:
            # N_en = assemble.assemble('inner(g,v)*ds', kvs_ux, bfuns=[('v',2)], geo=geo_ux, g=item[2], boundary=item[1], layout='packed')
            # normal vector with loading
            N_en = -item[3] * assemble.assemble('inner(n,v)*ds', kvs_ux, bfuns=[('v', 2)], geo=geo_ux, boundary=item[1],
                                                layout='packed')
            bdofs = assemble.boundary_dofs(kvs_ux, item[1], ravel=True)
    return N_en, bdofs


### assemble Robin bdc ####

def ass_Robin(MP_block, robin_data):
    firstR = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]

        for item in robin_data:
            if item[0] == p:
                AR_u = item[3] * assemble.assemble('u * v * g *ds', kvs_ux, bfuns=[('u', 1), ('v', 1)], geo=geo_ux,
                                                   g=item[2],  # Here, only scalar!
                                                   format='csr', layout='blocked', boundary=item[1])
                bdofs_R = assemble.boundary_dofs(kvs_ux, item[1], ravel=True)
                for c in range(2):  # x and y
                    offset = MP_block.compute_local_offset_for_component(p, c)
                    if firstR:
                        AR = X.tocsr()[:, bdofs_R + offset] @ AR_u @ X.tocsr()[:, bdofs_R + offset].T
                        firstR = False
                    else:
                        AR += X.tocsr()[:, bdofs_R + offset] @ AR_u @ X.tocsr()[:, bdofs_R + offset].T
    return AR


### mass matrix ###
def ass_mass(MP_block):
    first = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy

        M_x = assemble.mass(kvs_ux, geo_ux)
        M_p = scipy.sparse.bmat(
            [[M_x, None],
             [None, M_x]], format='csr')

        if first:
            M = X @ M_p @ X.T
            first = False
        else:
            M += X @ M_p @ X.T

    return M


##########################################

######  3 D #########################
###############
# split the solution into its components (displacement in x- and y- direction) and convert to BSpline function
def get_components3d(u, kvs_u):
    """Split solution vector into displacement components."""
    # print("shape_u", shape(u))
    N = np.prod(tuple(kv.numdofs for kv in kvs_u))
    assert u.shape[0] == 3 * N
    m_u = tuple(kv.numdofs for kv in kvs_u)
    u1 = u[:N].reshape(m_u)
    u2 = u[N:2 * N].reshape(m_u)
    u3 = u[2 * N:3 * N].reshape(m_u)
    U = np.stack((u1, u2, u3), axis=-1)
    return bspline.BSplineFunc(kvs_u, U)


###############################################

# Energy functional
def energy3d(u_p, kvs_ux, geo_ux, X):
    Lam = 440.307 * 1e-3  # N/mm²
    dis = get_components3d(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).dot(grad(dis).T) ))'
    return sum(assemble.assemble(f'( Lam/2*tr({e_term})*tr({e_term}) + Mu*tr({e_term}.dot({e_term})) ) * aux * dx',
                                 kvs_ux, bfuns=[('aux', 1)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu,
                                 idmat=np.identity(3)).ravel())


# Nonlinear variational form
def nonlinear_form3d(u_p, kvs_ux, geo_ux, X):
    Lam = 440.307 * 1e-3  # N/mm²
    dis = get_components3d(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).dot(grad(dis).T) ))'
    f_term = '( idmat + grad(dis) )'
    return assemble.assemble(f'inner( ( Lam*tr({e_term})*idmat + 2*Mu* {e_term} ).dot({f_term}), grad(v) ) * dx',
                             kvs_ux, bfuns=[('v', 3)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu,
                             idmat=np.identity(3)).ravel()


# Linearized variational form
def linearized_form3d(u_p, kvs_ux, geo_ux, X):
    Lam = 440.307 * 1e-3  # N/mm²
    dis = get_components3d(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).T.dot(grad(dis)) ))'
    e_deriv = '(.5*( grad(u).T + grad(u) + grad(u).T.dot(grad(dis)) + grad(dis).T.dot(grad(u)) ))'
    f_term = '( idmat + grad(dis) )'
    f_deriv = 'grad(u)'
    return assemble.assemble(
        f'inner( ( Lam * tr({e_term}) * idmat + 2*Mu*{e_term} ).dot({f_deriv}) + ( Lam * tr({e_deriv}) * idmat + 2*Mu*{e_deriv}).dot({f_term}), grad(v) ) * dx',
        kvs_ux, bfuns=[('u', 3), ('v', 3)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu, idmat=np.identity(3))


### mass matrix ###
def ass_mass3d(MP_block):
    first = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy

        M_x = assemble.mass(kvs_ux, geo_ux)
        M_p = scipy.sparse.bmat(
            [[M_x, None, None],
             [None, M_x, None],
             [None, None, M_x]], format='csr')

        if first:
            M = X @ M_p @ X.T
            first = False
        else:
            M += X @ M_p @ X.T

    return M


########################################
# assemble linear system and right-hand side (with Neumann)
def ass_nonlinsystem3d(u, MP_block, neu_data):
    first = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy

        # The whole patch-local stiffness matrix
        A_p = linearized_form3d(u, kvs_ux, geo_ux, X)  # take linearized variatonal form

        # The patch-local Neumann boundary data, right-hand side
        b_p = -nonlinear_form3d(u, kvs_ux, geo_ux, X)

        # Neumann data
        N_e, bdofs = ass_Neumann3d(p, MP_block, neu_data)
        for c in range(3):  # x and y
            offset = MP_block.compute_local_offset_for_component(p, c)
            for i in range(len(bdofs)):
                b_p[bdofs[i] + offset] += N_e[c, i]  # (43,1,2)

        if first:
            A = X @ A_p @ X.T
            b = X @ b_p
            first = False
        else:
            A += X @ A_p @ X.T
            b += X @ b_p
            
    return A, b


#######################
# assemble linear system and right-hand side (with Neumann)
def ass_nonlinsystem_RN3d(u, MP_block, neu_data, AR):
    A, b = ass_nonlinsystem3d(u, MP_block, neu_data)
    return A + AR, b - AR @ u


###########################
### assemble Neumann bdc ####

def ass_Neumann3d(p, MP_block, neu_data):
    kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy
    for item in neu_data:
        if item[0] == p:
           # with function gN
           # N_e = item[3] * assemble.assemble('inner(g,v)*ds', kvs_ux, geo=geo_ux, g=item[2], bfuns=[('v', 3)],
                                             # symmetric=True, boundary=item[1]).ravel()

            # normal vector with loading!
            N_e = -item[3] * assemble.assemble('inner(n,v)*ds', kvs_ux, bfuns=[('v',3)], geo=geo_ux, boundary=item[1]).ravel()
            #nv= -item[3] * assemble.assemble('inner(n,v)*ds', kvs_ux, bfuns=[('v',3)], geo=geo_ux, boundary=item[1])
            #nv= (nv.sum(axis=(0,1,2)))
            #print('\n normal vector:',nv)
            #nn= nv/np.linalg.norm(nv)
            #print(' \n normed normal vector:',nn) #normed normal vector

            N_e = N_e.reshape(3, -1)  # 3dim, -1 remaining factor to get to the total number of elements
            bdofs = assemble.boundary_dofs(kvs_ux, item[1], ravel=True)
    return N_e, bdofs


### assemble Robin bdc ####

def ass_Robin3d(MP_block, robin_data):
    firstR = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]

        for item in robin_data:
            if item[0] == p:
                AR_u = item[3] * assemble.assemble('u * v *g*ds', kvs_ux, bfuns=[('u', 1), ('v', 1)], geo=geo_ux,
                                                   g=item[2],  # Here, only scalar!
                                                   format='csr', layout='blocked', boundary=item[1])
                bdofs_R = assemble.boundary_dofs(kvs_ux, item[1], ravel=True)
                for c in range(3):  # x and y
                    offset = MP_block.compute_local_offset_for_component(p, c)
                    if firstR:
                        AR = X.tocsr()[:, bdofs_R + offset] @ AR_u @ X.tocsr()[:, bdofs_R + offset].T
                        firstR = False
                    else:
                        AR += X.tocsr()[:, bdofs_R + offset] @ AR_u @ X.tocsr()[:, bdofs_R + offset].T
    return AR


#######################
#### rhs + Neumann BC 
def ass_rhs3d(u, MP_block, neu_data):
    first = True

    for p in range(MP_block.numpatches):
        X = MP_block.patch_to_global(p)

        # All the geometries are supposed to be equal; also kvs_ux and kvs_uy are equal
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy

        b_p = -nonlinear_form3d(u, kvs_ux, geo_ux, X)

        N_e, bdofs = ass_Neumann3d(p, MP_block, neu_data)
        for c in range(2):  # x and y
            offset = MP_block.compute_local_offset_for_component(p, c)
            for i in range(len(bdofs)):
                b_p[bdofs[i] + offset] += N_e[c, i]  # (43,1,2)

        if first:
            b = X @ b_p
            first = False
        else:
            b += X @ b_p
            
    return b


#########################################

# for Robin bdc only
##################################
#### rhs + Neumann BC + Robin BC ####
def ass_rhs_RN3d(u, MP_block, neu_data, AR):
    b = ass_rhs3d(u, MP_block, neu_data)
    return b - AR @ u


################################################
# derive energy functional (with Neumann + Robin)
def ass_energy3d(u, MP_block, kvs_j, neu_data=None, robin_data=None):
    j = 0
    for p in range(MP_block.numpatches):  # go through each patch
        X = MP_block.patch_to_global(p)
        kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]
        j_p = energy3d(u, kvs_ux, geo_ux, X)
        dis = get_components3d(X.T @ u, kvs_ux)

        j_R = 0  ###!!
        j_N = 0  ###!!

        # Robin-Data
        if robin_data != None:
            for item in robin_data:
                if item[0] == p:
                    # Rg =assemble.assemble('inner(g,aux)*ds', kvs_j, bfuns=[('aux',2)], geo=geo_ux, g=item[2], boundary=item[1])
                    # print(Rg.sum())
                    R_p = item[3] / 2 * assemble.assemble('inner(dis,dis) *g* aux * ds', kvs_j, bfuns=[('aux', 1)],
                                                          dis=dis, geo=geo_ux, g=item[2], boundary=item[1],
                                                          layout='packed')  # first attempt
                    j_R += R_p.sum()
        if neu_data != None:
            for item in neu_data:
                if item[0] == p:
                    # N_e = -item[3]*assemble.assemble('inner(n, dis) * aux * ds', kvs_j, bfuns=[('aux',1)], geo=geo_ux, boundary=item[1], dis=dis)
                    N_e = item[3] * assemble.assemble('inner(g,dis) *v *ds', kvs_j, bfuns=[('v', 1)], geo=geo_ux,
                                                      g=item[2], boundary=item[1], dis=dis)
                    # print('N_e=', N_e)
                    j_N += N_e.sum()

        j += (j_p + j_R - j_N)

    return j


####################################
###################################
