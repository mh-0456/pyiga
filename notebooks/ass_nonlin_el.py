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
#Lam = 440.307 * 1e-3  # N/mm²


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
    Lam = 14.618 * 1e-3  # N/mm²
    dis = get_components(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).dot(grad(dis).T) ))'
    return sum(assemble.assemble(f'( Lam/2*tr({e_term})*tr({e_term}) + Mu*tr({e_term}.dot({e_term})) ) * aux * dx',
                                 kvs_ux, bfuns=[('aux', 1)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu,
                                 idmat=np.identity(2)).ravel())


# Nonlinear variational form
def nonlinear_form(u_p, kvs_ux, geo_ux, X):
    Lam = 14.618 * 1e-3  # N/mm²
    dis = get_components(X.T @ u_p, kvs_ux)
    e_term = '(.5*( grad(dis).T + grad(dis) + grad(dis).dot(grad(dis).T) ))'
    f_term = '( idmat + grad(dis) )'
    return assemble.assemble(f'inner( ( Lam*tr({e_term})*idmat + 2*Mu* {e_term} ).dot({f_term}), grad(v) ) * dx',
                             kvs_ux, bfuns=[('v', 2)], geo=geo_ux, dis=dis, Lam=Lam, Mu=Mu,
                             idmat=np.identity(2)).ravel()


# Linearized variational form
def linearized_form(u_p, kvs_ux, geo_ux, X):
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
        # b_p = assemble.inner_products(kvs_ux, f, f_physical=True, geo=geo_ux).ravel() - nonlinear_form(u, kvs_ux, geo_ux, X)  # for arbitrary rhs
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
        # b_p = assemble.inner_products(kvs_ux, f, f_physical=True, geo=geo_ux).ravel() - nonlinear_form(u, kvs_ux, geo_ux, X)  # for arbitrary rhs
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
    # b is a residual of the form l-A(u)
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
                AR_u = item[3] * assemble.assemble('u * v *g*ds', kvs_ux, bfuns=[('u', 1), ('v', 1)], geo=geo_ux,
                                                   g=item[2],  # Here, only scalar!
                                                   format='csr', layout='blocked', boundary=item[1])

                # AR_u = item[3] * assemble.assemble('u * v * ds', kvs_ux, bfuns=[('u',1),('v',1)], geo=geo_ux, format='csr', layout='blocked',boundary=item[1])
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
        # b_p = assemble.inner_products(kvs_ux, f, f_physical=True, geo=geo_ux).ravel() - nonlinear_form(u, kvs_ux, geo_ux, X)  # for arbitrary rhs
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

            # b is a residual of the form l-A(u)
    return A, b


#######################


# assemble linear system and right-hand side (with Neumann)
def ass_nonlinsystem_RN3d(u, MP_block, neu_data, AR):
    A, b = ass_nonlinsystem3d(u, MP_block, neu_data)
    # b is a residual of the form l-A(u)
    return A + AR, b - AR @ u


######


### assemble Neumann bdc ####
def ass_Neumann3d(p, MP_block, neu_data):
    kvs_ux, geo_ux = MP_block.multi_patches[0].patches[p]  # kvs_uy, geo_uy
    for item in neu_data:
        if item[0] == p:
            # with function gN!
            N_e = item[3] * assemble.assemble('inner(g,v)*ds', kvs_ux, geo=geo_ux, g=item[2], bfuns=[('v', 3)],
                                              symmetric=True, boundary=item[1]).ravel()

            # normal vector with loading
            # N_e = -item[3] * assemble.assemble('inner(n,v)*ds', kvs_ux, bfuns=[('v',3)], geo=geo_ux, boundary=item[1]).ravel()
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

        # The patch-local Neumann boundary data, right-hand side
        # b_p = assemble.inner_products(kvs_ux, f, f_physical=True, geo=geo_ux).ravel() - nonlinear_form(u, kvs_ux, geo_ux, X)  # for arbitrary rhs
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

            # b is a residual of the form l-A(u)
    return b


#########################################

# for Robin bdc only
##################################
#### rhs + Neumann BC + Robin BC ####
def ass_rhs_RN3d(u, MP_block, neu_data, AR):
    b = ass_rhs3d(u, MP_block, neu_data)
    # b is a residual of the form l-A(u)
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
# for data evaluation
## 2D #######

# incremental loading for 2D Robin
def func_Robin_2d(MP_block, AR= None):
    # solve linearized variational problem - iterative, without line-search
    X = MP_block.patch_to_global(2)  # patch 2
    dd = shape(X)[0]
    # initial value
    u = np.zeros(dd)

    epsilon = 1e-4  # 1e-5

    solutions = []
    iter_counts = []

    ### Linear elasticity for largest loading
    l_val = loading[-1]  # take last loading value
    neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val),
                (3, 'left', gN, l_val)]  # set neumann

    A, b = ass_nonlinsystem_RN(u, MP_block, neu_data, AR)
    M = ass_mass(MP_block)
    Minv = make_solver(M)
    r0 = np.transpose(b).dot(Minv.dot(b))  # L2-norm

    # print("Norm of rhs for max loading: {}".format(r0))
    # print("Tolerance:                   {}".format(r0*epsilon))
    ###

    # incremental loading # ----------------------------------
    for t in range(len(loading)):  # time steps
        print(" \n \n {}. loading: {} \n".format(t + 1, loading[t]))

        # set Neumann data via incremental loading
        l_val = loading[t]
        neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val),
                    (3, 'left', gN, l_val)]  # set neumann

        count = 0
        while True:
            count += 1

            # Assemble matrices and rhs in every iteration step
            if count == 1:
                A, b = ass_nonlinsystem_RN(u, MP_block, neu_data, AR)
                r = np.transpose(b).dot(Minv.dot(b))  # dual of L2-norm

            print(count)

            # # solve system # #
            # u_d = make_solver(A).dot(b)
            u_d = make_solver(A).dot(b)
            u += u_d

            # Compute new non-linear residual, already to be used for next iteration (*)
            A, b = ass_nonlinsystem_RN(u, MP_block, neu_data, AR)
            r = np.transpose(b).dot(Minv.dot(b))  # dual of L2-norm

            if r < epsilon * r0:  # break condition
                break
            elif count == 30:
                break
        # 
        w1 = np.abs(make_solver(A).dot(b)-u)        #TODO: We do not want to solve that often!!
        w2 = np.abs(b - A.dot(u))
        err= np.sqrt(np.inner(w1,w2))
        print('u= ', u)
        iter_counts.append(count)
        # solutions.append(np.array(u))
    return u, r, iter_counts, err

##################

# incremental loading for 2D Dirichlet
def func_Dirichlet_2d(MP_block):
    # solve linearized variational problem - iterative, without line-search
    X = MP_block.patch_to_global(2)  # patch 2
    dd = shape(X)[0]
    # initial value
    u = np.zeros(dd)

    epsilon = 1e-4  # 1e-5

    solutions = []
    stepsizes = []
    ud_array = []
    iter_counts = []

    ### Linear elasticity for largest loading
    l_val = loading[-1]  # take last loading value
    neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val), (3, 'left', gN, l_val)]  # set neumann

    A, b = ass_nonlinsystem(u, MP_block, neu_data)
    LS = assemble.RestrictedLinearSystem(A, b, bc) # solve linearized system
    rhs= LS.complete(LS.b)
    r0 = np.transpose(rhs).dot(Minv.dot(rhs)) #L2-norm

    M = ass_mass(MP_block)
    Minv = make_solver(M)

    # print("Norm of rhs for max loading: {}".format(r0))
    # print("Tolerance:                   {}".format(r0*epsilon))
    ###

    # incremental loading # ----------------------------------
    for t in range(len(loading)):  # time steps
        print(" \n \n {}. loading: {} \n".format(t + 1, loading[t]))

        # set Neumann data via incremental loading
        l_val = loading[t]
        neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val), (3, 'left', gN, l_val)]  # set neumann

        count = 0
        while True:
            count += 1

            # Assemble matrices and rhs in every iteration step
            if count == 1:
                A, b = ass_nonlinsystem(u, MP_block, neu_data)
                LS = assemble.RestrictedLinearSystem(A, b, bc) # solve linearized system
                rhs= LS.complete(LS.b)
                r = np.transpose(rhs).dot(Minv.dot(rhs)) #L2-norm
            print(count)
            
            # # solve system # # 
            u_d = make_solver(LS.A).dot(LS.b)
            u_d = LS.complete(u_d)
            u += u_d        

            # Compute new non-linear residual, already to be used for next iteration (*)
            A, b = ass_nonlinsystem(u, MP_block, neu_data)
            LS = assemble.RestrictedLinearSystem(A, b, bc) # solve linearized system
            rhs= LS.complete(LS.b)
            r = np.transpose(rhs).dot(Minv.dot(rhs)) #L2-norm

            if r < epsilon * r0:  # break condition
                break
            elif count == 30:
                break
        #
        w1 = np.abs(make_solver(LS.A).dot(LS.b)-LS.restrict(u))        #TODO: We do not want to solve that often!!
        w2 = np.abs(LS.b - LS.A.dot(LS.restrict(u)))
        err= np.sqrt(np.inner(w1,w2))
        #print('Error=', err)
        sol = LS.restrict(u)
        print('u= ', sol)
        #solutions.append(LS.restrict(u))
        iter_counts.append(count)
        # solutions.append(np.array(u))
    return sol, r, iter_counts, err


##############################
## define h_rate !

def func_p_2d(p, m, n_el, geos, str_bc):
    multi =m # pp  # FEM
    # multi= 1 # classical IgA
    kvs_u = tuple(bspline.make_knots(p, 0.0, 1.0, n, mult=multi) for n in n_el)  # or : mult=2
    patches_u = [(kvs_u, g) for g in geos]

    # Here we auto-detect the interfaces between the patches.
    MP_u = assemble.Multipatch(patches_u, automatch=True)
    # dofs
    dofs= MP_u.numdofs
    
    # Multipatch objects for all variables
    MP_block = multipatch_block_handler([MP_u, MP_u])
    # set up Dirichlet boundary conditions
    bc = MP_block.compute_dirichlet_bcs([
        (1, 'right', g_zero)
    ])

    # define constant spline functions for integration
    kvs_j = tuple(bspline.make_knots(0, 0.0, 1.0, n, mult=multi) for n in n_el)  # constant basis vector for integration

    count = 0

    x_el = n_el[0]
    y_el = n_el[1]
    xgrid = np.linspace(0, 1, x_el)
    ygrid = np.linspace(0, 1, y_el)
    xygrid = (xgrid, ygrid)
    if str_bc =='robin':
        # assemble Robin-matrix
        AR = ass_Robin(MP_block, robin_data)
        u, res, it, err= func_Robin_2d(MP_block, AR)
    if str_bc =='dirichlet':
        u, res, it, err = func_Dirichlet_2d(MP_block)
        
    u1_funcs, u2_funcs = split_u(u, MP_u, kvs_u, patches_u)
    for (u1_func, u2_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, patches_u):  # u_funcs
        count += 1
        if count == 2: # displacement for 3rd patch
            print('\n patch:', count)
            # func_p(pp, geo, n_el, xygrid)
            G = geo.grid_eval(xygrid)

            dis1 = u1_func.grid_eval(xygrid)  # x-value
            dis2 = u2_func.grid_eval(xygrid)  # y-value
            radius_inner = np.sqrt( (G[0,0, 0]+dis1[0, 0])**2 + (G[0,0, 1]+dis2[0,0])**2) # inner radius (left domain)
            return radius_inner, res, it, err,dofs
        
############################
##############################
### 3D #####

####################################
# incremental loading for 3D Robin
def func_Robin_3d(MP_block, AR= None):
    # solve linearized variational problem - iterative, without line-search
    X = MP_block.patch_to_global(2)  # patch 2
    dd = shape(X)[0]
    # initial value
    u = np.zeros(dd)

    epsilon = 1e-4  # 1e-5

    solutions = []
    iter_counts = []

    ### Linear elasticity for largest loading
    l_val = loading[-1]  # take last loading value
    neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val),
                (3, 'left', gN, l_val)]  # set neumann

    A, b = ass_nonlinsystem_RN3d(u, MP_block, neu_data, AR)
    M = ass_mass3d(MP_block)
    Minv = make_solver(M)
    r0 = np.transpose(b).dot(Minv.dot(b))  # L2-norm

    # print("Norm of rhs for max loading: {}".format(r0))
    # print("Tolerance:                   {}".format(r0*epsilon))
    ###

    # incremental loading # ----------------------------------
    for t in range(len(loading)):  # time steps
        print(" \n \n {}. loading: {} \n".format(t + 1, loading[t]))

        # set Neumann data via incremental loading
        l_val = loading[t]
        neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val),
                    (3, 'left', gN, l_val)]  # set neumann

        count = 0
        while True:
            count += 1

            # Assemble matrices and rhs in every iteration step
            if count == 1:
                A, b = ass_nonlinsystem_RN3d(u, MP_block, neu_data, AR)
                r = np.transpose(b).dot(Minv.dot(b))  # dual of L2-norm

            print(count)

            # # solve system # #
            # u_d = make_solver(A).dot(b)
            u_d = make_solver(A).dot(b)
            u += u_d

            # Compute new non-linear residual, already to be used for next iteration (*)
            A, b = ass_nonlinsystem_RN3d(u, MP_block, neu_data, AR)
            r = np.transpose(b).dot(Minv.dot(b))  # dual of L2-norm

            if r < epsilon * r0:  # break condition
                break
            elif count == 30:
                break
        # 
        print('u= ', u)
        iter_counts.append(count)
        # solutions.append(np.array(u))
    return u, r, iter_counts

##################

# incremental loading for 3D Dirichlet
def func_Dirichlet_3d(MP_block):
    # solve linearized variational problem - iterative, without line-search
    X = MP_block.patch_to_global(2)  # patch 2
    dd = shape(X)[0]
    # initial value
    u = np.zeros(dd)

    epsilon = 1e-4  # 1e-5

    solutions = []
    stepsizes = []
    ud_array = []
    iter_counts = []

    ### Linear elasticity for largest loading
    l_val = loading[-1]  # take last loading value
    neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val), (3, 'left', gN, l_val)]  # set neumann

    A, b = ass_nonlinsystem3d(u, MP_block, neu_data)
    LS = assemble.RestrictedLinearSystem(A, b, bc) # solve linearized system
    rhs= LS.complete(LS.b)
    r0 = np.transpose(rhs).dot(Minv.dot(rhs)) #L2-norm

    M = ass_mass3d(MP_block)
    Minv = make_solver(M)

    # print("Norm of rhs for max loading: {}".format(r0))
    # print("Tolerance:                   {}".format(r0*epsilon))
    ###

    # incremental loading # ----------------------------------
    for t in range(len(loading)):  # time steps
        print(" \n \n {}. loading: {} \n".format(t + 1, loading[t]))

        # set Neumann data via incremental loading
        l_val = loading[t]
        neu_data = [(0, 'left', gN, l_val), (1, 'left', gN, l_val), (2, 'left', gN, l_val), (3, 'left', gN, l_val)]  # set neumann

        count = 0
        while True:
            count += 1

            # Assemble matrices and rhs in every iteration step
            if count == 1:
                A, b = ass_nonlinsystem3d(u, MP_block, neu_data)
                LS = assemble.RestrictedLinearSystem(A, b, bc) # solve linearized system
                rhs= LS.complete(LS.b)
                r = np.transpose(rhs).dot(Minv.dot(rhs)) #L2-norm
            print(count)
            
            # # solve system # # 
            u_d = make_solver(LS.A).dot(LS.b)
            u_d = LS.complete(u_d)
            u += u_d        

            # Compute new non-linear residual, already to be used for next iteration (*)
            A, b = ass_nonlinsystem3d(u, MP_block, neu_data)
            LS = assemble.RestrictedLinearSystem(A, b, bc) # solve linearized system
            rhs= LS.complete(LS.b)
            r = np.transpose(rhs).dot(Minv.dot(rhs)) #L2-norm

            if r < epsilon * r0:  # break condition
                break
            elif count == 30:
                break
        # 
        sol = LS.restrict(u)
        print('u= ', sol)
        #solutions.append(LS.restrict(u))
        iter_counts.append(count)
        # solutions.append(np.array(u))
    return sol, r, iter_counts


##############################
##############################
## define h_rate 

def func_p_3d(pp, m, n_el, geos_3d, str_bc):
    #multi =m # pp  # FEM;  multi= 1 # classical IgA
    kvs_u = tuple(bspline.make_knots(p, 0.0, 1.0, n, mult=m) for n in n_el)  # or : mult=2
    patches_u = [(kvs_u, g) for g in geos_3d]

    # Here we auto-detect the interfaces between the patches.
    MP_u = assemble.Multipatch(patches_u, automatch=True)
    
    # Multipatch objects for all variables (x, y, z)
    MP_block = multipatch_block_handler( [MP_u, MP_u, MP_u] ) 

    # set up Dirichlet boundary conditions
    bc = MP_block.compute_dirichlet_bcs([
        (1, 'right', g_zero)
    ])

    # define constant spline functions for integration
    kvs_j = tuple(bspline.make_knots(0, 0.0, 1.0, n, mult=multi) for n in n_el)  # constant basis vector for integration

    count = 0

    x_el = n_el[0]
    y_el = n_el[1]
    z_el = n_el[2]
    xgrid = np.linspace(0, 1, x_el)
    ygrid = np.linspace(0, 1, y_el)
    xygrid = (xgrid, ygrid)
    if str_bc =='robin':
        # assemble Robin-matrix
        AR = ass_Robin3d(MP_block, robin_data)
        u,r, it = func_Robin_3d(MP_block, AR)
    if str_bc =='dirichlet':
        u,r, it = func_Dirichlet_3d(MP_block)
        
   

    u1_funcs, u2_funcs, u3_funcs = split_u3d(u, MP_u, kvs_u, patches_u)
    for (u1_func, u2_func, u3_func, (kvs, geo)) in zip(u1_funcs, u2_funcs, u3_funcs, patches_u):  # u_funcs
        count += 1
        if count == 2: # displacement for 3rd patch
            print('\n patch:', cdount)
            # func_p(pp, geo, n_el, xygrid)
            G = geo.grid_eval(xygrid)

            dis1 = u1_func.grid_eval(xygrid)  # x-value
            dis2 = u2_func.grid_eval(xygrid)  # y-value
            return dis1, dis2, G, r, it
