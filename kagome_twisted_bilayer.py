import numpy as np
from pymatgen.core.structure import Structure as pmg_struct
from scipy.linalg.lapack import zheev
from tBG.fortran.spec_func import get_pk
from tBG.utils import *
from tBG.hopping import filter_neig_list
import copy

def pmg_sublatts_twisted_bilayer_hexagonal_lattice(m, n, a, h, sublatts_frac):
    """
    this function is to generate the twisted bilayer hexgonal lattice: such as tisted bilayer graphene or kagome lattice
        depending on the inputed sublatts_frac, return pymategn structure
    a: the lattice constant
    h: the interlayer distance
    m, n: integrs for the description of twisted hexagonal lattice
    sublatts_frac: the frac coordinates for monolayer
    """
    def rotate_mat_mn(m,n):
        cos = (n**2+m**2+4*m*n)/(2*(n**2+m**2+m*n))
        sin = np.sqrt(3)*(m**2-n**2)/(2*(n**2+m**2+m*n))
        return np.array([[cos, -sin, 0],\
                         [sin,  cos, 0],
                         [0,     0,  1]])
    #latt_vec_bott = self.a*np.array([[1., 0., 0.],[0.5, np.sqrt(3)/2, 0.], [0, 0, 100/self.a]])
    latt_vec_bott = a*np.array([[np.sqrt(3)/2, -1/2., 0.],[np.sqrt(3)/2, 1/2., 0.], [0, 0, 100/a]])
    latt_vec_top = np.matmul(rotate_mat_mn(m,n), latt_vec_bott.T).T
    sublatts = [[],[]]
    for sub_latt in sublatts_frac:
        site_bott = np.append(sub_latt,[0], axis=0)
        latt_bott = pmg_struct(latt_vec_bott, ['C'], [site_bott])
        latt_bott.make_supercell([[m+n,-n,0],[n, m, 0],[0,0,1]])
        sublatts[0].append(latt_bott)

        site_top = np.append(sub_latt,[h/100], axis=0)
        latt_top = pmg_struct(latt_vec_top, ['C'], [site_top])
        latt_top.make_supercell([[m+n, -m, 0],[m, n,0],[0,0,1]])
        sublatts[1].append(latt_top)
    return sublatts, latt_vec_bott, latt_vec_top

class _PBCMethods:
    """
    methods for twisted bilayer kagome lattice
    """
    def pymatgen_struct(self):
        return pmg_struct(self.latt_vec, ['C']*len(self.coords), self.coords, coords_are_cartesian=True)

    def hamilt_cell_diff(self, k, elec_field=0.0):
        Hk = np.zeros((self.nsite, self.nsite),dtype=complex)
        if elec_field:
            np.fill_diagonal(Hk,self.coords[:,-1]*elec_field)
        latt_vec = self.latt_vec[0:2][:,0:2]
        for i in range(self.nsite):
            for m,n,j in self.hoppings[i]:
                R = m*latt_vec[0]+n*latt_vec[1]
                t = self.hoppings[i][(m,n,j)]
                phase = np.exp(1j*np.dot(k, R))
                Hk[i,j] = Hk[i,j] + t*phase
                Hk[j,i] = Hk[j,i] + t*np.conj(phase)
        return Hk

    def hamilt_pos_diff(self, k, elec_field=0.0):
        if len(k) == 2:
            k = np.array([k[0],k[1],0.])
        Hk = np.zeros((self.nsite, self.nsite),dtype=complex)
        if elec_field:
            np.fill_diagonal(Hk,self.coords[:,-1]*elec_field)
        latt_vec = self.latt_vec[0:2]
        for i in range(self.nsite):
            ri = self.coords[i] 
            for m,n,j in self.hoppings[i]:
                R = m*latt_vec[0]+n*latt_vec[1]
                rj = R + self.coords[j]
                t = self.hoppings[i][(m,n,j)]
                phase = np.exp(1j*np.dot(k, rj-ri))
                Hk[i,j] = Hk[i,j] + t*phase
                Hk[j,i] = Hk[j,i] + t*np.conj(phase)
        return Hk

    def diag_kpts(self, kpts, vec=0, pmk=0, elec_field=0.):
        """
        kpts: the coordinates of kpoints
        vec: whether to calculate the eigen vectors
        pmk: whether to calculate PMK for effective band structure
        elec_field: the electric field perpendicular to graphane plane
        fname: the file saveing results
        """
        val_out = []
        vec_out = []
        pmk_out = []
        for k in kpts:
            #Hk = self.hamilt_cell_diff(k, elec_field)
            Hk = self.hamilt_pos_diff(k, elec_field)
            vals, vecs, info = zheev(Hk, vec)
            if info:
                raise ValueError('zheev failed')
            if pmk:
                Pk = get_pk(k, np.array(self.layer_nsites)/2, [1,1], 2, 2, vecs, self.coords, self.species())
                pmk_out.append(Pk)
            val_out.append(vals)
            vec_out.append(vecs)
        return np.array(val_out), np.array(vec_out), np.array(pmk_out)

def get_hop_func_intra(t, beta, bond_length):
    def hop_func(r):
        return -t*np.exp(-beta*(r/bond_length-1))
    return hop_func

def get_hop_func_inter(tz, beta, interlayer_dist):
    def hop_func(r):
        return tz*np.exp(-beta*(r/interlayer_dist-1))
    return hop_func

def get_neighbors(pmg_struct, dist_cut, layer_nsites):
    neig_list = pmg_struct.get_neighbor_list(dist_cut)
    p0, p1, offset, dist = filter_neig_list(neig_list)
    nsite = np.sum(layer_nsites)
    neigh_intra_pair = [[] for _ in range(nsite)]
    neigh_intra_dist = [[] for _ in range(nsite)]
    def add_neigs_intra(layer):
        if layer=='bottom':
            ind0 = np.where(p0<=layer_nsites[0]-1)[0]
            ind1 = np.where(p1<=layer_nsites[0]-1)[0]
        elif layer=='top':
            ind0 = np.where(p0>=layer_nsites[0])[0]
            ind1 = np.where(p1>=layer_nsites[0])[0]
        ind = np.intersect1d(ind0, ind1)
        p0i = p0[ind]
        p1i = p1[ind]
        offseti = np.array(offset[ind][:,0:2], dtype=int)
        p1i_extend = np.append(offseti, p1i.reshape(-1,1), axis=1)
        disti = dist[ind]
        for i in range(len(ind)):
            neigh_intra_pair[p0i[i]].append(tuple(p1i_extend[i]))
            neigh_intra_dist[p0i[i]].append(disti[i])
    add_neigs_intra('bottom')
    add_neigs_intra('top')

    neigh_inter_pair = [[] for _ in range(layer_nsites[0])]
    neigh_inter_dist = [[] for _ in range(layer_nsites[0])]
    def add_neigs_inter():
        ind0 = np.where(p0<=layer_nsites[0]-1)[0]
        ind1 = np.where(p1>=layer_nsites[0])[0]
        ind = np.intersect1d(ind0, ind1)
        p0i = p0[ind]
        p1i = p1[ind]
        offseti = np.array(offset[ind][:,0:2], dtype=int)
        p1i_extend = np.append(offseti, p1i.reshape(-1,1), axis=1)
        disti = dist[ind]
        for i in range(len(ind)):
            neigh_inter_pair[p0i[i]].append(tuple(p1i_extend[i]))
            neigh_inter_dist[p0i[i]].append(disti[i])
    add_neigs_inter()
    return neigh_intra_pair, neigh_intra_dist, neigh_inter_pair, neigh_inter_dist

class _KTB_Methods:
    """
    methods for twisted bilayer kagome lattice
    """

    def add_hopping(self, t=1, tz=0.3, beta=20, dist_cut=20.):
        """
        prb 100 155421 (2019)
        """
        hop_func_intra = get_hop_func_intra(t, beta, self.a/2)
        hop_func_inter = get_hop_func_inter(tz, beta, self.h)
        pmg_st = self.pymatgen_struct()
        neigh_intra_pair, neigh_intra_dist, neigh_inter_pair, neigh_inter_dist = \
                   get_neighbors(pmg_st, dist_cut, self.layer_nsites)

        neigh_intra_hop = [[] for _ in range(len(neigh_intra_pair))]
        for i in range(len(neigh_intra_hop)):
            neigh_intra_hop[i] = hop_func_intra(np.array(neigh_intra_dist[i]))

        neigh_inter_hop = [[] for _ in range(len(neigh_inter_pair))]
        for i in range(len(neigh_inter_hop)):
            neigh_inter_hop[i] = hop_func_inter(np.array(neigh_inter_dist[i]))

        ## put value for hoplist ##
        hop_list_intra = [dict(zip(neigh_intra_pair[i], neigh_intra_hop[i])) for i in range(len(neigh_intra_pair))]
        hop_list_inter = [dict(zip(neigh_inter_pair[i], neigh_inter_hop[i])) for i in range(len(neigh_inter_pair))]
        hop_list = hop_list_intra
        for i in range(len(hop_list_inter)):
            hop_list[i].update(hop_list_inter[i])
        #tmp = [hop_list_intra[i].update(hop_list_inter[i]) for i in range(len(hop_list_inter))]
        self.hoppings = hop_list

class Kagome(_KTB_Methods, _PBCMethods):
    def __init__(self, a=14.86, h=3.64):
        self.a = a
        self.h = h
        self._make_structure()

    def _make_structure(self):
        sublatts_frac = np.array([[1/2.,0, 0],[1/2.,1/2.,0],[0,1/2.,0]])
        self.latt_vec = self.a*np.array([[np.sqrt(3)/2, -1/2., 0.],[np.sqrt(3)/2, 1/2., 0.], [0, 0, 100/self.a]])
        self.coords = frac2cart(sublatts_frac, self.latt_vec)
        self.layer_nsites = [len(sublatts_frac)]
        self.nsite = len(self.coords)

    def to_bilayer(self, stack='AA'):
        coords = copy.deepcopy(self.coords)
        if stack=='AA':
            coords1 = copy.deepcopy(coords)
            coords1[:,-1] = self.h
            self.coords = np.concatenate([coords, coords1], axis=0)
            self.nsite = self.nsite + len(coords1)
            self.layer_nsites.append(len(coords1))    
        elif stack=='AB':
            coords1 = coords + (self.latt_vec[0]+self.latt_vec[1])*1/3.
            coords1[:,-1] = self.h
            self.coords = np.concatenate([coords, coords1], axis=0)
            self.nsite = self.nsite + len(coords1)
            self.layer_nsites.append(len(coords1))    


class KagomeTwistedBilayer(_KTB_Methods):
    """
    prb 100 155421 (2019)
    Kagome twisted bilayer
    """
    def __init__(self, a=14.86, h=3.64, rotate_cent='hole'):
        self.a = 1
        self.h = 1
        if rotate_cent=='hole':
            self.sublatts_frac = np.array([[1/2.,0],[1/2.,1/2.],[0,1/2.]])

    def make_structure(self, m, n):
        pmg_sublatts = pmg_sublatts_twisted_bilayer_hexagonal_lattice(m, n, self.a, self.h, self.sublatts_frac)
        #return pmg_sublatts
        self.layer_nsites= [0,0]
        for i in range(len(self.sublatts_frac)):
            latt_bott = pmg_sublatts[0][i]
            self.layer_nsites[0] += latt_bott.num_sites
            latt_top = pmg_sublatts[1][i]
            self.layer_nsites[1] += latt_top.num_sites
        self.layer_nsites_sublatt = [[latt.num_sites for latt in pmg_sublatts[0]],[latt.num_sites for latt in pmg_sublatts[1]]]
        self.latt_vec = pmg_sublatts[0][0].lattice.matrix
        coords_bott = np.concatenate([latt.cart_coords for latt in pmg_sublatts[0]], axis=0)
        coords_top = np.concatenate([latt.cart_coords for latt in pmg_sublatts[1]], axis=0)
        self.coords = np.concatenate([coords_bott, coords_top], axis=0)
        self.nsite = len(self.coords)
 
