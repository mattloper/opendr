#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['Rodrigues', 'VertNormals', 'TriNormals', 'TriNormalsScaled', 'CrossProduct', 'TriArea', 'AcosTriAngles', 'volume']

from .cvwrap import cv2
import numpy as np
import scipy.sparse as sp
from chumpy.utils import row, col
from chumpy import *
import chumpy as ch
from chumpy.ch import MatVecMult
from .topology import get_faces_per_edge, get_vert_connectivity


def volume(v, f):

    # Construct a 3D matrix which is of size (nfaces x 3 x 3)
    # Each row corresponds to a face; the third dimension indicates
    # which triangle in that face is being referred to
    vs = ch.dstack((
        v[f[:,0],:],
        v[f[:,1],:],
        v[f[:,2],:]))

    v321 = vs[:,0,2]*vs[:,1,1]*vs[:,2,0];
    v231 = vs[:,0,1]*vs[:,1,2]*vs[:,2,0];
    v312 = vs[:,0,2]*vs[:,1,0]*vs[:,2,1];
    v132 = vs[:,0,0]*vs[:,1,2]*vs[:,2,1];
    v213 = vs[:,0,1]*vs[:,1,0]*vs[:,2,2];
    v123 = vs[:,0,0]*vs[:,1,1]*vs[:,2,2];

    volumes =  (-v321 + v231 + v312 - v132 - v213 + v123) * (1./6.)
    return ch.abs(ch.sum(volumes))
    

class NormalizedNx3(Ch):
    dterms = 'v'

    def on_changed(self, which):
        if 'v' in which:
            self.ss = np.sum(self.v.r.reshape(-1,3)**2, axis=1)
            self.ss[self.ss==0] = 1e-10
            self.s = np.sqrt(self.ss)
            self.s_inv = 1. / self.s

    def compute_r(self):
        return (self.v.r.reshape(-1,3) / col(self.s)).reshape(self.v.r.shape)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.v:
            return None

        v = self.v.r.reshape(-1,3)
        blocks = -np.einsum('ij,ik->ijk', v, v) * (self.ss**(-3./2.)).reshape((-1,1,1))
        for i in range(3):
            blocks[:,i,i] += self.s_inv

        if True:
            data = blocks.ravel()
            indptr = np.arange(0,(self.v.r.size+1)*3,3)
            indices = col(np.arange(0,self.v.r.size))
            indices = np.hstack([indices, indices, indices])
            indices = indices.reshape((-1,3,3))
            indices = indices.transpose((0,2,1)).ravel()
            result = sp.csc_matrix((data, indices, indptr), shape=(self.v.r.size, self.v.r.size))
            return result
        else:
            matvec = lambda x : np.einsum('ijk,ik->ij', blocks, x.reshape((blocks.shape[0],3))).ravel()
            return sp.linalg.LinearOperator((self.v.r.size,self.v.r.size), matvec=matvec)


class Sum3xN(Ch):
    dterms = 'v'

    def compute_r(self):
        return np.sum(self.v.r.reshape((-1,3)), axis=1)

    def compute_dr_wrt(self, wrt):
        if wrt is self.v:
            IS = np.tile(col(np.arange(self.v.r.size/3)), (1, 3)).ravel()
            JS = np.arange(self.v.r.size)
            data = np.ones_like(JS)
            result = sp.csc_matrix((data, (IS, JS)), shape=(self.v.r.size/3, self.v.r.size))
            return result

class ndot(ch.Ch):
    dterms = 'mtx1', 'mtx2'

    def compute_r(self):
        return np.einsum('abc,acd->abd', self.mtx1.r, self.mtx2.r)
    
    def compute_d1(self):
        # To stay consistent with numpy, we must upgrade 1D arrays to 2D
        mtx1r = row(self.mtx1.r) if len(self.mtx1.r.shape)<2 else self.mtx1.r
        mtx2r = col(self.mtx2.r) if len(self.mtx2.r.shape)<2 else self.mtx2.r

        if mtx1r.ndim <= 2:
            return sp.kron(sp.eye(mtx1r.shape[0], mtx1r.shape[0]),mtx2r.T)
        else:
            mtx2f = mtx2r.reshape((-1, mtx2r.shape[-2], mtx2r.shape[-1]))
            mtx2f = np.rollaxis(mtx2f, -1, -2) #transpose basically            
            result = sp.block_diag([np.kron(np.eye(mtx1r.shape[-2], mtx1r.shape[-2]),m2) for m2 in mtx2f])
            assert(result.shape[0] == self.r.size)
            return result

    def compute_d2(self):
        
        # To stay consistent with numpy, we must upgrade 1D arrays to 2D
        mtx1r = row(self.mtx1.r) if len(self.mtx1.r.shape)<2 else self.mtx1.r
        mtx2r = col(self.mtx2.r) if len(self.mtx2.r.shape)<2 else self.mtx2.r

        if mtx2r.ndim <= 1:
            return self.mtx1r
        elif mtx2r.ndim <= 2:
            return sp.kron(mtx1r, sp.eye(mtx2r.shape[1],mtx2r.shape[1]))
        else:
            mtx1f = mtx1r.reshape((-1, mtx1r.shape[-2], mtx1r.shape[-1]))            
            result = sp.block_diag([np.kron(m1, np.eye(mtx2r.shape[-1],mtx2r.shape[-1])) for m1 in mtx1f])
            assert(result.shape[0] == self.r.size)
            return result
            
    
    def compute_dr_wrt(self, wrt):

        if wrt is self.mtx1 and wrt is self.mtx2:
            return self.compute_d1() + self.compute_d2()
        elif wrt is self.mtx1:
            return self.compute_d1()
        elif wrt is self.mtx2:
            return self.compute_d2()


def face_bases(v, f):
    t1 = TriEdges(f, 1, 0, v).reshape((-1,3))
    t2 = TriEdges(f, 2, 0, v).reshape((-1,3))
    #t3 = NormalizedNx3(CrossProduct(t1, t2)).reshape((-1,3))
    #t3 = CrossProduct(t1, t2).reshape((-1,3))
    
    # Problem: cross-product is proportional in length to len(t1)*len(t2)
    # Solution: divide by sqrt(sqrt(len(cross-product)))
    t3 = CrossProduct(t1, t2).reshape((-1,3)); t3 = t3 / col(ch.sum(t3**2., axis=1)**.25)
    result = ch.hstack((t1, t2, t3)).reshape((-1,3,3))
    return result


def edge_defs(v,f):
    fb = face_bases(v, f)
    fpe = get_faces_per_edge(v.r, f)
    return ndot(fb[fpe[:,0]], ch.linalg.inv(fb[fpe[:,1]]))
    

def FirstEdgesMtx(v, f, want_big=True):
    cnct = get_vert_connectivity((v.r if hasattr(v, 'r') else v), f)
    nbrs = [np.nonzero(np.array(cnct[:,i].todense()))[0][0] for i in range(cnct.shape[1])]
    JS = np.array(nbrs)
    IS = np.arange(len(JS))
    data = np.ones(IS.size)
    
    if want_big:
        IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
        JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
        data = np.concatenate((data, data, data))
        
    return sp.csc_matrix((data, (IS, JS)), shape=(JS.size, JS.size))
    


def SecondFundamentalForm(v, f):    
    from chumpy import hstack, vstack
    from chumpy.linalg import Pinv
    nbrs = MatVecMult(FirstEdgesMtx(v, f, want_big=True), v.ravel()).reshape((-1,3))
    
    b0 = VertNormals(f=f, v=v)
    b1 = NormalizedNx3(CrossProduct(b0, nbrs-v)).reshape((-1,3))
    b2 = NormalizedNx3(CrossProduct(b0, b1)).reshape((-1,3))
    
    cnct = get_vert_connectivity(np.asarray(v), f)
    ffs = []
    for i in range(v.size/3):
        nbrs = v[np.nonzero(np.asarray(cnct[i].todense()).ravel())[0]] - row(v[i])
        us = nbrs.dot(b2[i])
        vs = nbrs.dot(b1[i])
        hs = nbrs.dot(b0[i])
        coeffs = Pinv(hstack((col((us*.5)**2), col(us*vs), col((vs*.5)**2)))).dot(hs)
        ffs.append(row(coeffs))
        # if i == 3586:
        #     import pdb; pdb.set_trace()

    ffs = vstack(ffs)
    return ffs
        
        
def GaussianCurvature(v, f):
    ff = SecondFundamentalForm(v, f)
    result = ff[:,0] * ff[:,2] - ff[:,1]**2.
    return result


class Rodrigues(Ch):
    dterms = 'rt'
    
    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T
    
    
def TriArea(v_init, f, normalize):
    """ Returns a Ch object whose only attribute "v" represents the flattened vertices."""
    
    if normalize:
        nm = lambda x : NormalizedNx3(x)
    else:
        nm = lambda x : x
    result = Ch(lambda v : (Sum3xN(CrossProduct(TriEdges(f,1,0,nm(v)), TriEdges(f,2,0, nm(v)))**2.) ** 0.5) * 0.5)
    result.v = v_init
    return result


def AcosTriAngles(v, f, normalize):
    """ Returns a Ch object whose only attribute "v" represents the flattened vertices."""

    if normalize:
        nm = lambda x : NormalizedNx3(x)
    else:
        nm = lambda x : x

    return Ch(lambda v :
        Sum3xN(NormalizedNx3(TriEdges(f, 1, 0, nm(v))) * NormalizedNx3(TriEdges(f, 2, 0, nm(v)))) &
        Sum3xN(NormalizedNx3(TriEdges(f, 2, 1, nm(v))) * NormalizedNx3(TriEdges(f, 0, 1, nm(v)))) &
        Sum3xN(NormalizedNx3(TriEdges(f, 0, 2, nm(v))) * NormalizedNx3(TriEdges(f, 1, 2, nm(v)))))
        


class VertNormals(Ch):
    """If normalized==True, normals are normalized; otherwise they'll be about as long as neighboring edges."""
    
    dterms = 'v'
    terms = 'f', 'normalized'
    term_order = 'v', 'f', 'normalized'

    def on_changed(self, which):

        if not hasattr(self, 'normalized'):
            self.normalized = True
            
        if hasattr(self, 'v') and hasattr(self, 'f'):
            if 'f' not in which and hasattr(self, 'faces_by_vertex') and self.faces_by_vertex.shape[0]/3 == self.v.shape[0]:
                self.tns.v = self.v
            else: # change in f or in size of v. shouldn't happen often.
                f = self.f

                IS = f.ravel()
                JS = np.array([list(range(f.shape[0]))]*3).T.ravel()
                data = np.ones(len(JS))

                IS = np.concatenate((IS*3, IS*3+1, IS*3+2))
                JS = np.concatenate((JS*3, JS*3+1, JS*3+2))
                data = np.concatenate((data, data, data))

                sz = self.v.size
                self.faces_by_vertex = sp.csc_matrix((data, (IS, JS)), shape=(sz, f.size))

                self.tns = Ch(lambda v : CrossProduct(TriEdges(f,1,0,v), TriEdges(f,2,0, v)))
                self.tns.v = self.v
                
                if self.normalized:
                    tmp = MatVecMult(self.faces_by_vertex, self.tns)
                    self.normals = NormalizedNx3(tmp)
                else:                    
                    test = self.faces_by_vertex.dot(np.ones(self.faces_by_vertex.shape[1]))
                    faces_by_vertex = sp.diags([1. / test], [0]).dot(self.faces_by_vertex).tocsc()
                    normals = MatVecMult(faces_by_vertex, self.tns).reshape((-1,3))
                    normals = normals / (ch.sum(normals**2, axis=1) ** .25).reshape((-1,1))
                    self.normals = normals
                    
                    

    def compute_r(self):
        return self.normals.r.reshape((-1,3))
        
    def compute_dr_wrt(self, wrt):
        if wrt is self.v:
            return self.normals.dr_wrt(wrt)



def TriNormals(v, f):
    return NormalizedNx3(TriNormalsScaled(v,f))

def TriNormalsScaled(v, f):
    return CrossProduct(TriEdges(f,1,0,v), TriEdges(f,2,0, v))



class TriEdges(Ch):
    terms = 'f', 'cplus', 'cminus'
    dterms = 'v'

    def compute_r(self):
        cplus = self.cplus
        cminus = self.cminus
        return _edges_for(self.v.r, self.f, cplus, cminus)

    def compute_dr_wrt(self, wrt):
        if wrt is not self.v:
            return None
            
        cplus = self.cplus
        cminus = self.cminus
        vplus  = self.f[:,cplus]
        vminus = self.f[:,cminus]
        vplus3 = row(np.hstack([col(vplus*3), col(vplus*3+1), col(vplus*3+2)]))
        vminus3 = row(np.hstack([col(vminus*3), col(vminus*3+1), col(vminus*3+2)]))

        IS = row(np.arange(0,vplus3.size))
        ones = np.ones(vplus3.size)
        shape = (self.f.size, self.v.r.size)
        return sp.csc_matrix((ones, np.vstack([IS, vplus3])), shape=shape) - sp.csc_matrix((ones, np.vstack([IS, vminus3])), shape=shape)

def _edges_for(v, f, cplus, cminus):
    return (
        v.reshape(-1,3)[f[:,cplus],:] - 
        v.reshape(-1,3)[f[:,cminus],:]).ravel()
        
class CrossProduct(Ch):
    terms = []
    dterms = 'a', 'b'
    
    def on_changed(self, which):
        if 'a' in which:
            a = self.a.r.reshape((-1,3))
            self.a1 = a[:,0]
            self.a2 = a[:,1]
            self.a3 = a[:,2]
            
        if 'b' in which:
            b = self.b.r.reshape((-1,3))            
            self.b1 = b[:,0]
            self.b2 = b[:,1]
            self.b3 = b[:,2]        

    def compute_r(self):

        # TODO: maybe use cross directly? is it faster?
        # TODO: check fortran ordering?
        return _call_einsum_matvec(self.Ax, self.b.r)

    def compute_dr_wrt(self, obj):
        if obj not in (self.a, self.b):
            return None
            
        sz = self.a.r.size
        if not hasattr(self, 'indices') or self.indices.size != sz*3:
            self.indptr = np.arange(0,(sz+1)*3,3)
            idxs = col(np.arange(0,sz))
            idxs = np.hstack([idxs, idxs, idxs])
            idxs = idxs.reshape((-1,3,3))
            idxs = idxs.transpose((0,2,1)).ravel()
            self.indices = idxs

        if obj is self.a:
            # m = self.Bx
            # matvec = lambda x : _call_einsum_matvec(m, x)
            # matmat = lambda x : _call_einsum_matmat(m, x)
            # return sp.linalg.LinearOperator((self.a1.size*3, self.a1.size*3), matvec=matvec, matmat=matmat)
            data = self.Bx.ravel()
            result = sp.csc_matrix((data, self.indices, self.indptr), shape=(sz, sz))
            return -result


        elif obj is self.b:
            # m = self.Ax
            # matvec = lambda x : _call_einsum_matvec(m, x)
            # matmat = lambda x : _call_einsum_matmat(m, x)
            # return sp.linalg.LinearOperator((self.a1.size*3, self.a1.size*3), matvec=matvec, matmat=matmat)
            data = self.Ax.ravel()
            result = sp.csc_matrix((data, self.indices, self.indptr), shape=(sz, sz))
            return -result


    @depends_on('a')
    def Ax(self):
        """Compute a stack of skew-symmetric matrices which can be multiplied
        by 'b' to get the cross product. See:

        http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
        """
        #  0         -self.a3   self.a2
        #  self.a3    0        -self.a1
        # -self.a2    self.a1   0        
        m = np.zeros((len(self.a1), 3, 3))
        m[:, 0, 1] = -self.a3
        m[:, 0, 2] = +self.a2
        m[:, 1, 0] = +self.a3
        m[:, 1, 2] = -self.a1
        m[:, 2, 0] = -self.a2
        m[:, 2, 1] = +self.a1        
        return m

    @depends_on('b')
    def Bx(self):        
        """Compute a stack of skew-symmetric matrices which can be multiplied
        by 'a' to get the cross product. See:

        http://en.wikipedia.org/wiki/Cross_product#Conversion_to_matrix_multiplication
        """
        #  0         self.b3  -self.b2
        # -self.b3   0         self.b1
        #  self.b2  -self.b1   0
        
        
        m = np.zeros((len(self.b1), 3, 3))
        m[:, 0, 1] = +self.b3
        m[:, 0, 2] = -self.b2
        m[:, 1, 0] = -self.b3
        m[:, 1, 2] = +self.b1
        m[:, 2, 0] = +self.b2
        m[:, 2, 1] = -self.b1
        return m


def _call_einsum_matvec(m, righthand):
    r = righthand.reshape(m.shape[0],3)
    return np.einsum('ijk,ik->ij', m, r).ravel()

def _call_einsum_matmat(m, righthand):
    r = righthand.reshape(m.shape[0],3,-1)
    return np.einsum('ijk,ikm->ijm', m, r).reshape(-1,r.shape[2])        

def main():
    pass
    

if __name__ == '__main__':
    main()

