#!/usr/bin/env python
# encoding: utf-8

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['LambertianPointLight', 'SphericalHarmonics']

import os, sys, logging
import numpy as np
import scipy.sparse as sp
import scipy

from chumpy.utils import row, col
import chumpy as ch
from chumpy.ch import Ch
from .geometry import VertNormals
from chumpy import multiply, maximum


logger = logging.getLogger(__name__)

def real_sh_coeff(xyz_samples):
    d_sqrt_pi = 2*np.sqrt(np.pi)
    real_coeff = np.zeros((len(xyz_samples), 9))
    real_coeff[:,0] = 1/d_sqrt_pi

    real_coeff[:,1] = (np.sqrt(3)/d_sqrt_pi)*xyz_samples[:,0]
    real_coeff[:,2] = (np.sqrt(3)/d_sqrt_pi)*xyz_samples[:,2]
    real_coeff[:,3] = (np.sqrt(3)/d_sqrt_pi)*xyz_samples[:,1]

    real_coeff[:,4] = (np.sqrt(15)/d_sqrt_pi)*xyz_samples[:,0]*xyz_samples[:,1]
    real_coeff[:,5] = (np.sqrt(15)/d_sqrt_pi)*xyz_samples[:,1]*xyz_samples[:,2]
    real_coeff[:,6] = (np.sqrt(5)/(2*d_sqrt_pi))*(3*xyz_samples[:,2]**2-1)
    real_coeff[:,7] = (np.sqrt(15)/d_sqrt_pi)*xyz_samples[:,0]*xyz_samples[:,2]
    real_coeff[:,8] = (np.sqrt(15)/(2*d_sqrt_pi))*(xyz_samples[:,0]**2 - xyz_samples[:,1]**2)
    return real_coeff	



class SphericalHarmonics(Ch):
    dterms = 'vn', 'components'
    terms = ['light_color']

    d_sqrt_pi = 2*np.sqrt(np.pi)
    K = np.array([
        1./d_sqrt_pi,
        np.sqrt(3)/d_sqrt_pi,        
	np.sqrt(3)/d_sqrt_pi,
	np.sqrt(3)/d_sqrt_pi,
	np.sqrt(15)/d_sqrt_pi,
	np.sqrt(15)/d_sqrt_pi,
        np.sqrt(5)/(2*d_sqrt_pi),
	np.sqrt(15)/d_sqrt_pi,
	np.sqrt(15)/(2*d_sqrt_pi)])

    @property
    def num_channels(self):
        return self.light_color.size

    def on_changed(self, which):
        if 'vn' in which:
            vn = self.vn.r.reshape((-1,3))
            
            # Conversion from normals to spherical harmonics found in...
            # http://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates
            self.theta = np.arccos(vn[:,2])
            self.phi = np.arctan2(vn[:,1], vn[:,0])
            
            self.sh_coeffs = real_sh_coeff(vn)            
            self.num_verts = self.sh_coeffs.shape[0]

        if 'light_color' in which or self.mtx.shape[1] != self.num_verts:
            nc = self.num_channels
            IS = np.arange(self.num_verts*nc)
            JS = np.repeat(np.arange(self.num_verts), nc)
            data = (row(self.light_color)*np.ones((self.num_verts, nc))).ravel()
            self.mtx = sp.csc_matrix((data, (IS,JS)), shape=(self.num_verts*nc, self.num_verts))
            
    
    def compute_r(self):
        comps = self.components.r
        n = len(comps)
        result = self.mtx.dot(self.sh_coeffs[:,:n].dot(col(self.components.r)))
        result[result<0] = 0
        return result.reshape((-1,self.num_channels))
    
    def compute_dr_wrt(self, wrt):
        comps = np.zeros(9)
        comps[:len(self.components.r)] = self.components.r
        comps = comps * self.K.ravel()
        if wrt is self.vn:
            vn = self.vn.r.reshape((-1,3))

            #real_coeff[:,1] = (np.sqrt(3)/d_sqrt_pi)*xyz_samples[:,0]
            VS0 = np.ones(self.sh_coeffs.shape[0]) * comps[1]
            #real_coeff[:,2] = (np.sqrt(3)/d_sqrt_pi)*xyz_samples[:,2]
            VS1 = np.ones(self.sh_coeffs.shape[0]) * comps[3]
            #real_coeff[:,3] = (np.sqrt(3)/d_sqrt_pi)*xyz_samples[:,1]
            VS2 = np.ones(self.sh_coeffs.shape[0]) * comps[2]

            #real_coeff[:,4] = (np.sqrt(15)/d_sqrt_pi)*xyz_samples[:,0]*xyz_samples[:,1]
            VS0 += vn[:,1] * comps[4]
            VS1 += vn[:,0] * comps[4]
            
            #real_coeff[:,5] = (np.sqrt(15)/d_sqrt_pi)*xyz_samples[:,1]*xyz_samples[:,2]
            VS1 += vn[:,2]*comps[5]
            VS2 += vn[:,1]*comps[5]
            
            #real_coeff[:,6] = (np.sqrt(5)/2*d_sqrt_pi)*(3*xyz_samples[:,2]**2-1)
            VS2 += 6*vn[:,2] * comps[6]
            
            #real_coeff[:,7] = (np.sqrt(15)/d_sqrt_pi)*xyz_samples[:,0]*xyz_samples[:,2]
            VS0 += vn[:,2] * comps[7]
            VS2 += vn[:,0] * comps[7]

            #real_coeff[:,8] = (np.sqrt(15)/(2*d_sqrt_pi))*(xyz_samples[:,0]**2 - xyz_samples[:,1]**2)
            VS0 += 2. * vn[:,0] * comps[8]
            VS1 -= 2. * vn[:,1] * comps[8]
            
            rng = np.arange(self.sh_coeffs.shape[0])
            IS = np.concatenate((rng, rng, rng))
            JS = np.concatenate((rng*3, rng*3+1, rng*3+2))
            data = np.concatenate((VS0, VS1, VS2))
            result = self.mtx.dot(sp.csc_matrix((data, (IS, JS))))
            
        elif wrt is self.components:
            comps = self.components.r
            n = len(comps)            
            result = self.mtx.dot(self.sh_coeffs[:,:n])
        else:
            return None
            
        which = np.nonzero(self.r.ravel()>0)[0]
        data = np.ones_like(which)
        gr_equal_zero = sp.csc_matrix((data, (which, which)), shape=(self.r.size, self.r.size))

        return gr_equal_zero.dot(result)
    

def lambertian_spotlight(v, vn, pos, dir, spot_exponent, camcoord=False, camera_t=None, camera_rt=None):
    """
    :param v: vertices
    :param vn: vertex normals
    :param light_pos: light position
    :param light_dir: light direction
    :param spot_exponent: spot exponent (a la opengl)
    :param camcoord: if True, then pos and dir are wrt the camera
    :param camera_t: 3-vector indicating translation of camera
    :param camera_rt: 3-vector indicating direction of camera
    :return: Vx1 array of brightness
    """

    if camcoord: # Transform pos and dir from camera to world coordinate system
        assert(camera_t is not None and camera_rt is not None)
        from opendr.geometry import Rodrigues
        rot = Rodrigues(rt=camera_rt)
        pos = rot.T.dot(pos-camera_t)
        dir = rot.T.dot(dir)

    dir = dir / ch.sqrt(ch.sum(dir**2.))
    v_minus_light = v - pos.reshape((1,3))
    v_distances = ch.sqrt(ch.sum(v_minus_light**2, axis=1))
    v_minus_light_normed = v_minus_light / v_distances.reshape((-1,1))
    cosangle = v_minus_light_normed.dot(dir.reshape((3,1)))
    light_dot_normal = ch.sum(vn*v_minus_light_normed, axis=1)
    light_dot_normal.label = 'light_dot_normal'
    cosangle.label = 'cosangle'
    result = light_dot_normal.ravel() * cosangle.ravel()**spot_exponent
    result = result / v_distances ** 2.
    result = maximum(result, 0.0)

    return result





class LambertianPointLight(Ch):
    terms = 'f', 'num_verts', 'light_color'
    dterms = 'light_pos', 'v', 'vc', 'vn'
    
    def on_changed(self, which):
        if not hasattr(self, '_lpl'):
            self.add_dterm('_lpl', maximum(multiply(a=multiply()), 0.0))
        if not hasattr(self, 'ldn'):
            self.ldn = LightDotNormal(self.v.r.size/3)            
        if not hasattr(self, 'vn'):
            logger.info('LambertianPointLight using auto-normals. This will be slow for derivative-free computations.')
            self.vn = VertNormals(f=self.f, v=self.v)
            self.vn.needs_autoupdate = True
        if 'v' in which and hasattr(self.vn, 'needs_autoupdate') and self.vn.needs_autoupdate:
            self.vn.v = self.v
        
        ldn_args = {k: getattr(self, k) for k in which if k in ('light_pos', 'v', 'vn')}
        if len(ldn_args) > 0:
            self.ldn.set(**ldn_args)
            self._lpl.a.a.a = self.ldn.reshape((-1,1))

        if 'num_verts' in which or 'light_color' in which:
            # nc = self.num_channels
            # IS = np.arange(self.num_verts*nc)
            # JS = np.repeat(np.arange(self.num_verts), 3)
            # data = (row(self.light_color)*np.ones((self.num_verts, 3))).ravel()
            # mtx = sp.csc_matrix((data, (IS,JS)), shape=(self.num_verts*3, self.num_verts))
            self._lpl.a.a.b = self.light_color.reshape((1,self.num_channels))

        if 'vc' in which:
            self._lpl.a.b = self.vc.reshape((-1,self.num_channels))

    @property
    def num_channels(self):
        return self.light_color.size
        
    def compute_r(self):
        return self._lpl.r
        
    def compute_dr_wrt(self, wrt):
        if wrt is self._lpl:
            return 1



# def compute_light_repeat(num_verts):
#     IS = np.arange(num_verts*3)
#     JS = IS % 3
#     data = np.ones_like(IS, dtype=np.float64)
#     ij = np.vstack((row(IS), row(JS)))
#     return sp.csc_matrix((data, ij), shape=(num_verts*3, 3))

def LightDotNormal(num_verts):

    normalize_rows = lambda v : v / col(ch.sqrt(ch.sum(v.reshape((-1,3))**2, axis=1)))
    sum_rows = lambda v :  ch.sum(v.reshape((-1,3)), axis=1)

    return Ch(lambda light_pos, v, vn :
        sum_rows(normalize_rows(light_pos.reshape((1,3)) - v.reshape((-1,3))) * vn.reshape((-1,3))))



def main():
    pass


if __name__ == '__main__':
    main()

