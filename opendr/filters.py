#!/usr/bin/env python

"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__all__ = ['gaussian_pyramid', 'laplacian_pyramid', 'GaussPyrDownOne']

from .cvwrap import cv2
import chumpy as ch
import numpy as np
from copy import deepcopy
import scipy.sparse as sp
from chumpy.ch import MatVecMult, Ch, depends_on
from functools import reduce

    

def laplacian_pyramid(input_objective, imshape, normalization, n_levels, as_list):

    if normalization is None:
        norm2 = lambda x : x
    elif normalization is 'SSE':
        norm2 = lambda x : x / np.sqrt(np.sum(x.r**2.))
    elif normalization is 'size':
        norm2 = lambda x : x / x.r.size
    else:
        norm2 = normalization

    
    output_objs = []
    for level in range(n_levels):
    
        blur_mtx = filter_for(imshape[0], imshape[1], imshape[2] if len(imshape)>2 else 1, kernel = GaussianKernel2D(3, 1))
        blurred = MatVecMult(blur_mtx, input_objective).reshape(imshape)
        output_objs.append(norm2(input_objective - blurred))


        halfsampler_mtx, imshape = halfsampler_for(imshape)
        input_objective = MatVecMult(halfsampler_mtx, blurred.ravel()).reshape(imshape)
        
    output_objs.append(norm2(input_objective).reshape(imshape)) 
        
    return output_objs if as_list else reduce(lambda x, y : ch.concatenate((x.ravel(), y.ravel())), output_objs)



def gaussian_pyramid(input_objective, imshape=None, normalization='SSE', n_levels=3, as_list=False, label=None):
    
    if imshape is None:
        imshape = input_objective.shape

    if normalization is None:
        norm2 = lambda x : x
    elif normalization is 'SSE':
        norm2 = lambda x : x / np.sqrt(np.sum(x.r**2.))
    elif normalization is 'size':
        norm2 = lambda x : x / x.r.size
    else:
        norm2 = normalization

    cur_imshape = deepcopy(imshape)
    cur_obj = input_objective

    input_objective = norm2(input_objective)
    output_objectives = [input_objective]

    for ik in range(n_levels):    
        cur_obj = GaussPyrDownOne(px=cur_obj, im_shape = cur_imshape)
        cur_imshape = cur_obj.output_shape
        output_objectives.append(norm2(cur_obj) if label is None else norm2(cur_obj) >> '%s%d' % (label,ik))
        
    if not as_list:
        andit = lambda a : reduce(lambda x, y : ch.concatenate((x.ravel(), y.ravel())), a)
        output_objectives = andit(output_objectives)

    return output_objectives

def GaussianKernel2D(ksize, sigma):
    if ksize % 2 != 1:
        raise Exception('ksize should be an odd number')
    if sigma <= 0:
        raise Exception('sigma should be positive')
    oneway = np.tile(cv2.getGaussianKernel(ksize,sigma), (1, ksize))
    return oneway * oneway.T


class GaussPyrDownOneOld(Ch):
    terms =  'im_shape', 'want_downsampling', 'kernel'
    dterms = 'px'
    
    # Approximation to a 3x3 Gaussian kernel
    default_kernel = GaussianKernel2D(3, 1)

    def on_changed(self, which):
        if not hasattr(self, 'kernel'):
            self.kernel = self.default_kernel.copy()
            
        if 'im_shape' in which:
            sh = self.im_shape
            self.transf_mtx, self._output_shape = filter_for_nopadding(sh, self.kernel)
            if True: # self.want_downsampling: <-- setting want_downsampling to "False" is broken
                halfsampler, self._output_shape = halfsampler_for(self._output_shape)
                self.transf_mtx = halfsampler.dot(self.transf_mtx)
            self.add_dterm('transform', MatVecMult(mtx=self.transf_mtx))

        if 'px' in which:
            self.transform.vec = self.px
    
    
    @property
    def output_shape(self):
        self._call_on_changed() # trigger changes from im_shape
        return self._output_shape
    
    def compute_r(self):
        result = self.transf_mtx.dot(self.px.r.ravel()).reshape(self.output_shape)
        #print result.shape
        return result
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.transform:
            return 1
        
    

class GaussPyrDownOneNew(Ch):
    terms =  'im_shape'
    dterms = 'px'
    
    @property
    def output_shape(self):
        return self.r.shape
    
    def compute_r(self):
        result = cv2.pyrDown(self.px.r)
        return result
    
    def compute_dr_wrt(self, wrt):
        if wrt is self.px:
            linop = lambda x : cv2.pyrDown(x.reshape(self.im_shape)).ravel()
            return sp.linalg.LinearOperator((self.r.size, self.px.size), linop)


GaussPyrDownOne = GaussPyrDownOneOld

def halfsampler_for(shape):
    h = shape[0]
    w = shape[1]
    d = shape[2] if len(shape) > 2 else 1
    
    JS = np.arange(h*w*d).reshape((h,w,d))
    JS = JS[::2,::2,:]
    JS = JS.flatten()
    
    IS = np.arange(len(JS))
    data = np.ones(len(JS))
    
    if len(shape) > 2:
        shape = (int(np.ceil(h/2.)), int(np.ceil(w/2.)), int(d))
    else:
        shape = (int(np.ceil(h/2.)), int(np.ceil(w/2.)))
    return sp.csc_matrix((data, (IS, JS)), shape=(len(IS), h*w*d)), shape
    

def filter_for_nopadding(shape, kernel):
    new_shape = np.array(shape).copy()
    new_shape[0] -= kernel.shape[0] - 1
    new_shape[1] -= kernel.shape[1] - 1

    IS = []
    JS = []
    data = []

    new_h = int(new_shape[0])
    new_w = int(new_shape[1])
    old_h = int(shape[0])
    old_w = int(shape[1])

    xs = np.tile(np.arange(old_w), (old_h, 1))
    ys = np.tile(np.arange(old_h), (old_w, 1)).T.copy()

    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):

            xs2 = xs[ky:new_h+ky, kx:new_w+kx]
            ys2 = ys[ky:new_h+ky, kx:new_w+kx]

            JS.append(xs2.ravel() + ys2.ravel()*old_w)
            IS.append(np.arange(new_shape[0]*new_shape[1]))
            data.append(np.ones(IS[-1].size) * kernel[ky, kx])
    

    IS = np.concatenate(IS)
    JS = np.concatenate(JS)
    data = np.concatenate(data)

    if len(shape) > 2:
        d = int(shape[2])
        if d > 1:
            IS = [IS*d+k for k in range(d)]
            JS = [JS*d+k for k in range(d)]
            data = [data for k in range(d)]
            IS = np.concatenate(IS)
            JS = np.concatenate(JS)
            data = np.concatenate(data)
            
    return sp.csc_matrix((data, (IS, JS))), new_shape


def filter_for(h, w, d, kernel):
    kernel = np.atleast_3d(kernel)   
    if kernel.shape[2] != d:
        kernel = np.tile(kernel, (1, 1, d))
   
    kxm = (kernel.shape[1]+1) / 2
    kym = (kernel.shape[0]+1) / 2
    
    
    xs = np.tile(np.arange(w), (h, 1))
    ys = np.tile(np.arange(h), (w, 1)).T.copy()

    IS = []
    JS = []
    data = []
    for ky in range(kernel.shape[0]):
        for kx in range(kernel.shape[1]):
            for channel in range(d):            
                cky = ky - kym
                ckx = kx - kxm
    
                xs2 = np.clip(xs + ckx, 0, w-1)
                ys2 = np.clip(ys + cky, 0, h-1)
                
                IS.append(np.arange(w*h)*d+channel)
                JS.append((xs2.ravel() + ys2.ravel()*w)*d+channel)
                data.append(np.ones(IS[-1].size) * kernel[ky, kx, channel])
    
    IS = np.concatenate(IS)
    JS = np.concatenate(JS)
    data = np.concatenate(data)
    
    #if d > 1:
    #    IS = [IS*d+k for k in range(d)]
    #    JS = [JS*d+k for k in range(d)]
    #    data = [data for k in range(d)]
    #    IS = np.concatenate(IS)
    #    JS = np.concatenate(JS)
    #    data = np.concatenate(data)
    #    
            
    return sp.csc_matrix((data, (IS, JS)), shape=(h*w*d, h*w*d))
            
            
            

def main():
    pass



    
if __name__ == '__main__':
    main()
