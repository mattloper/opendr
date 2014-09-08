__author__ = 'matt'

from copy import deepcopy
from os.path import join, split, exists
import numpy as np
from cvwrap import cv2

from utils import wget

def download_earthmesh():
    import os, sys
    def wg(url):
        dest = join(os.path.dirname(__file__), split(url)[1])
        if not exists(dest):
            sys.stderr.write('Downloading %s...\n' % (url))
            wget(url, dest)
            sys.stderr.write('Downloading %s...done.\n' % (url))
    wg('http://files.is.tue.mpg.de/mloper/opendr/images/nasa_earth.obj')
    wg('http://files.is.tue.mpg.de/mloper/opendr/images/nasa_earth.mtl')
    wg('http://files.is.tue.mpg.de/mloper/opendr/images/nasa_earth.jpg')

def get_earthmesh(trans, rotation):
    from chumpy.utils import row, col
    from serialization import load_mesh
    import os.path
    import sys

    from copy import deepcopy
    if not hasattr(get_earthmesh, 'm'):

        download_earthmesh()

        fname = join(os.path.dirname(__file__), 'nasa_earth.obj')
        mesh = load_mesh(fname)

        mesh.v = np.asarray(mesh.v, order='C')
        mesh.vc = mesh.v*0 + 1
        mesh.v -= row(np.mean(mesh.v, axis=0))
        mesh.v /= np.max(mesh.v)
        mesh.v *= 2.0
        get_earthmesh.mesh = mesh

    mesh = deepcopy(get_earthmesh.mesh)
    mesh.v = mesh.v.dot(cv2.Rodrigues(np.asarray(np.array(rotation), np.float64))[0])
    mesh.v = mesh.v + row(np.asarray(trans))
    return mesh



def process(im, vmin, vmax):
    shape = im.shape
    im = deepcopy(im).flatten()
    im[im>vmax] = vmax
    im[im<vmin] = vmin
    im -= vmin
    im /= (vmax-vmin)
    im = im.reshape(shape)
    return im
