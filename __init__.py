

def test():
    from os.path import split
    import unittest
    test_loader = unittest.TestLoader()
    test_loader = test_loader.discover(split(__file__)[0])
    test_runner = unittest.TextTestRunner()
    test_runner.run( test_loader )

demos = {}

demos['texture'] = """
# Create renderer
import chumpy as ch
from opendr.renderer import TexturedRenderer
rn = TexturedRenderer()

# Assign attributes to renderer
from opendr.util_tests import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m.v, f=m.f, vc=m.vc, texture_image=m.texture_image, ft=m.ft, vt=m.vt)

# Show it
import matplotlib.pyplot as plt
plt.ion()
plt.imshow(rn.r)
plt.show()

dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc
"""

demos['moments'] = """
from opendr.util_tests import get_earthmesh
from opendr.simple import * 
import numpy as np

w, h = 320, 240

m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))

# Create V, A, U, f: geometry, brightness, camera, renderer
V = ch.array(m.v)
A = SphericalHarmonics(vn=VertNormals(v=V, f=m.f), 
                       components=[3.,1.,0.,0.,0.,0.,0.,0.,0.], 
                       light_color=ch.ones(3))
U = ProjectPoints(v=V, f=[300,300.], c=[w/2.,h/2.], k=ch.zeros(5),
                  t=ch.zeros(3), rt=ch.zeros(3))
rn = TexturedRenderer(vc=A, camera=U, f=m.f, bgcolor=[0.,0.,0.],
                     texture_image=m.texture_image, vt=m.vt, ft=m.ft,
                     frustum={'width':w, 'height':h, 'near':1,'far':20})

i, j = ch.array([2.]), ch.array([1.])
xs, ys = ch.meshgrid(range(rn.shape[1]), range(rn.shape[0]))
ysp = ys ** j
xsp = xs ** i
rn_bw = ch.sum(rn, axis=2)
moment = ch.sum((rn_bw * ysp * xsp).ravel())

# Print our numerical result
print moment 

# Note that opencv produces the same result for 'm21',
# and that other moments can be created by changing "i" and "j" above
import cv2
print cv2.moments(rn_bw.r)['m21']

# Derivatives wrt vertices and lighting
print moment.dr_wrt(V)
print moment.dr_wrt(A.components)
"""

demos['per_face_normals'] = """
# Create renderer
import chumpy as ch
import numpy as np
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
rn = ColoredRenderer()

# Assign attributes to renderer
from opendr.util_tests import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)

# THESE ARE THE 3 CRITICAL LINES 
m.v = m.v[m.f.ravel()]
m.vc = m.vc[m.f.ravel()]
m.f = np.arange(m.f.size).reshape((-1,3))

from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m.v, f=m.f, bgcolor=ch.zeros(3))

# Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m.v),
    light_pos=ch.array([-1000,-1000,-1000]),
    vc=m.vc,
    light_color=ch.array([1., 1., 1.]))

# Show it
import matplotlib.pyplot as plt
plt.ion()
plt.imshow(rn.r)
plt.show()

dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc
"""

demos['silhouette'] = """
# Create renderer
import chumpy as ch
from opendr.renderer import ColoredRenderer
rn = ColoredRenderer()

# Assign attributes to renderer
from opendr.util_tests import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m.v, f=m.f, vc=m.vc*0+1, bgcolor=ch.zeros(3))

# Show it
import matplotlib.pyplot as plt
plt.ion()
plt.imshow(rn.r)
plt.show()

dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc
"""

demos['boundary'] = """
# Create renderer
import chumpy as ch
from opendr.renderer import BoundaryRenderer
rn = BoundaryRenderer()

# Assign attributes to renderer
from opendr.util_tests import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m.v, f=m.f, vc=m.vc*0+1, bgcolor=ch.zeros(3), num_channels=3)

# Show it
import matplotlib.pyplot as plt
plt.ion()
plt.imshow(rn.r)
plt.show()

dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc
"""

demos['point_light'] = """
# Create renderer
import chumpy as ch
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight
rn = ColoredRenderer()

# Assign attributes to renderer
from opendr.util_tests import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)

from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m.v, f=m.f, bgcolor=ch.zeros(3))

# Construct point light source
rn.vc = LambertianPointLight(
    f=m.f,
    v=rn.v,
    num_verts=len(m.v),
    light_pos=ch.array([-1000,-1000,-1000]),
    vc=m.vc,
    light_color=ch.array([1., 1., 1.]))

# Show it
import matplotlib.pyplot as plt
plt.ion()
plt.imshow(rn.r)
plt.show()

dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc
"""

demos['spherical_harmonics'] = """
# Create renderer
import chumpy as ch
from opendr.renderer import ColoredRenderer
from opendr.lighting import SphericalHarmonics
from opendr.geometry import VertNormals

rn = ColoredRenderer()

# Assign attributes to renderer
from opendr.util_tests import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m.v, f=m.f, bgcolor=ch.zeros(3))

vn = VertNormals(v=rn.v, f=rn.f)
sh_red = SphericalHarmonics(vn=vn, light_color=ch.array([1,0,0]), components=ch.random.randn(9))
sh_green = SphericalHarmonics(vn=vn, light_color=ch.array([0,1,0]), components=ch.random.randn(9))
sh_blue = SphericalHarmonics(vn=vn, light_color=ch.array([0,0,1]), components=ch.random.randn(9))
rn.vc = sh_red + sh_green + sh_blue

# Show it
import matplotlib.pyplot as plt
plt.ion()
plt.imshow(rn.r)
plt.show()

dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc
"""

demos['optimization'] = """
from opendr.simple import *      
import numpy as np 
import matplotlib.pyplot as plt
w, h = 320, 240                                                                 
        
try:
    m = load_mesh('earth.obj')
except:                                                             
    from opendr.util_tests import get_earthmesh
    m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))                                             
                                                                                
# Create V, A, U, f: geometry, brightness, camera, renderer                     
V = ch.array(m.v)                                                               
A = SphericalHarmonics(vn=VertNormals(v=V, f=m.f),                              
                       components=[3.,2.,0.,0.,0.,0.,0.,0.,0.],
                       light_color=ch.ones(3))                                  
U = ProjectPoints(v=V, f=[300,300.], c=[w/2.,h/2.], k=ch.zeros(5),              
                  t=ch.zeros(3), rt=ch.zeros(3))                                
f = TexturedRenderer(vc=A, camera=U, f=m.f, bgcolor=[0.,0.,0.],                 
                     texture_image=m.texture_image, vt=m.vt, ft=m.ft,           
                     frustum={'width':w, 'height':h, 'near':1,'far':20})     

# Parameterize the vertices                                                     
translation, rotation = ch.array([0,0,4]), ch.zeros(3)                          
f.v = translation + V.dot(Rodrigues(rotation))                                  
                                                                                
observed = f.r 
translation[:] = translation.r + np.random.rand(3)*.2
rotation[:] = rotation.r + np.random.rand(3)*.2
A.components[1:] = 0

# Create the energy
difference = f - observed
E = gaussian_pyramid(difference, n_levels=6, as_list=True)[-3:]
E = ch.concatenate([e.ravel() for e in E])

plt.ion()
global cb
global difference
global plt

def cb(_):
    plt.imshow(np.abs(difference.r))
    plt.title('Absolute difference')
    plt.draw()
         
         
# Minimize the energy                                                           
light_parms = A.components     
print 'OPTIMIZING TRANSLATION'                                                 
ch.minimize({'energy': E}, x0=[translation], callback=lambda _ : cb(difference)) 

print 'OPTIMIZING TRANSLATION, ROTATION, AND LIGHT PARMS'                                                 
ch.minimize({'energy': E}, x0=[translation, rotation, light_parms], callback=cb) 

E = gaussian_pyramid(difference, n_levels=6, normalization='size')
ch.minimize({'energy': E}, x0=[translation, rotation, light_parms], callback=cb) 

"""


def demo(which=None):
    import re
    if which not in demos:
        print 'Please indicate which demo you want, as follows:'
        for key in demos:
            print "\tdemo('%s')" % (key,)
        return

    print '- - - - - - - - - - - <CODE> - - - - - - - - - - - -'
    print re.sub('global.*\n','',demos[which])
    print '- - - - - - - - - - - </CODE> - - - - - - - - - - - -\n'
    exec('global np\n' + demos[which], globals(), locals())
