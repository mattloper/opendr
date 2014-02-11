

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
from drender.renderer import TexturedRenderer
rn = TexturedRenderer()

# Assign attributes to renderer
from drender.test_dr.common import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from drender.camera import ProjectPoints
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

demos['silhouette'] = """
# Create renderer
import chumpy as ch
from drender.renderer import ColoredRenderer
rn = ColoredRenderer()

# Assign attributes to renderer
from drender.test_dr.common import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from drender.camera import ProjectPoints
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
from drender.renderer import BoundaryRenderer
rn = BoundaryRenderer()

# Assign attributes to renderer
from drender.test_dr.common import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from drender.camera import ProjectPoints
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
from drender.renderer import ColoredRenderer
from drender.lighting import LambertianPointLight
rn = ColoredRenderer()

# Assign attributes to renderer
from drender.test_dr.common import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)

from drender.camera import ProjectPoints
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
from drender.renderer import ColoredRenderer
from drender.lighting import SphericalHarmonics
from drender.geometry import VertNormals

rn = ColoredRenderer()

# Assign attributes to renderer
from drender.test_dr.common import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from drender.camera import ProjectPoints
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


def demo(which=None):
    if which not in demos:
        print 'Please indicate which demo you want, as follows:'
        for key in demos:
            print "\tdemo('%s')" % (key,)
        return

    print '- - - - - - - - - - - <CODE> - - - - - - - - - - - -'
    print demos[which]
    print '- - - - - - - - - - - </CODE> - - - - - - - - - - - -\n'
    exec('global np\n' + demos[which], globals(), locals())
