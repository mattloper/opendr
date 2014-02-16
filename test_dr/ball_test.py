__author__ = 'matt'


# Create renderer
import chumpy as ch
import numpy as np
import cv2
from opendr.renderer import DepthRenderer
rn = DepthRenderer()

# Assign attributes to renderer
from opendr.test_dr.common import get_earthmesh
m = get_earthmesh(trans=ch.array([0,0,4]), rotation=ch.zeros(3))
w, h = (320, 240)
from opendr.camera import ProjectPoints
rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
rn.set(v=m.v, f=m.f, bgcolor=ch.zeros(3))

r1 = rn.r

change = rn.v.r * 0 + np.array([[0,0,1]])*.01
dr_pred = rn.dr_wrt(rn.v).dot(change.ravel()).reshape(rn.shape)
rn.v = rn.v.r + change

r2 = rn.r
dr_emp = r2 - r1

cv2.imshow('pred', dr_pred / np.max(dr_emp.ravel())*200.)
cv2.imshow('emp', dr_emp / np.max(dr_emp.ravel())*200.)

import pdb; pdb.set_trace()
# Show it
# import matplotlib.pyplot as plt
# plt.ion()
# plt.imshow(rn.r)
# plt.show()
#
# dr = rn.dr_wrt(rn.v) # or rn.vc, or rn.camera.rt, rn.camera.t, rn.camera.f, rn.camera.c, etc