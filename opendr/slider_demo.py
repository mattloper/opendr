from .cvwrap import cv2
import numpy as np
import chumpy as ch
from copy import deepcopy

def nothing(x):
    pass
    
    
def get_renderer():
    import chumpy as ch
    from opendr.everything import *

    # Load mesh
    m = load_mesh('/Users/matt/geist/OpenDR/test_dr/nasa_earth.obj')
    m.v += ch.array([0,0,3.])
    w, h = (320, 240)
    trans = ch.array([[0,0,0]])

    # Construct renderer
    rn = TexturedRenderer()
    rn.camera = ProjectPoints(v=m.v, rt=ch.zeros(3), t=ch.zeros(3), f=ch.array([w,w])/2., c=ch.array([w,h])/2., k=ch.zeros(5))
    rn.frustum = {'near': 1., 'far': 10., 'width': w, 'height': h}
    rn.set(v=trans+m.v, f=m.f, texture_image=m.texture_image[:,:,::-1], ft=m.ft, vt=m.vt, bgcolor=ch.zeros(3))
    rn.vc = SphericalHarmonics(vn=VertNormals(v=rn.v, f=rn.f), components=ch.array([4.,0.,0.,0.]), light_color=ch.ones(3))

    return rn
    
    
def main():
    # Create a black image, a window
    img = np.zeros((300,512,3), np.uint8)
    cv2.namedWindow('image')
    cv2.namedWindow('derivatives')

    rn = get_renderer()

    tracked = {
        'sph0': rn.vc.components[0],
        'sph1': rn.vc.components[1],
        'sph2': rn.vc.components[2],
        'sph3': rn.vc.components[3],
        'k0': rn.camera.k[0],
        'k1': rn.camera.k[1],
        'k2': rn.camera.k[2]
    }

    cnst = 1000
    for k in sorted(tracked.keys()):
        v = tracked[k]
        cv2.createTrackbar(k, 'image', 0,cnst, nothing)

    old_tracked = tracked
    cv2.setTrackbarPos('sph0', 'image', 800)
    while(1):
        cv2.imshow('image',rn.r)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    
        for k, v in list(tracked.items()):
            v[:] = np.array(cv2.getTrackbarPos(k, 'image')).astype(np.float32)*4/cnst
            if tracked[k].r[0] != old_tracked[k].r[0]:
                drim = rn.dr_wrt(v).reshape(rn.shape)
                mn = np.mean(drim)
                drim /= np.max(np.abs(drim.ravel()))*2.
                drim += .5
                # drim = drim - np.min(drim)
                # drim = drim / np.max(drim)
                cv2.imshow('derivatives', drim)
               
        cv2.waitKey(1)
        old_tracked = deepcopy(tracked)
    # while True: 
    #     for k_change in sorted(tracked.keys()):
    #         if k_change == 'sph0':
    #             continue
    #         for t in np.arange(0, np.pi, .05):
    #             cv2.setTrackbarPos(k_change, 'image', int(np.sin(t)*1000))
    #             cv2.imshow('image',rn.r)
    #             k = cv2.waitKey(1) & 0xFF
    #             if k == 27:
    #                 break
    # 
    #             for k, v in tracked.items():
    #                 v[:] = np.array(cv2.getTrackbarPos(k, 'image')).astype(np.float32)*4/cnst
    #                 if tracked[k].r[0] != old_tracked[k].r[0]:
    #                     drim = rn.dr_wrt(v).reshape(rn.shape)
    #                     mn = np.mean(drim)
    #                     drim /= np.max(np.abs(drim.ravel()))*2.
    #                     drim += .5
    #                     # drim = drim - np.min(drim)
    #                     # drim = drim / np.max(drim)
    #                     cv2.imshow('derivatives', drim)
    #         
    #         
    #             print rn.vc.components
    # 
    #             cv2.waitKey(1)
    #             old_tracked = deepcopy(tracked)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
