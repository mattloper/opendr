"""
Author(s): Matthew Loper

See LICENCE.txt for licensing and contact information.
"""

__author__ = 'matt'

__all__ = ['load_mesh', 'load_image']

from os.path import split, splitext, join, exists, normpath
from .cvwrap import cv2
import numpy as np
from .dummy import Minimal


def load_image(filename):
    return (cv2.imread(filename)[:,:,::-1]/255.).copy()

def load_mesh(filename):

    extension = splitext(filename)[1]
    if  extension == '.ply':
        return read_ply(filename)
    elif extension == '.obj':
        return read_obj(filename)
    else:
        raise Exception('Unsupported file extension for %s' % (filename,))

def _update_mtl(mtl, filename):

    lines = [l.strip() for l in open(filename).read().split('\n')]

    curkey = ''
    for line in lines:
        spl = line.split()

        if len(spl) < 2:
            continue
        key = spl[0]
        values = spl[1:]

        if key == 'newmtl':
            curkey = values[0]
            mtl[curkey] = {'filename': filename}
        elif curkey:
            mtl[curkey][key] = values


def read_obj(filename):

    obj_directory = split(filename)[0]
    lines = open(filename).read().split('\n')

    d = {'v': [], 'vn': [], 'f': [], 'vt': [], 'ft': []}

    mtls = {}
    for line in lines:
        line = line.split()
        if len(line) < 2:
            continue

        key = line[0]
        values = line[1:]

        if key == 'v':
            d['v'].append([np.array([float(v) for v in values[:3]])])
        elif key == 'f':
            spl = [l.split('/') for l in values]
            d['f'].append([np.array([int(l[0])-1 for l in spl[:3]], dtype=np.uint32)])
            if len(spl[0]) > 1 and spl[1] and 'ft' in d:
                d['ft'].append([np.array([int(l[1])-1 for l in spl[:3]])])

            # TOO: redirect to actual vert normals?
            #if len(line[0]) > 2 and line[0][2]:
            #    d['fn'].append([np.concatenate([l[2] for l in spl[:3]])])
        elif key == 'vn':
            d['vn'].append([np.array([float(v) for v in values])])
        elif key == 'vt':
            d['vt'].append([np.array([float(v) for v in values])])
        elif key == 'mtllib':
            fname = join(obj_directory, values[0])
            if not exists(fname):
                fname = values[0]
            if not exists(fname):
                raise Exception("Can't find path %s" % (values[0]))
            _update_mtl(mtls, fname)
        elif key == 'usemtl':
            cur_mtl = mtls[values[0]]

            if 'map_Kd' in cur_mtl:
                src_fname = cur_mtl['map_Kd'][0]
                dst_fname = join(split(cur_mtl['filename'])[0], src_fname)
                if not exists(dst_fname):
                    dst_fname = join(obj_directory, src_fname)
                if not exists(dst_fname):
                    dst_fname = src_fname
                if not exists(dst_fname):
                    raise Exception("Unable to find referenced texture map %s" % (src_fname,))
                else:
                    d['texture_filepath'] = normpath(dst_fname)
                    im = cv2.imread(dst_fname)
                    sz = np.sqrt(np.prod(im.shape[:2]))
                    sz = int(np.round(2 ** np.ceil(np.log(sz) / np.log(2))))
                    d['texture_image'] = cv2.resize(im, (sz, sz)).astype(np.float64)/255.

    for k, v in list(d.items()):
        if k in ['v','vn','f','vt','ft']:
            if v:
                d[k] = np.vstack(v)
            else:
                del d[k]
        else:
            d[k] = v

    result = Minimal(**d)

    return result




def read_ply(filename):

    import numpy as np
    str2dtype = {
        'char':   np.int8,
        'uchar':  np.uint8,
        'short':  np.int16,
        'ushort': np.uint16,
        'int':    np.int32,
        'uint':   np.uint32,
        'float':  np.float32,
        'double': np.float64
    }

    data = open(filename, 'rb').read()

    if data[:3] != 'ply':
        raise Exception('Need ply header.')

    if data[3:5] == '\r\n':
        newline = '\r\n'
    elif data[3] == '\r':
        newline = '\r'
    elif data[3] == '\n':
        newline = '\n'
    else:
        raise Exception('Bad newline after ply header.')


    pos = data.find('end_header' + newline)
    header = data[:pos].split(newline)
    body = data[pos + len('end_header' + newline):]

    elements = []
    for line in header:
        tokens = [ln for ln in line.split(' ') if len(ln) > 0]
        if len(tokens) <= 0:
            continue
        if tokens[0] == 'format':
            format = tokens[1]

        elif tokens[0] == 'element':
            elements.append({'name': tokens[1], 'len': int(tokens[2]), 'properties': []})

        elif tokens[0] == 'property':
            prop = {'name': tokens[-1], 'dtype': str2dtype[tokens[-2]]}
            if tokens[1] == 'list':
                prop['list_dtype'] = str2dtype[tokens[-3]]
            elements[-1]['properties'].append(prop)

    newelems = {}
    if format == 'ascii':
        body = body.split(newline)
        while len(elements) > 0:
            element = elements.pop(0)
            newelem = {}
            if 'list_dtype' in element['properties'][0]:
                tokens = [np.asarray(ln.split()[1:], dtype=element['properties'][0]['dtype']) for ln in body[:element['len']]]
                newelems[element['name']] = {element['properties'][0]['name']: np.vstack(tokens)}
            else:
                tokens = [ln.split() for ln in body[:element['len']]]
                for idx, prop in enumerate(element['properties']):
                    tmp = [tokens[i][idx] for i in range(len(tokens))]
                    newelem[prop['name']] = np.asarray(tmp, dtype=prop['dtype'])
                newelems[element['name']] = newelem

            body = body[element['len']:]

    v = np.vstack([newelems['vertex'][s] for s in ['x', 'y', 'z']]).T.copy()
    v = np.asarray(v, dtype=np.float64)
    f = list(newelems['face'].values())[0]

    mesh = Minimal(v=v, f=f)

    return mesh

if __name__ == '__main__':
    pass

