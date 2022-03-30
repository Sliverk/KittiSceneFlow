# Code to recover sceneflow
# Also display sceneflow on point cloud
# For kitti sceneflow dataset
# 2022.03.28
# Haodi ZHANG
# INSA Rouen

import os
import png
import numpy as np 
import skimage.io as io 
import cv2

########
# STEP 1: Read Optical Flow, Disparity 0, Disparity 1
# Optical Flow: ofl
# Disparity 0: disp0
# Disparity 1: disp1
########

KITTI_ROOT = '../data/'
f_ofl = os.path.join(KITTI_ROOT, 'flow_noc/000002_10.png')
f_disp0 = os.path.join(KITTI_ROOT, 'disp_noc_0/000002_10.png')
f_disp1 = os.path.join(KITTI_ROOT, 'disp_noc_1/000002_10.png')


def load_ofl(filepath):
    '''
    # USE Pypng0.0.18
    # Need to replace with source code
    # From self-mono-sf/common.py/read_png_flow()
    '''
    ofl_object = png.Reader(filename=filepath)
    ofl_direct = ofl_object.asDirect()
    ofl_data = list(ofl_direct[2])
    (w, h) = ofl_direct[3]['size']
    ofl = np.zeros((h,w,3), dtype=np.float64)
    for i in range(len(ofl_data)):
        ofl[i, :, 0] = ofl_data[i][0::3]
        ofl[i, :, 1] = ofl_data[i][1::3]
        ofl[i, :, 2] = ofl_data[i][2::3]
    invalid_idx = (ofl[:,:,2] == 0)
    '''
    # From KITTI Sceneflow 2015 devkit/readme.txt
    # Optical flow maps are saved as 3-channel uint16 PNG images: The first channel
    # contains the u-component, the second channel the v-component and the third
    # channel denotes if the pixel is valid or not (1 if true, 0 otherwise). To convert
    # the u-/v-flow into floating point values, convert the value to float, subtract
    # 2^15 and divide the result by 64.0:
    # flow_u(u,v) = ((float)I(u,v,1)-2^15)/64.0;
    # flow_v(u,v) = ((float)I(u,v,2)-2^15)/64.0;
    # valid(u,v)  = (bool)I(u,v,3);
    '''
    ofl[:, :, 0:2] = (ofl[:, :, 0:2] - 2**15) / 64.0
    ofl[invalid_idx, 0] = 0
    ofl[invalid_idx, 1] = 0
    return ofl[:, :, 0:2], (1-invalid_idx*1)[:, :, None]

def load_disp(filepath):
    '''
    # From self-mono-sf/common.py/read_png_flow()
    # From KITTI Sceneflow 2015 devkit/readme.txt
    # Disparity maps are saved as uint16 PNG images, which can be opened with
    # either MATLAB or libpng++. A 0 value indicates an invalid pixel (ie, no
    # ground truth exists, or the estimation algorithm didn't produce an estimate
    # for that pixel). Otherwise, the disparity for a pixel can be computed by
    # converting the uint16 value to float and dividing it by 256.0:
    # disp(u,v)  = ((float)I(u,v))/256.0;
    # valid(u,v) = I(u,v)>0;
    '''
    disp_np = io.imread(filepath).astype(np.uint16) / 256.0
    disp_np = np.expand_dims(disp_np, axis=2)
    disp_mask = (disp_np > 0).astype(np.float64)
    return disp_np, disp_mask
    # disp = cv2.imread(filepath, -1)
    # disp = disp.astype(np.float32) / 256
    # return disp

ofl, ofl_mask = load_ofl(f_ofl)
disp0, disp0_mask = load_disp(f_disp0)
disp1, disp1_mask = load_disp(f_disp1)
# disp0 = load_disp(f_disp0)
# disp1 = load_disp(f_disp1)


########
# STEP 2: Use x - x' = Bf / Z to recovery the depth
# Doc: https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html
########

# B: baseline, in meter
B = 0.54
'''
# f: focal length, in meter
# f = dx * fx = dy * fy
# Camera: FL2-14S3C-C, 
# dx = dy = 4.65um/pixel
# fx, fy = 7.215377e2 in P_rect_02 
# fx, fy find in Paper http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
'''
f = 7.215377 * 100 

mask = disp0 > 0
depth0 = f * 0.54 / (disp0 + (1.0 - mask))

mask = disp1 > 0
depth1 = f * 0.54 / (disp1 + (1.0 - mask))



########
# STEP 3: Get 3D points projection
########

from utils import kittiLidarProjection as klp

pcl = klp()
l2i = proj_pcl = pcl.projection(False)




########
# STEP 4: Joint Optical Flow, recover the sceneflow
########

ul,vl,zl = l2i
sf = 0
num = ul.shape[1]
for ix in range(num):
    u = int(ul[0,ix])
    v = int(vl[0,ix])
    z = zl[0,ix]
    
    if disp0_mask[v,u] == 0: continue        
    if ofl_mask[v,u] == 0: continue
    v1 = int(v + ofl[v,u,0])
    u1 = int(u + ofl[v,u,1])
    if v1 > 374 or v1 < 0 or u1 > 1241 or u1 < 0: continue
    if disp1_mask[v1, u1] == 0: continue
    sf += 1
    print(z, depth0[v,u], ofl[v,u], depth1[v,u])


