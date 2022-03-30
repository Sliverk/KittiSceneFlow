import os 
import numpy as np 
import matplotlib.image as mpimg

class kittiLidarProjection():
    def __init__(self, KITTI_ROOT='../data', prefix='000002'):
        self.calib_velo = os.path.join(KITTI_ROOT, f'calib_velo/{prefix}.txt')
        self.calib_cam  = os.path.join(KITTI_ROOT, f'calib_cam/{prefix}.txt')
        self.liadr = os.path.join(KITTI_ROOT, f'velodyne/{prefix}.bin')
        self.image = os.path.join(KITTI_ROOT, f'image_2/{prefix}_10.png')
    
    def parse_velo_calib(self, calib_velo):
        with open(calib_velo, 'r') as f: lines = f.readlines()
        lines = [x.strip() for x in lines]
        R = np.matrix([float(x) for x in lines[1].split(' ')[1:]]).reshape(3,3)
        T = np.matrix([float(x) for x in lines[2].split(' ')[1:]]).reshape(3,1)
        Tr_velo_to_cam = np.concatenate((R,T), axis=1)
        # To homogeneous
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0,0,0,1], axis=0)
        
        return Tr_velo_to_cam

    def parse_cam_calib(self, calib_cam):
        with open(calib_cam, 'r') as f: lines = f.readlines()
        lines = [x.strip() for x in lines]
        R0_rect = np.matrix([float(x) for x in lines[8].split(' ')[1:]])
        P2 = np.matrix([float(x) for x in lines[25].split(' ')[1:]])
        # To homogeneous
        P2 = P2.reshape(3, 4)
        R0_rect = R0_rect.reshape(3, 3)
        R0_rect = np.insert(R0_rect, 3, values=[0,0,0], axis=0)
        R0_rect = np.insert(R0_rect, 3, values=[0,0,0,1], axis=1)
        
        return P2, R0_rect
    
    def projection(self, show=True):
        Tr_velo_to_cam = self.parse_velo_calib(self.calib_velo)
        P2, R0_rect = self.parse_cam_calib(self.calib_cam)

        # Load Velodyne
        velo = np.fromfile(self.liadr, dtype=np.float32).reshape((-1,4))[:, 0:3]

        # Load Image
        data_img = mpimg.imread(self.image)

        # Projection
        velo = np.insert(velo,3,1,axis=1).T
        velo = np.delete(velo, np.where(velo[0,:]<0), axis=1)

        l2i = P2 * R0_rect * Tr_velo_to_cam * velo
        l2i = np.delete(l2i, np.where(l2i[2,:]<0)[1], axis=1)
        l2i[:2] /= l2i[2,:]

        # filter point out of canvas
        u,v,z = l2i
        IMG_H,IMG_W,_ = data_img.shape
        u_out = np.logical_or(u<0, u>IMG_W)
        v_out = np.logical_or(v<0, v>IMG_H)
        outlier = np.logical_or(u_out, v_out)
        l2i = np.delete(l2i,np.where(outlier),axis=1)

        # generate color map from depth
        # 
        if show == True:
            import matplotlib.pyplot as plt
            plt.axis([0,IMG_W,IMG_H,0])
            plt.imshow(data_img)
            u,v,z = l2i
            plt.scatter([u],[v],c=[z],cmap='rainbow_r',alpha=0.5,s=2)
            plt.show()

        return l2i 



