import numpy as np


class Question1(object):
    def rotate_matrix(self, theta):
        cosRad, sinRad = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        R_theta = np.array([[cosRad, -sinRad],
                            [sinRad, cosRad]])
        return R_theta

    def rotate_2d(self, points, theta):
        rot_points = self.rotate_matrix(theta) @ points
        return rot_points

    def combine_rotation(self, theta1, theta2):
        err = np.linalg.norm(
            self.rotate_matrix(theta1) @ self.rotate_matrix(theta2)
            - self.rotate_matrix(theta1+theta2))
        return err


class Question2(object):
    def rotate_matrix_x(self, theta):
        cosRad, sinRad = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        R_x = np.array([[1, 0, 0],
                       [0, cosRad, -sinRad],
                       [0, sinRad, cosRad]])
        return R_x

    def rotate_matrix_y(self, theta):
        cosRad, sinRad = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        R_y = np.array([[cosRad, 0, sinRad],
                       [0, 1, 0],
                       [-sinRad, 0, cosRad]])
        return R_y

    def rotate_matrix_z(self, theta):
        cosRad, sinRad = np.cos(np.radians(theta)), np.sin(np.radians(theta))
        R_z = np.array([[cosRad, -sinRad, 0],
                       [sinRad, cosRad, 0],
                       [0, 0, 1]])
        return R_z

    def rot_matrix(self, alpha, beta, gamma):
        R = self.rotate_matrix_z(gamma) @ self.rotate_matrix_y(beta) @ self.rotate_matrix_z(alpha)
        return R

    def rotate_point(self, points, R):
        rot_points = R @ points
        return rot_points


class Question3(object):
    def rotate_x_axis(self, image_size, theta):
        N = image_size // 2
        regular_grid = np.linspace(-N, N, image_size)
        x_grid, y_grid = np.meshgrid(regular_grid, 0)
        points = np.concatenate((np.reshape(x_grid, [1, -1]),
                                np.reshape(y_grid, [1, -1])),
                                axis=0)
        rot_points = (Question1().rotate_matrix(theta) @ points)
        return rot_points

    def nudft2(self, img, grid_f):
        image_size = img.shape[0]
        N = image_size // 2
        regular_grid = np.linspace(-N, N, image_size)
        x_grid, y_grid = np.meshgrid(regular_grid, regular_grid)

        img_f = np.zeros((grid_f.shape[1]), dtype='complex_')
        for i, kPoint in enumerate(grid_f.T):
            expMat = np.exp(-1j*2*np.pi/image_size*(x_grid*kPoint[0]+y_grid*kPoint[1]))
            img_f[i] = np.sum(img*expMat)

        return img_f

    def gen_projection(self, img, theta):
        # grid on which to compute the fourier transform
        points_rot = self.rotate_x_axis(img.shape[0], theta)

        # Don't change the rest of the code and the output!
        ft_img = self.nudft2(img, points_rot)
        proj = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(ft_img)))
        proj = np.real(proj)
        return proj


class Question4(object):
    def rotate_z_axis(self, L, R_theta):
        N = L // 2
        regular_grid = np.linspace(-N, N, L)
        x_grid, y_grid, z_grid = np.meshgrid(regular_grid, regular_grid, 0)
        points = np.concatenate((np.reshape(x_grid, [1, -1]),
                                np.reshape(y_grid, [1, -1]),
                                np.reshape(z_grid, [1, -1])),
                                axis=0)
        rot_points = R_theta @ points
        return rot_points

    def nudft3(self, vol, grid_f):
        L = vol.shape[0]
        N = L // 2
        regular_grid = np.linspace(-N, N, L)
        x_grid, y_grid, z_grid = np.meshgrid(regular_grid, regular_grid, regular_grid)

        vol_f = np.zeros((grid_f.shape[1]), dtype='complex_')
        for i, kPoint in enumerate(grid_f.T):
            expMat = np.exp(-1j*2*np.pi/L*(x_grid*kPoint[0]+y_grid*kPoint[1]+z_grid*kPoint[2]))
            vol_f[i] = np.sum(vol*expMat)

        return vol_f

    def gen_projection(self, vol, R_theta):
        vol_sz = vol.shape[0]
        # grid on which to compute the fourier transform
        xy_plane_rot = self.rotate_z_axis(vol.shape[0], R_theta)

        # Put your code here
        # ...

        # Don't change the rest of the code and the output!
        ft_vol = self.nudft3(vol, xy_plane_rot)
        ft_vol = np.reshape(ft_vol, [vol_sz, vol_sz])
        proj_img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(ft_vol)))
        proj_img = np.real(proj_img)
        return proj_img

    def apply_ctf(self, img, ctf):
        # Nothing to add here!
        fm = np.fft.fftshift(np.fft.fftn(img))
        cm = np.real(np.fft.ifftn(np.fft.ifftshift(fm * ctf)))
        return cm
