import numpy as np
from skimage.transform import radon, iradon, resize
import scipy


class Question1(object):
    def complete_ortho_matrix(self, M):
        cross = np.cross(M[:, 0], M[:, 1]).reshape((3, 1))
        orth_mat = np.concatenate((M, cross), axis=1)
        return orth_mat

    def recover_ortho_matrix(self, M):
        U, S, vh = np.linalg.svd(M)
        orth_mat = U @ vh
        return orth_mat

    def comp_rec_ortho_matrix(self, M):
        orth_mat = self.complete_ortho_matrix(M)
        orth_mat = self.recover_ortho_matrix(orth_mat)
        return orth_mat


class Question2(object):
    def template_matching(self, noisy_proj, I0, M, Tmax):
        theta = np.linspace(0, 360, M)
        
        for t in range(Tmax):
            radonResult = radon(I0, theta=theta)
            corr = (noisy_proj.T @ radonResult) / np.sqrt(np.diagonal(radonResult.T @ radonResult))
            idx = np.argmax(corr, axis=1)
            theta_best = theta[idx]
            I0 = iradon(noisy_proj, theta=theta_best)

        theta = theta_best
        I_rec = I0
        return I_rec, theta
