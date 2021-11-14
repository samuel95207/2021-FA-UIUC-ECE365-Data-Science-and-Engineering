import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix


class Question1(object):
    def svm_classifiers(self, X, y):
        svm_linear = svm.SVC(kernel="linear").fit(X, y)
        svm_non_linear = svm.SVC(kernel="rbf").fit(X, y)
        return svm_linear, svm_non_linear

    def acc_prec_recall(self, y_pred, y_test):
        tn, fp, fn, tp = confusion_matrix(y_true=y_test, y_pred=y_pred).ravel()
        acc = (tp+tn)/(tp+fp+tn+fn)
        prec = tp/(tp+fp)
        recall = tp/(tp+fn)
        return acc, prec, recall


class Question2(object):
    def CCF_1d(self, x1, x2):
        ccf = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            x1d = x1
            x2d = np.roll(x2, -i)
            ccf[i] = sum(x1d*x2d)
        return ccf

    # def align_1d(self, x1, x2):
    #     maxCCF = -np.inf
    #     aligned_sig = x2
    #     for i in range(x2.shape[0]):
    #         x2Shift = np.roll(x2, i)
    #         cffPeak = self.CCF_1d(x1, x2Shift)[0]
    #         if(cffPeak > maxCCF):
    #             maxCCF = cffPeak
    #             aligned_sig = x2Shift

    #     return aligned_sig

    def align_1d(self, x1, x2):
        maxCcfPos = np.argmax(self.CCF_1d(x1, x2))
        aligned_sig = np.roll(x2, -maxCcfPos)
        return aligned_sig


class Question3(object):
    def CCF_2d(self, x1, x2):
        ccf = np.zeros(x1.shape)
        for i in range(x1.shape[0]):
            for j in range(x1.shape[1]):
                x1d = x1
                x2d = np.roll(x2, -i, axis=0)
                x2d = np.roll(x2d, -j, axis=1)
                ccf[i][j] = np.sum(x1d*x2d)

        return ccf

    def align_2d(self, x1, x2):
        maxCcfPos = np.unravel_index(np.argmax(self.CCF_2d(x1, x2)), x1.shape)
        aligned_img = np.roll(x2, -maxCcfPos[0], axis=0)
        aligned_img = np.roll(aligned_img, -maxCcfPos[1], axis=1)

        return aligned_img

    def response_signal(self, ref_images, query_image):
        resp = np.zeros(ref_images.shape[2])
        for idx in range(ref_images.shape[2]):
            C_fk_g = self.CCF_2d(ref_images[:,:,idx], query_image)
            resp[idx] = np.max(C_fk_g) - 1/(query_image.shape[0]**2) * np.sum(C_fk_g)
        return resp
