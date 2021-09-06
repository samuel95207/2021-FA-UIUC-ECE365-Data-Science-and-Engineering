import numpy as np
import scipy.spatial.distance as dist
from scipy import stats
from sklearn import neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time


class Question1(object):
    def bayesClassifier(self, data, pi, means, cov):

        mid0 = np.log(pi)
        mid1 = np.dot(means, np.dot(np.linalg.inv(cov), data.T)).T
        mid2 = -0.5 * np.sum(np.dot(means, np.linalg.inv(cov)) * means, axis=1)

        labels = np.argmax(mid0 + mid1 + mid2, axis=1)

        return labels

    def classifierError(self, truelabels, estimatedlabels):

        errorMat = truelabels != estimatedlabels
        error = np.sum(errorMat) / truelabels.shape[0]

        return error


class Question2(object):
    def trainLDA(self, trainfeat, trainlabel):
        # Assuming all labels up to nlabels exist.
        nlabels = int(trainlabel.max())+1
        pi = np.zeros(nlabels)            # Store your prior in here
        # Store the class means in here
        means = np.zeros((nlabels, trainfeat.shape[1]))
        # Store the covariance matrix in here
        cov = np.zeros((trainfeat.shape[1], trainfeat.shape[1]))
        # Put your code below

        N = trainfeat.shape[0]
        M = means.shape[0]

        for i in range(nlabels):
            labelFilter = (trainlabel == i)
            filteredTrainfeat = trainfeat[labelFilter]

            Nl = filteredTrainfeat.shape[0]

            pi[i] = Nl / N
            means[i] = np.sum(filteredTrainfeat, axis=0) / Nl

            cov += np.dot((filteredTrainfeat - means[i]).T, (filteredTrainfeat - means[i]))

        cov /= N - M

        # Don't change the output!
        return (pi, means, cov)

    def estTrainingLabelsAndError(self, trainingdata, traininglabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)

        pi, means, cov = self.trainLDA(trainingdata, traininglabels)
        esttrlabels = q1.bayesClassifier(trainingdata, pi, means, cov)
        trerror = q1.classifierError(traininglabels, esttrlabels)

        # Don't change the output!
        return (esttrlabels, trerror)

    def estValidationLabelsAndError(self, trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        # You can use results from Question 1 by calling q1.bayesClassifier(...), etc.
        # If you want to refer to functions under the same class, you need to use self.fun(...)

        pi, means, cov = self.trainLDA(trainingdata, traininglabels)
        estvallabels = q1.bayesClassifier(valdata, pi, means, cov)
        valerror = q1.classifierError(vallabels, estvallabels)

        # Don't change the output!
        return (estvallabels, valerror)


class Question3(object):
    def kNN(self, trainfeat, trainlabel, testfeat, k):

        N = testfeat.shape[0]

        distanceMat = dist.cdist(testfeat, trainfeat)

        nearestKPointIndex = np.argpartition(distanceMat, k-1)

        KLabels = np.zeros((N, k))
        for i in range(N):
            for j in range(k):
                pointIndex = nearestKPointIndex[i][j]
                KLabels[i][j] = trainlabel[pointIndex]

        labels = stats.mode(KLabels, axis=1)[0][:, 0]
        return labels

    def kNN_errors(self, trainingdata, traininglabels, valdata, vallabels):
        q1 = Question1()
        trainingError = np.zeros(4)
        validationError = np.zeros(4)
        k_array = [1, 3, 4, 5]

        for i in range(len(k_array)):
            # Please store the two error arrays in increasing order with k
            # This function should call your previous self.kNN() function.
            # Put your code below
            trainingResult = self.kNN(trainingdata, traininglabels, trainingdata, k_array[i])
            validateResult = self.kNN(trainingdata, traininglabels, valdata, k_array[i])

            trainingError[i] = q1.classifierError(traininglabels, trainingResult)
            validationError[i] = q1.classifierError(vallabels, validateResult)

        # Don't change the output!
        return (trainingError, validationError)


class Question4(object):
    def sklearn_kNN(self, traindata, trainlabels, valdata, vallabels):

        classifier = neighbors.KNeighborsClassifier(n_neighbors=1, algorithm='brute')

        fitStart = time.time()
        classifier.fit(traindata, trainlabels)
        fitEnd = time.time()

        predictionStart = time.time()
        prediction = classifier.predict(valdata)
        predictionEnd = time.time()

        errorMat = vallabels != prediction
        valerror = np.sum(errorMat) / vallabels.shape[0]

        fitTime = fitEnd - fitStart
        predTime = predictionEnd - predictionStart

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

    def sklearn_LDA(self, traindata, trainlabels, valdata, vallabels):

        classifier = LinearDiscriminantAnalysis()

        fitStart = time.time()
        classifier.fit(traindata, trainlabels)
        fitEnd = time.time()

        predictionStart = time.time()
        prediction = classifier.predict(valdata)
        predictionEnd = time.time()

        errorMat = vallabels != prediction
        valerror = np.sum(errorMat) / vallabels.shape[0]

        fitTime = fitEnd - fitStart
        predTime = predictionEnd - predictionStart

        # Don't change the output!
        return (classifier, valerror, fitTime, predTime)

###
