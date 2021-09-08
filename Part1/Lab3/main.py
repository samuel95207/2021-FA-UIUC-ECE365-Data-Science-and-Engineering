from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
import numpy as np
import time
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# You may use this function as you like.


def error(y, yhat): return np.mean(y != yhat)


class Question1(object):
    # The sequence in this problem is different from the one you saw in the jupyter notebook. This makes it easier to grade. Apologies for any inconvenience.
    def BernoulliNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a BernoulliNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = BernoulliNB()

        fitStart = time.time()
        classifier.fit(traindata, trainlabels)
        fitEnd = time.time()

        trainingPred = classifier.predict(traindata)
        predictionStart = time.time()
        valPred = classifier.predict(valdata)
        predictionEnd = time.time()

        trainingError = error(trainlabels, trainingPred)
        validationError = error(vallabels, valPred)

        fittingTime = fitEnd - fitStart
        valPredictingTime = predictionEnd - predictionStart

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def MultinomialNB_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a MultinomialNB classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = MultinomialNB()

        fitStart = time.time()
        classifier.fit(traindata, trainlabels)
        fitEnd = time.time()

        trainingPred = classifier.predict(traindata)
        predictionStart = time.time()
        valPred = classifier.predict(valdata)
        predictionEnd = time.time()

        trainingError = error(trainlabels, trainingPred)
        validationError = error(vallabels, valPred)

        fittingTime = fitEnd - fitStart
        valPredictingTime = predictionEnd - predictionStart

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LinearSVC_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LinearSVC classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = LinearSVC()

        fitStart = time.time()
        classifier.fit(traindata, trainlabels)
        fitEnd = time.time()

        trainingPred = classifier.predict(traindata)
        predictionStart = time.time()
        valPred = classifier.predict(valdata)
        predictionEnd = time.time()

        trainingError = error(trainlabels, trainingPred)
        validationError = error(vallabels, valPred)

        fittingTime = fitEnd - fitStart
        valPredictingTime = predictionEnd - predictionStart

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def LogisticRegression_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a LogisticRegression classifier using the given data.

        Parameters:
        1. traindata    (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels  (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata      (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels    (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = LogisticRegression()

        fitStart = time.time()
        classifier.fit(traindata, trainlabels)
        fitEnd = time.time()

        trainingPred = classifier.predict(traindata)
        predictionStart = time.time()
        valPred = classifier.predict(valdata)
        predictionEnd = time.time()

        trainingError = error(trainlabels, trainingPred)
        validationError = error(vallabels, valPred)

        fittingTime = fitEnd - fitStart
        valPredictingTime = predictionEnd - predictionStart

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def NN_classifier(self, traindata, trainlabels, valdata, vallabels):
        """ Train and evaluate a Nearest Neighbor classifier using the given data.

        Make sure to modify the default parameter.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. valdata              (Nv, d) numpy ndarray. The features in the validation set.
        4. vallabels            (Nv, ) numpy ndarray. The labels in the validation set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. trainingError        Float. The reported training error. It should be less than 1.
        3. validationError      Float. The reported validation error. It should be less than 1.
        4. fittingTime          Float. The time it takes to fit the classifier (i.e. time to perform xxx.fit(X,y)). This is not evaluated.
        5. valPredictingTime    Float. The time it takes to run the classifier on the validation data (i.e. time to perform xxx.predict(X,y)). This is not evaluated.

        You can ignore all errors, if any.
        """
        # Put your code below

        classifier = KNeighborsClassifier(n_neighbors=1, algorithm='brute')

        fitStart = time.time()
        classifier.fit(traindata, trainlabels)
        fitEnd = time.time()

        trainingPred = classifier.predict(traindata)
        predictionStart = time.time()
        valPred = classifier.predict(valdata)
        predictionEnd = time.time()

        trainingError = error(trainlabels, trainingPred)
        validationError = error(vallabels, valPred)

        fittingTime = fitEnd - fitStart
        valPredictingTime = predictionEnd - predictionStart

        # Do not change this sequence!
        return (classifier, trainingError, validationError, fittingTime, valPredictingTime)

    def confMatrix(self, truelabels, estimatedlabels):
        """ Write a function that calculates the confusion matrix (cf. Fig. 2.1 in the notes).

        You may wish to read Section 2.1.1 in the notes -- it may be helpful, but is not necessary to complete this problem.

        Parameters:
        1. truelabels           (Nv, ) numpy ndarray. The ground truth labels.
        2. estimatedlabels      (Nv, ) numpy ndarray. The estimated labels from the output of some classifier.

        Outputs:
        1. cm                   (2,2) numpy ndarray. The calculated confusion matrix.
        """
        cm = np.zeros((2, 2))
        # Put your code below

        AP = (truelabels == 1)  # Actually Positive
        AN = (truelabels == -1)  # Actually Negative
        PP = (estimatedlabels == 1)  # Predicted Positive
        PN = (estimatedlabels == -1)  # Predicted Negative

        cm[0, 0] = np.sum(AP & PP)
        cm[0, 1] = np.sum(AN & PP)
        cm[1, 0] = np.sum(AP & PN)
        cm[1, 1] = np.sum(AN & PN)

        return cm

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Run the classifier you selected in the previous part of the problem on the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data.
        2. testError            Float. The reported test error. It should be less than 1.
        3. confusionMatrix      (2,2) numpy ndarray. The resulting confusion matrix. This will not be graded here.
        """
        # Put your code below

        classifier = LogisticRegression()
        classifier.fit(traindata, trainlabels)
        estimatedlabels = classifier.predict(testdata)
        testError = error(testlabels, estimatedlabels)

        # You can use the following line after you finish the rest
        confusionMatrix = self.confMatrix(testlabels, estimatedlabels)
        # Do not change this sequence!
        return (classifier, testError, confusionMatrix)


class Question2(object):
    def crossValidationkNN(self, traindata, trainlabels, k):
        """ Write a function which implements 5-fold cross-validation to estimate the error of a classifier with cross-validation with the 0,1-loss for k-Nearest Neighbors (kNN).

        For this problem, take your folds to be 0:N/5, N/5:2N/5, ..., 4N/5:N for cross-validation.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. k                    Integer. The cross-validated error estimates will be outputted for 1,...,k.

        Outputs:
        1. err                  (k+1,) numpy ndarray. err[i] is the cross-validated estimate of using i neighbors (the zero-th component of the vector will be meaningless).
        """
        err = np.zeros(k+1)
        # Put your code below

        foldNum = 5
        N = traindata.shape[0]

        for i in range(1, k+1):
            iter_error = 0

            for j in range(foldNum):
                cut0_idx = j*N//foldNum
                cut1_idx = (j+1)*N//foldNum

                fold_valData = traindata[cut0_idx: cut1_idx, :]
                fold_valLabels = trainlabels[cut0_idx: cut1_idx]
                fold_trainData = np.concatenate((traindata[: cut0_idx, :], traindata[cut1_idx:, :]))
                fold_trainLabels = np.concatenate((trainlabels[: cut0_idx], trainlabels[cut1_idx:]))

                classifier = KNeighborsClassifier(n_neighbors=i, algorithm='brute')
                classifier.fit(fold_trainData, fold_trainLabels)

                fold_pred = classifier.predict(fold_valData)

                iter_error += error(fold_valLabels, fold_pred)

            iter_error /= foldNum
            err[i] = iter_error

        return err

    def minimizer_K(self, kNN_errors):
        """ Write a function that calls the above function and returns 1) the output from the previous function, 2) the number of neighbors within  1,...,k  that minimizes the cross-validation error, and 3) the correponding minimum error.

        Parameters:
        1. kNN_errors           (k+1,) numpy ndarray. The output from self.crossValidationkNN()

        Outputs:
        1. k_min                Integer (np.int64 or int). The number of neighbors within  1,...,k  that minimizes the cross-validation error.
        2. err_min              Float. The correponding minimum error.
        """
        # Put your code below

        k_min = np.argmin(kNN_errors[1:]) + 1
        err_min = kNN_errors[k_min]

        # Do not change this sequence!
        return (k_min, err_min)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train a kNN model on the whole training data using the number of neighbors you found in the previous part of the question, and apply it to the test data.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best k value that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below
        best_k = 14

        classifier = KNeighborsClassifier(best_k)
        classifier.fit(traindata, trainlabels)
        estimatedlabels = classifier.predict(testdata)
        testError = error(testlabels, estimatedlabels)

        # Do not change this sequence!
        return (classifier, testError)


class Question3(object):
    def LinearSVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15}.

        Write this without using GridSearchCV.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        k = 10
        C_list = [2 ** i for i in range(-5, 16)]
        error_list = np.zeros(len(C_list))

        for idx, C in enumerate(C_list):
            classifier = LinearSVC(C=C)
            cross_error_list = 1 - cross_val_score(classifier, traindata, trainlabels, cv=k)
            error_list[idx] = np.mean(cross_error_list)

        min_error_idx = np.argmin(error_list)
        min_err = error_list[min_error_idx]
        C_min = C_list[min_error_idx]

        # Do not change this sequence!
        return (C_min, min_err)

    def SVC_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-5},...,2^{15} and \gamma from 2^{-15},...,2^{3}.

        Use GridSearchCV to perform a grid search.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. gamma_min            Float. The hyper-parameter \gamma that minimizes the validation error.
        3. min_err              Float. The correponding minimum error.
        """
        # Put your code below
        k = 10
        C_list = [2 ** i for i in range(-14, 15)]
        gamma_list = [2 ** i for i in range(-15, 4)]

        classifier = SVC()
        gridSearch = GridSearchCV(classifier,
                                  param_grid={"C": C_list, "gamma": gamma_list},
                                  cv=k)
        gridSearch.fit(traindata, trainlabels)

        C_min = gridSearch.best_params_['C']
        gamma_min = gridSearch.best_params_['gamma']
        min_err = 1 - gridSearch.best_score_

        # Do not change this sequence!
        return (C_min, gamma_min, min_err)

    def LogisticRegression_crossValidation(self, traindata, trainlabels):
        """ Use cross-validation to select a value of C for a linear SVM by varying C from 2^{-14},...,2^{14}.

        You may either use GridSearchCV or search by hand.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.

        Outputs:
        1. C_min                Float. The hyper-parameter C that minimizes the validation error.
        2. min_err              Float. The correponding minimum error.
        """
        # Put your code below

        k = 10
        C_list = [2 ** i for i in range(-14, 15)]

        classifier = LogisticRegression()
        gridSearch = GridSearchCV(classifier,
                                  param_grid={'C': C_list},
                                  cv=k)
        gridSearch.fit(traindata, trainlabels)

        C_min = gridSearch.best_params_['C']
        min_err = 1 - gridSearch.best_score_

        # Do not change this sequence!
        return (C_min, min_err)

    def classify(self, traindata, trainlabels, testdata, testlabels):
        """ Train the best classifier selected above on the whole training set.

        Parameters:
        1. traindata            (Nt, d) numpy ndarray. The features in the training set.
        2. trainlabels          (Nt, ) numpy ndarray. The labels in the training set.
        3. testdata             (Nte, d) numpy ndarray. The features in the test set.
        4. testlabels           (Nte, ) numpy ndarray. The labels in the test set.

        Outputs:
        1. classifier           The classifier already trained on the training data. Use the best classifier that you choose.
        2. testError            Float. The reported test error. It should be less than 1.
        """
        # Put your code below

        best_C = 8
        best_gamma = 0.125

        classifier = SVC(C=best_C, gamma=best_gamma)
        classifier.fit(traindata, trainlabels)
        estimatedlabels = classifier.predict(testdata)
        testError = error(testlabels, estimatedlabels)

        # Do not change this sequence!
        return (classifier, testError)
