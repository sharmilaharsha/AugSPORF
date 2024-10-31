import multiprocessing
from warnings import warn

import numpy as np
from math import sqrt, floor
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.exceptions import DataConversionWarning

import pyfp


class rerfClassifier(BaseEstimator, ClassifierMixin):
    """A random forest classifier.
   
    """

    def __init__(
        self,
        projection_matrix="RerF",
        n_estimators=500,
        max_depth=None,
        min_samples_split=1,
        max_features="auto",
        feature_combinations=1.5,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        image_height=None,
        image_width=None,
        patch_height_max=None,
        patch_height_min=1,
        patch_width_max=None,
        patch_width_min=1,
    ):
        self.projection_matrix = projection_matrix
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.feature_combinations = feature_combinations
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state

        # s-rerf params
        self.image_height = image_height
        self.image_width = image_width
        self.patch_height_max = patch_height_max
        self.patch_height_min = patch_height_min
        self.patch_width_max = patch_width_max
        self.patch_width_min = patch_width_min

    def fit(self, X, y):
        """Fit estimator.
        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
            Input data.  Rows are observations and columns are features.
        y : array-like, 1D numpy array
            Labels

        Returns
        -------
        self : object

        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        num_features = X.shape[1]

        # Check that labels are starting from 0
        if min(y) != 0:
            raise ValueError("Labels must start from 0")

        # Check that labels are an inclusive range
        y_set = set(y)
        if len(y_set) - 1 < max(y_set):
            raise ValueError("Labels must be contiguous from [0, k-1]")

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        num_obs = len(y)

        # setup the forest's parameters
        self.forest_ = pyfp.fpForest()

        if self.projection_matrix == "Base":
            if self.oob_score:
                forestType = "rfBase"
            else:
                forestType = "binnedBase"
            self.method_to_use_ = None
        elif self.projection_matrix == "RerF":
            if self.oob_score:
                forestType = "rerf"
            else:
                forestType = "binnedBaseTern"
            self.method_to_use_ = 1
        elif self.projection_matrix == "S-RerF":
            if self.oob_score:
                warn(
                    "OOB is not currently implemented for the S-RerF"
                    " algorithm.  Continuing with oob_score = False.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.oob_score = False

            forestType = "binnedBaseTern"  # this should change
            self.method_to_use_ = 2
            # Check that image_height and image_width are divisors of
            # the num_features.  This is the most we can do to
            # prevent an invalid value being passed in.
            if (num_features % self.image_height) != 0:
                raise ValueError("Incorrect image_height given:")
            else:
                self.image_height_ = self.image_height
            self.forest_.setParameter("imageHeight", self.image_height_)
            if (num_features % self.image_width) != 0:
                raise ValueError("Incorrect image_width given:")
            else:
                self.image_width_ = self.image_width
            self.forest_.setParameter("imageWidth", self.image_width_)
            # If patch_height_{min, max} and patch_width_{min, max} are
            # not set by the user, set them to defaults.
            if self.patch_height_max is None:
                self.patch_height_max_ = max(2, floor(sqrt(self.image_height_)))
            else:
                self.patch_height_max_ = self.patch_height_max
            if self.patch_width_max is None:
                self.patch_width_max_ = max(2, floor(sqrt(self.image_width_)))
            else:
                self.patch_width_max_ = self.patch_width_max
            if 1 <= self.patch_height_min <= self.patch_height_max_:
                self.patch_height_min_ = self.patch_height_min
            else:
                raise ValueError("Incorrect patch_height_min")
            if 1 <= self.patch_width_min <= self.patch_width_max_:
                self.patch_width_min_ = self.patch_width_min
            else:
                raise ValueError("Incorrect patch_width_min")
            self.forest_.setParameter("patchHeightMax", self.patch_height_max_)
            self.forest_.setParameter("patchHeightMin", self.patch_height_min_)
            self.forest_.setParameter("patchWidthMax", self.patch_width_max_)
            self.forest_.setParameter("patchWidthMin", self.patch_width_min_)
        else:
            raise ValueError("Incorrect projection matrix")
        self.forest_.setParameter("forestType", forestType)

        if self.method_to_use_ is not None:
            self.forest_.setParameter("methodToUse", self.method_to_use_)

        self.forest_.setParameter("numTreesInForest", self.n_estimators)

        # if max_depth is not set, C++ sets to maximum integer size
        if self.max_depth is not None:
            self.forest_.setParameter("maxDepth", self.max_depth)

        self.forest_.setParameter("minParent", self.min_samples_split)

        self.forest_.setParameter("mtryMult", self.feature_combinations)

        if self.n_jobs is None:
            self.n_jobs_ = 1
        elif self.n_jobs == -1:
            self.n_jobs_ = multiprocessing.cpu_count()
        else:
            self.n_jobs_ = self.n_jobs
        self.forest_.setParameter("numCores", self.n_jobs_)

        if self.random_state is None:
            self.random_state_ = np.random.randint(1, 1000000)
        else:
            self.random_state_ = self.random_state
        self.forest_.setParameter("seed", self.random_state_)

        # need to set mtry here (using max_features and calc num_features):
        if self.max_features in ("auto", "sqrt"):
            self.mtry_ = int(num_features ** (1 / 2))
        elif self.max_features is None:
            self.mtry_ = num_features
        elif self.max_features == "log2":
            self.mtry_ = int(np.log2(num_features))
        elif isinstance(self.max_features, int):
            self.mtry_ = self.max_features
        elif isinstance(self.max_features, float) and self.max_features > 0:
            self.mtry_ = int(self.max_features * num_features)
        else:
            raise ValueError("max_features has unexpected value")
        self.forest_.setParameter("mtry", self.mtry_)

        # Explicitly setting for numpy input
        self.forest_.setParameter("useRowMajor", 1)

        self.forest_._growForestnumpy(X, y, num_obs, num_features)

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y

        if self.oob_score:
            self.oob_score_ = self.forest_._report_OOB()

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array_like of shape [nsamples, n_features]
            The input samples.  If more than 1 row, run multiple predictions.
        Returns
        -------
        y : int, list of int
            Returns the class of prediction (int) or predictions (list)
            depending on input parameters.
        """

        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        if X.ndim == 1:
            predictions = self.forest_._predict(X.tolist())
        else:
            predictions = self.forest_._predict_numpy(X)
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities for X.
        The predicted class probabilities of an input sample are computed as
        the mean predicted class of the trees in the forest.

        Parameters
        ----------
        X : array_like of shape [nsamples, n_features]
            The input samples.  If more than 1 row, run multiple predictions.
        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """

        # Check is fit had been called
        check_is_fitted(self, ["X_", "y_"])

        # Input validation
        X = check_array(X)

        if X.ndim == 1:
            y = self.forest_._predict_post(X.tolist())
            y_prob = [p / sum(y) for p in y]
        else:
            y = self.forest_._predict_post_array(X)
            y_arr = np.asarray(y)
            y_prob = y_arr / y_arr.sum(1)[:, None]
        return y_prob

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.
        The predicted class log-probabilities of an input sample is computed as
        the log of the mean predicted class probabilities of the trees in the
        forest.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        p : array of shape = [n_samples, n_classes], or a list of n_outputs
            such arrays if n_outputs > 1.
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute `classes_`.
        """
        proba = self.predict_proba(X)

        return np.log(proba)
