import warnings
from abc import ABCMeta, abstractmethod
from numbers import Real

import numpy as np
from scipy.special import logsumexp

from sklearn.base import ClassifierMixin, BaseEstimator, _fit_context
from sklearn.preprocessing import label_binarize, LabelBinarizer
from sklearn.utils import Interval
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn.utils.validation import check_is_fitted, _check_sample_weight, check_non_negative

__all__ = [
    "MultinomialNB",
]
class BaseNB(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
    """朴素贝叶斯估计器的抽象基类"""

    @abstractmethod
    def joint_log_likelihood(self, X):
        """计算 X 的非标准化后验对数概率

        即``log P(c) + log P(x|c)`` 对于 X 的所有行 x，作为形状为 (n_samples, n_classes) 的类似数组。

        公共方法 predict、predict_proba、predict_log_proba 和 predict_joint_log_proba 在将输入传递给 _joint_log_likelihood 之前，
        通过 _check_X 进行检查。术语“联合对数似然”与“联合对数概率”可互换使用。
        """

    def predict_joint_log_proba(self, X):
        """Return joint log probability estimates for the test vector X.

        For each row x of X and class y, the joint log probability is given by
        ``log P(x, y) = log P(y) + log P(x|y),``
        where ``log P(y)`` is the class prior probability and ``log P(x|y)`` is
        the class-conditional probability.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples, n_classes)
            Returns the joint log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        return self.joint_log_likelihood(X)

    def predict(self, X):
        """
        Perform classification on an array of test vectors X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : ndarray of shape (n_samples,)
            Predicted target values for X.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self.joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def predict_log_proba(self, X):
        """
        Return log-probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        check_is_fitted(self)
        X = self._check_X(X)
        jll = self.joint_log_likelihood(X)
        # normalize by P(x) = P(f_1, ..., f_n)
        log_prob_x = logsumexp(jll, axis=1)
        return jll - np.atleast_2d(log_prob_x).T

    def predict_proba(self, X):
        """
        Return probability estimates for the test vector X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        C : array-like of shape (n_samples, n_classes)
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.
        """
        return np.exp(self.predict_log_proba(X))


class BaseDiscreteNB(BaseNB):
    """Abstract base class for naive Bayes on discrete/categorical data

    Any estimator based on this class should provide:

    __init__
    joint_log_likelihood(X) as per _BaseNB
    _update_feature_log_prob(alpha)
    _count(X, Y)
    """

    _parameter_constraints: dict = {
        "alpha": [Interval(Real, 0, None, closed="left"), "array-like"],
        "fit_prior": ["boolean"],
        "class_prior": ["array-like", None],
        "force_alpha": ["boolean"],
    }

    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None, force_alpha=True):
        self.alpha = alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.force_alpha = force_alpha

    @abstractmethod
    def _count(self, X, Y):
        """Update counts that are used to calculate probabilities.

        The counts make up a sufficient statistic extracted from the data.
        Accordingly, this method is called each time `fit` or `partial_fit`
        update the model. `class_count_` and `feature_count_` must be updated
        here along with any model specific counts.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.
        Y : ndarray of shape (n_samples, n_classes)
            Binarized class labels.
        """

    @abstractmethod
    def _update_feature_log_prob(self, alpha):
        """Update feature log probabilities based on counts.

        This method is called each time `fit` or `partial_fit` update the
        model.

        Parameters
        ----------
        alpha : float
            smoothing parameter. See :meth:`_check_alpha`.
        """

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        return self._validate_data(X, accept_sparse="csr", reset=False)

    def _check_X_y(self, X, y, reset=True):
        """Validate X and y in fit methods."""
        return self._validate_data(X, y, accept_sparse="csr", reset=reset)

    def _update_class_log_prior(self, class_prior=None):
        n_classes = len(self.classes_)
        if class_prior is not None:
            if len(class_prior) != n_classes:
                raise ValueError("Number of priors must match number of classes.")
            self.class_log_prior_ = np.log(class_prior)
        elif self.fit_prior:
            with warnings.catch_warnings():
                # silence the warning when count is 0 because class was not yet
                # observed
                warnings.simplefilter("ignore", RuntimeWarning)
                log_class_count = np.log(self.class_count_)

            # empirical prior, with sample_weight taken into account
            self.class_log_prior_ = log_class_count - np.log(self.class_count_.sum())
        else:
            self.class_log_prior_ = np.full(n_classes, -np.log(n_classes))

    def _check_alpha(self):
        alpha = (
            np.asarray(self.alpha) if not isinstance(self.alpha, Real) else self.alpha
        )
        alpha_min = np.min(alpha)
        if isinstance(alpha, np.ndarray):
            if not alpha.shape[0] == self.n_features_in_:
                raise ValueError(
                    "When alpha is an array, it should contains `n_features`. "
                    f"Got {alpha.shape[0]} elements instead of {self.n_features_in_}."
                )
            # check that all alpha are positive
            if alpha_min < 0:
                raise ValueError("All values in alpha must be greater than 0.")
        alpha_lower_bound = 1e-10
        if alpha_min < alpha_lower_bound and not self.force_alpha:
            warnings.warn(
                "alpha too small will result in numeric errors, setting alpha ="
                f" {alpha_lower_bound:.1e}. Use `force_alpha=True` to keep alpha"
                " unchanged."
            )
            return np.maximum(alpha, alpha_lower_bound)
        return alpha

    @_fit_context(prefer_skip_nested_validation=True)
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        first_call = not hasattr(self, "classes_")

        X, y = self._check_X_y(X, y, reset=first_call)
        _, n_features = X.shape

        if _check_partial_fit_first_call(self, classes):
            n_classes = len(classes)
            self._init_counters(n_classes, n_features)

        Y = label_binarize(y, classes=self.classes_)
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        if X.shape[0] != Y.shape[0]:
            msg = "X.shape[0]=%d and y.shape[0]=%d are incompatible."
            raise ValueError(msg % (X.shape[0], y.shape[0]))

        Y = Y.astype(np.float64, copy=False)
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        self._count(X, Y)

        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        X, y = self._check_X_y(X, y)
        _, n_features = X.shape

        labelbin = LabelBinarizer()
        Y = labelbin.fit_transform(y)
        self.classes_ = labelbin.classes_
        if Y.shape[1] == 1:
            if len(self.classes_) == 2:
                Y = np.concatenate((1 - Y, Y), axis=1)
            else:  # degenerate case: just one class
                Y = np.ones_like(Y)

        # LabelBinarizer().fit_transform() returns arrays with dtype=np.int64.
        # We convert it to np.float64 to support sample_weight consistently;
        # this means we also don't have to cast X to floating point
        if sample_weight is not None:
            Y = Y.astype(np.float64, copy=False)
            sample_weight = _check_sample_weight(sample_weight, X)
            sample_weight = np.atleast_2d(sample_weight)
            Y *= sample_weight.T

        class_prior = self.class_prior

        # Count raw events from data before updating the class log prior
        # and feature log probas
        n_classes = Y.shape[1]
        self._init_counters(n_classes, n_features)
        self._count(X, Y)
        alpha = self._check_alpha()
        self._update_feature_log_prob(alpha)
        self._update_class_log_prior(class_prior=class_prior)
        return self

    def _init_counters(self, n_classes, n_features):
        self.class_count_ = np.zeros(n_classes, dtype=np.float64)
        self.feature_count_ = np.zeros((n_classes, n_features), dtype=np.float64)

    def _more_tags(self):
        return {"poor_score": True}


class MultinomialNB(BaseDiscreteNB):
    def __init__(
        self, *, alpha=1.0, force_alpha=True, fit_prior=True, class_prior=None
    ):
        super().__init__(
            alpha=alpha,
            fit_prior=fit_prior,
            class_prior=class_prior,
            force_alpha=force_alpha,
        )

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _count(self, X, Y):
        """Count and smooth feature occurrences."""
        check_non_negative(X, "MultinomialNB (input X)")
        self.feature_count_ += safe_sparse_dot(Y.T, X)
        self.class_count_ += Y.sum(axis=0)

    def _update_feature_log_prob(self, alpha):
        """Apply smoothing to raw counts and recompute log probabilities"""
        smoothed_fc = self.feature_count_ + alpha
        smoothed_cc = smoothed_fc.sum(axis=1)

        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(
            smoothed_cc.reshape(-1, 1)
        )

    def joint_log_likelihood(self, X):
        """Calculate the posterior log probability of the samples X"""
        return safe_sparse_dot(X, self.feature_log_prob_.T) + self.class_log_prior_
