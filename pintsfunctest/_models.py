#
# This file is part of PINTS (https://github.com/pints-team/pints/) which is
# released under the BSD 3-clause license. See accompanying LICENSE.md for
# copyright notice and full license details.
#

import numpy as np
import pints
from scipy.special import logsumexp


class MixtureModel(pints.LogPDF):
    r"""
    A mixture model of probablity density functions.

    A mixture model takes a list of :class:`pints.LogPDF` instances and
    a list of weights for each log-pdf.

    Denoting the PDFs of the log-pdfs by :math:`p = [p_1, p_2, \ldots , p_n]`
    and the weights by :math:`w = [w_1, w_2, \ldots, w_n]` then the PDF of the
    mixture model is given by

    .. math::
        p _{\text{mix}} = \sum _{i=1}^n w_i p_i .

    The parameters of the individual PDFs will be concatenated in order of
    their appearance to form the parameters of the mixture model.

    .. warning
        A mixture model only leads to correct results when the individual
        :class:`pints.LogPDF` are normalised. Not all pdfs in pints
        are normalised!

    :param log_pdfs: A list of :class:`pints.LogPDF` instances.
    :type log_pdfs: List[pints.LogPDF]
    :param weights: A list of weights, one for each log-pdf. The weights do not
        have to be normalised.
    :type weights: List[float]
    """
    def __init__(self, log_pdfs, weights):
        super(MixtureModel, self).__init__()

        # Check inputs
        log_pdfs = list(log_pdfs)
        for log_pdf in log_pdfs:
            if not isinstance(log_pdf, pints.LogPDF):
                raise TypeError(
                    'The log-pdfs must be instances of pints.LogPDF.')

        weights = list(weights)
        weights = [float(w) for w in weights]

        # Save log-pdfs, and log-weights
        self._log_pdfs = log_pdfs
        self._log_weights = np.log(weights)

        # Save number of parameters for each PDF
        n_pdf_params = [p.n_parameters() for p in log_pdfs]
        self._n_pdf_params = np.cumsum(n_pdf_params)
        self._n_parameters = np.sum(n_pdf_params)

    def __call__(self, parameters):
        if len(parameters) != self._n_parameters:
            raise ValueError(
                'The number of parameters does not match n_parameters.')

        # Compute log-likelihood score for each PDF and compute weighted sum
        start = 0
        scores = []
        for _id, log_pdf in enumerate(self._log_pdfs):
            # Get number of parameters
            end = self._n_pdf_params[_id]

            # Add weighted log-score
            # Note:
            # 1. On log_scale log-weight is added
            # 2. Normalisation of weights only leads to constant factor on
            #    log-scale
            scores.append(
                self._log_weights[_id] + log_pdf(parameters[start:end]))

            # Shift start of the parameters
            start += end

        # Compute final score by transforming scores back to linear scale, add
        # them and take logarithm again
        score = logsumexp(scores)

        return score

    def compute_kullback_leibler_divergence(
            self, parameters, chain, n_samples):
        r"""
        Approximately computes the Kullback-Leibler divergence and its
        estimated error.

        The Kullback-Leibler divergence is defined as

        .. math::
            D(f || g) = \mathbb{E}\left[ \log \frac{f(x)}{g(x)} \right] ,

        where the expectation is taken w.r.t. :math:`f(x)`. We approximate the
        divergence by drawing :math:`n` i.i.d. samples from :math:`f(x)` and
        compute

        .. math::
            \hat{D}(f || g) \approx \frac{1}{n}
            \sum _{i=1}^n \log \frac{f(x_i)}{g(x_i)}.

        Note that the draws and the score :math:`f(x)` can be computed exactly
        from the mixture model, while :math:`g(x)` is computed by normalising
        the chain samples.

        The variance of the Kullback-Leibler divergence estimate error can be
        estimated with

        .. math::
            \text(Var)\left[ \hat{D}(f || g)\right] =
            \frac{1}{n} \text(Var)\left[ \log \frac{f(x)}{g(x)} \right]
        """

