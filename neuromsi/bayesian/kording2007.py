#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

"""Bayesian causal inference model for multisensory integration."""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np

from ..integrator import Integrator

# =============================================================================
# CONSTANTS
# =============================================================================

STRATEGIES = ("selection", "averaging", "matching")
DIMENSIONS = ("space", "time")


# =============================================================================
# CLASS
# =============================================================================


class Kording2007(Integrator):
    r"""
    Bayesian Causal Inference model for multisensory integration.

    This model based on Kording et al. (2007) uses Bayesian principles to
    infer whether two unimodal signals come from a common cause or
    different causes. It combines two unisensory signals and evaluates
    the probability of a common cause based on the observed signals.

    This implementation is inspired on the Matlab version of the BCI Toolbox
    (Zhu, Beierholm & Shams, 2024).

    The model is decoupled from the stimuli: the Backend encodes each
    stimulus as a :class:`~neuromsi.signal.BayesianSignal` (the stimulus
    variance plus the unisensory measurement drawn from it) and passes
    those signals to :meth:`execute`. Everything that is not an
    observation of the world lives here: the priors, the causal prior,
    the integration strategy and the estimators.

    References
    ----------
    :cite:p:`kording2007causal`
    :cite:p:`zhu2024bci`

    Notes
    -----
    The Bayesian Causal Inference model uses the following formulation:

    .. math::
       p(C \mid x_{1}, x_{2}) = \frac{p(x_{1}, x_{2} \mid C) \, p(C)}{p(x_{1},
       x_{2})}

    where :math:`x_{1}` and :math:`x_{2}` are two unimodal signals, and
    :math:`C` is a binary variable representing the number of causes in the
    environment.

    The posterior probability of the signals having a single cause in the
    environment is defined as follows:

    .. math::
        p(C = 1 \mid x_{1}, x_{2}) = \frac{p(x_{1}, x_{2} \mid C=1) \,
        p(C=1)}{p(x_{1}, x_{2} \mid C=1) \, p(C=1) + p(x_{1}, x_{2} \mid
        C=2) \, (1 - p(C=1))}

    The likelihood is computed as:

    .. math::
        p(x_{1}, x_{2} \mid C = 1) = \int\int p(x_{1}, x_{2} \mid X) p(X) dX

    Here, :math:`p(C = 1)` is the prior probability of a common cause
    (default is 0.5). :math:`X` denotes the attributes of the stimuli
    (e.g., distance), which are then represented in the nervous system as
    :math:`x_{1}` and :math:`x_{2}`.

    These equations show that the inference of a common cause of
    two unisensory signals is computed by combining the likelihood and prior
    of signals having a common cause. A higher likelihood occurs if the
    two unisensory signals are similar, which in turn increases the
    probability of inferring that the signals have a common cause.

    The generative model assumes that every stimulus location is drawn from
    a Gaussian prior with mean ``prior_mu`` and standard deviation
    ``prior_sigma``, and that each modality measures it with Gaussian
    noise (whose standard deviation is the ``sigma`` of the stimulus). The
    model then estimates the location of each stimulus with one of three
    strategies, which differ in how they use the posterior probability of
    a common cause :math:`p(C = 1 \mid x_{1}, x_{2})`:

    - ``"selection"``: fuses the signals if that probability is above 0.5,
      and keeps them separate otherwise.
    - ``"averaging"``: weights the fused and the separate estimates by
      that probability.
    - ``"matching"``: fuses the signals with a probability equal to it
      (a random threshold is drawn in every simulation).

    The result of a run is not a single estimate: the model simulates
    ``n_sim`` observers and reports the distribution of their estimates
    along the grid of positions (or of time).

    """

    #: This model consumes Bayesian signals, not neural ones.
    model_type = "Bayesian"

    def __init__(
        self,
        *,
        n_sim=1,
        mode0="auditory",
        mode1="visual",
        p_common=0.5,
        prior_sigma=20.0,
        prior_mu=0,
        strategy="averaging",
        noise=True,
        dimension="space",
    ):
        """
        Initializes the Kording 2007 model.

        Every parameter of the causal-inference mechanism belongs to the
        model. The stimulus descriptions (position, sigma) come from the
        Stimulus objects, and the grids, the seed and the causes
        parameters belong to the Backend.

        Parameters
        ----------
        n_sim : int
            Number of simulations to run. Each one is an independent
            observer, with its own noisy measurement of the stimuli. More
            simulations give smoother distributions.
        mode0 : str
            The name for the first sensory modality (e.g., "auditory").
        mode1 : str
            The name for the second sensory modality (e.g., "visual").
        p_common : float
            The prior probability of a common cause.
        prior_sigma : float
            The standard deviation of the prior cause (how widely the
            stimuli are expected to be spread).
        prior_mu : float
            The mean of the prior cause (where the stimuli are expected
            to be).
        strategy : str
            The strategy for model selection
            ("selection", "averaging", "matching").
        noise : bool
            Whether the unisensory measurements carry noise. The Backend
            reads this value when it encodes the stimuli.
        dimension : str
            The dimension to run the model ("space" or "time"). It
            decides which axis of the result holds the distribution of
            estimates.

        Raises
        ------
        ValueError
            If ``mode0`` and ``mode1`` are the same, or if ``strategy``
            or ``dimension`` are not valid.
        """
        if mode0 == mode1:
            raise ValueError("mode0 and mode1 must name different modalities.")
        if strategy not in STRATEGIES:
            strategies = ", ".join(repr(s) for s in STRATEGIES)
            raise ValueError(
                f"strategy must be one of {strategies}. "
                f"Got {strategy!r}."
            )
        if dimension not in DIMENSIONS:
            dimensions = ", ".join(repr(d) for d in DIMENSIONS)
            raise ValueError(
                f"dimension must be one of {dimensions}. "
                f"Got {dimension!r}."
            )

        self.n_sim = n_sim
        self.mode0 = mode0
        self.mode1 = mode1
        self.p_common = p_common
        self.prior_sigma = prior_sigma
        self.prior_mu = prior_mu
        self.strategy = strategy
        self.noise = noise
        self.dimension = dimension

    # PROPERTIES ============================================================

    @property
    def model_parameters(self):
        """dict: Every parameter that defines this model.

        Reported by the Backend in the run parameters of the result, so
        a simulation can be reproduced from its NDResult.
        """
        return {
            "n_sim": self.n_sim,
            "mode0": self.mode0,
            "mode1": self.mode1,
            "p_common": self.p_common,
            "prior_sigma": self.prior_sigma,
            "prior_mu": self.prior_mu,
            "strategy": self.strategy,
            "noise": self.noise,
            "dimension": self.dimension,
        }

    # MODEL METHODS =========================================================

    def check_stimuli_compatible(self, modalities):
        """Require exactly one signal for each configured sensory modality.

        Parameters
        ----------
        modalities : iterable of str
            Modality names of the incoming signals.

        Raises
        ------
        ValueError
            If there are not exactly two signals, or if they are not one
            of ``mode0`` and one of ``mode1``.
        """
        modalities = list(modalities)
        if len(modalities) != 2:
            raise ValueError(
                "Kording2007 expects exactly 2 stimuli (one for each "
                f"sensory modality), but {len(modalities)} were received."
            )
        if set(modalities) != {self.mode0, self.mode1}:
            raise ValueError(
                f"Kording2007 expects one {self.mode0!r} and one "
                f"{self.mode1!r} signal. Received: {modalities}."
            )

    def unisensory_estimator(
        self,
        unisensory_var,
        unisensory_var_hat,
        prior_var,
        prior_mu,
        unisensory_input,
        n,
    ):
        r"""Estimates the unisensory posterior given the input and prior.

        This is the location estimate of one modality when it is treated
        as an independent cause: the precision-weighted average of the
        measurement and the prior mean,

        .. math::
            \hat{s} = \left( \frac{x}{\sigma^2}
            + \frac{\mu_p}{\sigma_p^2} \right) \hat{\sigma}^2

        with :math:`\hat{\sigma}^2 = (1/\sigma^2 + 1/\sigma_p^2)^{-1}`.

        Parameters
        ----------
        unisensory_var : float
            The variance of the sensory modality.
        unisensory_var_hat : float
            The estimated variance of the sensory modality.
        prior_var : float
            The prior variance of the cause.
        prior_mu : float
            The prior mean of the cause.
        unisensory_input : float or numpy.ndarray
            The input value for the sensory modality.
        n : int
            Number of simulations.

        Returns
        -------
        numpy.ndarray
            The posterior estimate of the sensory modality.
        """
        # Combine the measurement and prior in precision space (precision is
        # inverse variance). Multiplying their weighted sum by posterior
        # variance gives the posterior mean location estimate.
        unisensory_hat_ind = (
            (unisensory_input / unisensory_var)
            + (np.ones(n) * prior_mu) / prior_var
        ) * unisensory_var_hat

        return unisensory_hat_ind

    def multisensory_estimator(
        self,
        mode0_estimate,
        mode1_estimate,
        mode0_var,
        mode1_var,
        prior_var,
        prior_mu,
        multisensory_var,
        multisensory_var_hat,
        n,
        mode0_ind_var,
        mode1_ind_var,
        p_common,
        strategy,
        possible_locations,
        locations_res,
        random,
        mode0_input,
        mode1_input,
    ):
        """Estimates the multisensory posterior using Bayesian inference.

        It does three things for each simulated observer. First, it
        computes the estimate under a common cause (fusion of both
        measurements and the prior). Second, it weighs the evidence for a
        common cause against the evidence for independent causes, giving
        the posterior probability of a common cause. Third, it combines the
        fused and the separate estimates according to the chosen strategy.
        Finally, the estimates of all the observers are binned into
        probability distributions along the grid.

        Note that the ``"multi"`` distribution is the fused estimate
        itself, so it does not depend on the strategy: only the estimates
        of each modality do.

        Parameters
        ----------
        mode0_estimate : numpy.ndarray
            The unisensory estimate of the first modality.
        mode1_estimate : numpy.ndarray
            The unisensory estimate of the second modality.
        mode0_var : float
            The variance of the first modality.
        mode1_var : float
            The variance of the second modality.
        prior_var : float
            The prior variance of the cause.
        prior_mu : float
            The prior mean of the cause.
        multisensory_var : float
            The variance of the multisensory estimate.
        multisensory_var_hat : float
            The estimated variance of the multisensory estimate.
        n : int
            Number of simulations.
        mode0_ind_var : float
            The variance of the independent estimate of mode0.
        mode1_ind_var : float
            The variance of the independent estimate of mode1.
        p_common : float
            The prior probability of a common cause.
        strategy : str
            The strategy for model selection
            ("selection", "averaging", "matching").
        possible_locations : numpy.ndarray
            The bin centers used to build the predicted distributions.
        locations_res : float
            The resolution of ``possible_locations`` (``position_res`` for
            the spatial grid, ``time_res`` for the temporal one).
        random : numpy.random.Generator
            The random number generator (used by "matching").
        mode0_input : float or numpy.ndarray
            The input value for the first modality.
        mode1_input : float or numpy.ndarray
            The input value for the second modality.

        Returns
        -------
        dict
            A dictionary with keys:
            - "mode0": The posterior estimate of the first modality.
            - "mode1": The posterior estimate of the second modality.
            - "multi": The multisensory posterior estimate.
            - "pc": The posterior probability of a common cause (one value
              per simulation).
        """
        # A nonzero count means at least one modality is missing, so the
        # independent-cause estimates will be used below.
        single_stim = np.sum(np.isnan([mode0_input, mode1_input]))

        # Preserve the prior-informed unisensory estimates for the separate-
        # cause hypothesis, and compute a fused estimate for one shared cause.
        mode0_hat_ind = mode0_estimate
        mode1_hat_ind = mode1_estimate
        multisensory_hat = (
            (mode0_input / mode0_var)
            + (mode1_input / mode1_var)
            + (np.ones(n) * prior_mu) / prior_var
        ) * multisensory_var_hat

        # Squared deviations feed the Gaussian likelihoods (Kording et al.,
        # 2007, eqs. 2 and 3): dis_common
        # measures joint discrepancy under one latent location, while the
        # modality-specific terms measure each observation from the prior mean.
        dis_common = (
            (mode0_input - mode1_input) ** 2 * prior_var
            + (mode0_input - prior_mu) ** 2 * mode1_var
            + (mode1_input - prior_mu) ** 2 * mode0_var
        )
        dis_mode0 = (mode0_input - prior_mu) ** 2
        dis_mode1 = (mode1_input - prior_mu) ** 2

        # Evaluate evidence under each hypothesis (the likelihood of both
        # measurements given a common cause, and given independent ones). Under independent causes,
        # the joint likelihood is the product of the two modality likelihoods.
        likelihood_common = np.exp(-dis_common / (2 * multisensory_var)) / (
            2 * np.pi * np.sqrt(multisensory_var)
        )
        likelihood_mode0 = np.exp(
            -dis_mode0 / (2 * mode0_ind_var)
        ) / np.sqrt(2 * np.pi * mode0_ind_var)
        likelihood_mode1 = np.exp(
            -dis_mode1 / (2 * mode1_ind_var)
        ) / np.sqrt(2 * np.pi * mode1_ind_var)
        likelihood_indep = likelihood_mode0 * likelihood_mode1
        # Weight each likelihood by its prior, then normalize the common-cause
        # weight to obtain P(common cause | both observations).
        post_common = likelihood_common * p_common
        post_indep = likelihood_indep * (1 - p_common)
        pc = post_common / (post_common + post_indep)

        # If one observation is absent, there is nothing to fuse; retain the
        # separate estimates for the modalities.
        if single_stim:
            mode0_hat = mode0_hat_ind
            mode1_hat = mode1_hat_ind
        else:
            # Apply the chosen rule for combining the shared-cause estimate
            # with the separate-cause estimate of each modality.
            # Selection makes a hard choice at posterior probability 0.5.
            if strategy == "selection":
                mode0_hat = (pc > 0.5) * multisensory_hat + (
                    pc <= 0.5
                ) * mode0_hat_ind
                mode1_hat = (pc > 0.5) * multisensory_hat + (
                    pc <= 0.5
                ) * mode1_hat_ind
            # Averaging weights both estimates by their posterior probability.
            elif strategy == "averaging":
                mode0_hat = (pc) * multisensory_hat + (
                    1 - pc
                ) * mode0_hat_ind
                mode1_hat = (pc) * multisensory_hat + (
                    1 - pc
                ) * mode1_hat_ind
            # Matching draws a random threshold per trial and fuses when the
            # common-cause probability exceeds that threshold.
            elif strategy == "matching":
                match = 1 - random.random(n)
                mode0_hat = (pc > match) * multisensory_hat + (
                    pc <= match
                ) * mode0_hat_ind
                mode1_hat = (pc > match) * multisensory_hat + (
                    pc <= match
                ) * mode1_hat_ind

        # Bin simulated estimates on the location grid, then normalize counts
        # to turn them into predicted probability distributions.
        # Each grid point is the center of a bin of width `locations_res`, so
        # the bin edges lie half a step to either side of the points.
        edges = possible_locations - locations_res / 2
        edges = np.append(edges, edges[-1] + locations_res)

        mode0_estimates = np.histogram(mode0_hat, edges)[0]
        mode1_estimates = np.histogram(mode1_hat, edges)[0]
        multisensory_estimates = np.histogram(multisensory_hat, edges)[0]

        pred_mode0 = mode0_estimates / np.sum(mode0_estimates, axis=0)
        pred_mode1 = mode1_estimates / np.sum(mode1_estimates, axis=0)
        pred_multi = multisensory_estimates / np.sum(
            multisensory_estimates, axis=0
        )

        res = {
            "mode0": pred_mode0,
            "mode1": pred_mode1,
            "multi": pred_multi,
            "pc": pc,
        }

        return res

    def integrate(
        self,
        signals,
        *,
        time_range,
        time_res,
        position_range,
        position_res,
        random,
    ):
        """Runs the causal inference over a pair of Bayesian signals.

        The model computes a single distribution of estimates for each
        modality. Since it has no dynamics, that distribution fills only
        one row (``dimension="space"``) or one column
        (``dimension="time"``) of the response; the rest stays at zero.

        Parameters
        ----------
        signals : dict
            Bayesian signals keyed by modality. Each one carries the
            stimulus variance (``values["var"]``) and the unisensory
            measurement (``values["input"]``) already computed by the
            stimulus.
        time_range : (float, float)
            The time grid of the simulation, given by the Backend. It
            defines the grid of the estimates when ``dimension="time"``.
        time_res : float
            Temporal resolution of the simulation.
        position_range : (float, float)
            The position grid of the simulation, given by the Backend. It
            defines the grid of the estimates when ``dimension="space"``.
        position_res : float
            Spatial resolution of the simulation.
        random : numpy.random.Generator
            The random number generator of the Backend.

        Returns
        -------
        tuple
            A tuple containing:
            - dict: Response dictionary with "mode0", "mode1" and "multi".
            - dict: Extra information with "mean_p_common_cause" (the
            average over the simulations) and "p_common_cause" (the value
            of each simulation).
        """
        mode0_signal = signals[self.mode0]
        mode1_signal = signals[self.mode1]

        # Build the grids of the result out of the Backend parameters and
        # allocate the two-dimensional arrays that hold the predictions.
        pos_range = np.arange(
            position_range[0], position_range[1], position_res
        )
        t_range = np.arange(time_range[0], time_range[1], time_res)

        shape = (t_range.size, pos_range.size)
        mode0_res = np.zeros(shape)
        mode1_res = np.zeros(shape)
        multi_res = np.zeros(shape)

        # Select the grid whose values will be used as histogram bin centers,
        # and the resolution that goes with it.
        if self.dimension == "space":
            possible_locations = pos_range
            locations_res = position_res
        else:
            possible_locations = t_range
            locations_res = time_res

        # The signals already carry the variance of each stimulus (the
        # square of its sigma); the prior is given as a standard deviation,
        # so it is squared here. The Bayesian calculations below work with
        # variances (and therefore with precisions, their inverses).
        mode0_var = mode0_signal.values["var"]
        mode1_var = mode1_signal.values["var"]
        prior_var = np.square(self.prior_sigma)

        # Set variances for the shared-cause likelihood and the two
        # independent-cause likelihoods: for independent causes each
        # modality is spread by its own noise plus the prior; for a common
        # cause the joint spread is the product-form expression of the paper.
        multisensory_var = (
            mode0_var * mode1_var
            + mode0_var * prior_var
            + mode1_var * prior_var
        )

        mode0_ind_var = mode0_var + prior_var
        mode1_ind_var = mode1_var + prior_var

        # Posterior variances are reciprocals of summed precisions. They
        # determine the weights in the posterior location estimates.
        multisensory_var_hat = 1 / (
            1 / mode0_var + 1 / mode1_var + 1 / prior_var
        )

        mode0_var_hat = 1 / (1 / mode0_var + 1 / prior_var)

        mode1_var_hat = 1 / (1 / mode1_var + 1 / prior_var)

        # The unisensory measurements were drawn when the Backend encoded
        # each stimulus, so they arrive with the signals.
        mode0_input = mode0_signal.values["input"]
        mode1_input = mode1_signal.values["input"]

        # Estimate each modality separately by combining its measurement
        # with the prior over stimulus location.
        mode0_estimate = self.unisensory_estimator(
            mode0_var,
            mode0_var_hat,
            prior_var,
            self.prior_mu,
            mode0_input,
            self.n_sim,
        )

        mode1_estimate = self.unisensory_estimator(
            mode1_var,
            mode1_var_hat,
            prior_var,
            self.prior_mu,
            mode1_input,
            self.n_sim,
        )

        # Compute causal evidence and predicted estimates using the requested
        # integration strategy (selection, averaging, or matching).
        estimate = self.multisensory_estimator(
            mode0_estimate,
            mode1_estimate,
            mode0_var,
            mode1_var,
            prior_var,
            self.prior_mu,
            multisensory_var,
            multisensory_var_hat,
            self.n_sim,
            mode0_ind_var,
            mode1_ind_var,
            self.p_common,
            self.strategy,
            possible_locations,
            locations_res,
            random,
            mode0_input,
            mode1_input,
        )

        # Insert the predicted distributions along the axis corresponding to
        # the chosen dimension; the other axis remains zero-filled.
        if self.dimension == "space":
            mode0_res[0, :] = estimate["mode0"]
            mode1_res[0, :] = estimate["mode1"]
            multi_res[0, :] = estimate["multi"]

        else:
            mode0_res[:, 0] = estimate["mode0"]
            mode1_res[:, 0] = estimate["mode1"]
            multi_res[:, 0] = estimate["multi"]

        # Return the distributions as the response and causal probabilities
        # as supplementary information.
        response = {
            self.mode0: mode0_res,
            self.mode1: mode1_res,
            "multi": multi_res,
        }

        extra = {
            "mean_p_common_cause": np.average(estimate["pc"]),
            "p_common_cause": estimate["pc"],
        }

        return response, extra

    def calculate_causes(self, mean_p_common_cause, causes_kind, **kwargs):
        """Calculates the causes of the stimuli.

        The number of causes is read from the probability of a common
        cause: more than 0.5 means one cause, otherwise two.

        Parameters
        ----------
        mean_p_common_cause : float or numpy.ndarray
            The average probability of a common cause across simulations.
        causes_kind : str
            The type of cause to calculate ("count" or "prob").
        kwargs : dict
            The remaining run extras reported by the Backend (they do not
            apply to this model and are ignored).

        Returns
        -------
        int or float or None
            The number of causes if `causes_kind` is "count",
            the probability of a unique cause if `causes_kind` is "prob",
            or None if no valid cause type is specified.
        """
        # Choose whether to report a hard cause count, the common-cause
        # probability itself, or None for an unsupported output type.
        if causes_kind == "count":
            # evaluate the probability of perceiving a common cause
            if mean_p_common_cause > 0.5:
                causes = 1
            else:
                causes = 2

        elif causes_kind == "prob":
            # If calculating the probability of a unique cause,
            # assign the probability
            # of perceiving a common cause
            causes = mean_p_common_cause
        else:
            # If no valid cause type is specified, assign None
            causes = None
        return causes
