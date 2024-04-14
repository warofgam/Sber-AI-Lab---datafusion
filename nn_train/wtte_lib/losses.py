from math import log
import torch
from torch import nn

EPS = torch.finfo(torch.float32).eps


def log_likelihood_discrete(tte, uncensored, alpha, beta, epsilon=EPS):
    """Discrete version of log likelihood for the Weibull TTE loss function.

    Parameters
    ----------
    tte : torch.tensor
        tensor of each subject's time to event at each time step.
    uncensored : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    alpha : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    beta : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.

    Examples
    --------
    FIXME: Add docs.

    """
    print(alpha, beta)
    hazard_0 = torch.pow((tte + epsilon) / alpha, beta)
    hazard_1 = torch.pow((tte + 1.0) / alpha, beta)
    return uncensored * torch.log(torch.exp(hazard_1 - hazard_0) - (1.0 - epsilon)) - hazard_1


def log_likelihood_continuous(tte, uncensored, alpha, beta, epsilon=EPS):
    """Continuous version of log likelihood for the Weibull TTE loss function.

    Parameters
    ----------
    tte : torch.tensor
        tensor of each subject's time to event at each time step.
    uncensored : torch.tensor
        tensor indicating whether each data point is censored (0) or
        not (1)
    alpha : torch.tensor
        Estimated Weibull distribution scale parameter per subject,
        per time step.
    beta : torch.tensor
        Estimated Weibull distribution shape parameter per subject,
        per time step.

    Examples
    --------
    FIXME: Add docs.

    """
    y_a = (tte + epsilon) / alpha
    return uncensored * (torch.log(beta) + beta * torch.log(y_a)) - torch.pow(y_a, beta)


def weibull_censored_nll_loss(
    inputs: torch.tensor,
    targets: torch.tensor,
    discrete: bool = False,
    reduction: str = "mean",
    clip_prob=1e-6,
):
    """Compute the loss.

    Compute the Weibull censored negative log-likelihood loss for
    forward propagation.

    Parameters
    ----------
    inputs : torch.tensor
        Estimated Weibull distribution scale (alpha) and shape (beta)
        parameter per subject, per time step.
    targets : torch.tensor
        Tensor of each subject's time to event at each time step and
        flag indicating whether each data point is censored (0) or
        not (1)
    clip_prob: float
        Clip likelihood to to [log(clip_prob),log(1-clip_prob)]
    """
    alpha = inputs[..., 0]
    beta = inputs[..., 1]
    tte = targets[..., 0]
    uncensored = targets[..., 1]
    reducer = {"mean": torch.mean, "sum": torch.sum}.get(reduction)
    likelihood = log_likelihood_discrete if discrete else log_likelihood_continuous
    log_likelihoods = likelihood(tte, uncensored, alpha, beta)
    if reducer:
        log_likelihoods = reducer(log_likelihoods, dim=-1)
    if clip_prob is not None:
        log_likelihoods = torch.clamp(log_likelihoods, log(clip_prob), log(1 - clip_prob))
    return -1.0 * log_likelihoods


class WeibullCensoredNLLLoss(nn.Module):
    """A negative log-likelihood loss function for Weibull distribution
    parameter estimation with right censoring.
    """

    def __init__(self, discrete: bool = False, reduction: str = "mean", clip_prob=1e-6):
        """Constructor.

        Construct the Weibull censored negative log-likelihood loss object.

        Parameters
        ----------
        discrete : bool
             Specifies whether to use the discrete (True) or continuous (False)
             variant of the Weibull distribution for parameter estimation.
        reduction : str
             Specifies the reduction to apply to the output: 'none' |
             'mean' | 'sum'. 'none': no reduction will be applied,
             'mean': the weighted mean of the output is taken, 'sum':
             the output will be summed. Note: size_average and reduce
             are in the process of being deprecated, and in the
             meantime, specifying either of those two args will
             override reduction. Default: 'mean'
        clip_prob: float
             Clip likelihood to to [log(clip_prob),log(1-clip_prob)]

        """
        super().__init__()
        self.discrete = discrete
        self.reduction = reduction
        self.clip_prob = clip_prob

    def forward(
        self,
        inputs: torch.tensor,
        target: torch.tensor,
    ):
        """Compute the loss.

        Compute the Weibull censored negative log-likelihood loss for
        forward propagation.

        Parameters
        ----------
        inputs : torch.tensor
            Estimated Weibull distribution scale (alpha) and shape (beta)
            parameter per subject, per time step.
        targets : torch.tensor
            Tensor of each subject's time to event at each time step and
            flag indicating whether each data point is censored (0) or
            not (1)
        """
        return weibull_censored_nll_loss(
            inputs, target, self.discrete, self.reduction, self.clip_prob
        )


class WeibullActivation(nn.Module):
    """Layer that initializes, activates and regularizes alpha and beta parameters of
    a Weibull distribution."""

    def __init__(self, init_alpha: float = 1.0, max_beta: float = 5.0, epsilon: float = EPS):
        super().__init__()
        self.init_alpha = torch.tensor(init_alpha)
        self.max_beta = torch.tensor(max_beta)
        self.epsilon = epsilon

    def forward(self, x: torch.tensor):
        """Compute the activation function.

        Parameters
        ----------
        x : torch.tensor
            An input tensor with innermost dimension = 2 ([alpha,
            beta])
        """

        alpha = x[..., 0]
        beta = x[..., 1]

        alpha = self.init_alpha * torch.exp(alpha)

        if self.max_beta > 1.05:
            shift = torch.log(self.max_beta - 1.0)
            beta = beta - shift

        beta = self.max_beta * torch.clamp(
            torch.sigmoid(beta), min=self.epsilon, max=1.0 - self.epsilon
        )

        return torch.stack([alpha, beta], axis=-1)