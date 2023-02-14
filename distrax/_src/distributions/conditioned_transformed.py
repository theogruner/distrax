import functools
import operator
from typing import Union, Tuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from distrax._src.bijectors import bijector as bjct_base
from distrax._src.distributions import distribution as dist_base
from distrax._src.distributions.transformed import Transformed
from distrax._src.distributions.distribution import convert_seed_and_sample_shape

from tensorflow_probability.substrates import jax as tfp


tfd = tfp.distributions

PRNGKey = dist_base.PRNGKey
Array = dist_base.Array
IntLike = Union[int, np.int16, np.int32, np.int64]
DistributionLike = dist_base.DistributionLike
BijectorLike = bjct_base.BijectorLike


class ConditionedTransformed(Transformed):
    def __init__(self, distribution: DistributionLike, bijector: BijectorLike, embedding_net):
        super().__init__(distribution, bijector)
        self.embedding_net = embedding_net

    def sample(self,
               *,
               seed: Union[IntLike, PRNGKey],
               context: Array,
               sample_shape: Union[IntLike, Sequence[IntLike]] = ()) -> Array:
        """Samples an event.

        Args:
          seed: PRNG key or integer seed.
          sample_shape: Additional leading dimensions for sample.

        Returns:
          A sample of shape `sample_shape` + `batch_shape` + `event_shape`.
        """
        rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
        num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product

        samples = self._sample_n(rng, num_samples, context)

        return samples.reshape(sample_shape + samples.shape[1:])

    def _sample_n(self, key: PRNGKey, n: int, context) -> Array:
        """Returns `n` samples."""
        embedded_context = self.embedding_net(context)
        x = self.distribution.sample(seed=key, sample_shape=n)
        combined_params = jnp.concatenate((x, embedded_context), axis=-1)
        y = jax.vmap(self.bijector.forward)(combined_params)
        return y[..., :x.shape[-1]]

    def log_prob(self, value: Array, context: Array):
        embedded_context = self.embedding_net(context)
        combined_value = jnp.concatenate((value, embedded_context), axis=-1)
        x, ildj_y = self.bijector.inverse_and_log_det(combined_value)
        lp_x = self.distribution.log_prob(x[..., :value.shape[-1]])
        lp_y = lp_x + ildj_y
        return lp_y

    def sample_and_log_prob(
            self,
            *,
            seed: Union[IntLike, PRNGKey],
            context: Array,
            sample_shape: Union[IntLike, Sequence[IntLike]] = ()
    ) -> Tuple[Array, Array]:
        """Returns a sample and associated log probability. See `sample`."""
        rng, sample_shape = convert_seed_and_sample_shape(seed, sample_shape)
        num_samples = functools.reduce(operator.mul, sample_shape, 1)  # product

        samples, log_prob = self._sample_n_and_log_prob(rng, num_samples, context)

        samples = samples.reshape(sample_shape + samples.shape[1:])
        log_prob = log_prob.reshape(sample_shape + log_prob.shape[1:])
        return samples, log_prob

    def _sample_n_and_log_prob(self, key: PRNGKey, n: int, context) -> Tuple[Array, Array]:
        """Returns `n` samples and their log probs.

        This function is more efficient than calling `sample` and `log_prob`
        separately, because it uses only the forward methods of the bijector. It
        also works for bijectors that don't implement inverse methods.

        Args:
          key: PRNG key.
          n: Number of samples to generate.

        Returns:
          A tuple of `n` samples and their log probs.
        """
        embedded_context = self.embedding_net(context)

        x, lp_x = self.distribution.sample_and_log_prob(seed=key, sample_shape=n)
        x = jnp.concatenate(x, embedded_context)
        y, fldj = jax.vmap(self.bijector.forward_and_log_det)(x)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y[..., :x.shape[-1]], lp_y