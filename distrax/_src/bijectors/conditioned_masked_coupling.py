from typing import Tuple, Callable, Optional, Any

import jax.numpy as jnp

from distrax._src.bijectors.masked_coupling import MaskedCoupling
from distrax._src.bijectors import bijector as base
from distrax._src.utils import math


Array = base.Array
BijectorParams = Any


class ConditionedMaskedCoupling(MaskedCoupling):

    def forward_and_log_det(self, x_context: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        self._check_forward_input_shape(x_context)

        # Split x into z and context
        x = x_context.at[..., :self._event_mask.shape[-1]].get()
        context = x_context.at[..., self._event_mask.shape[-1]:].get()

        masked_x = jnp.where(self._event_mask, x, 0.)
        params = self._conditioner(jnp.concatenate((masked_x, context), axis=-1))
        y0, log_d = self._inner_bijector(params).forward_and_log_det(x)
        y = jnp.where(self._event_mask, x, y0)
        logdet = math.sum_last(
            jnp.where(self._mask, 0., log_d),
            self._event_ndims - self._inner_event_ndims)
        return jnp.concatenate((y, context), axis=-1), logdet

    def inverse_and_log_det(self, y_context: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y_context)

        y = y_context.at[..., :self._event_mask.shape[-1]].get()
        context = y_context.at[..., self._event_mask.shape[-1]:].get()

        masked_y = jnp.where(self._event_mask, y, 0.)
        params = self._conditioner(jnp.concatenate((masked_y, context), axis=-1))
        x0, log_d = self._inner_bijector(params).inverse_and_log_det(y)
        x = jnp.where(self._event_mask, y, x0)
        logdet = math.sum_last(jnp.where(self._mask, 0., log_d),
                               self._event_ndims - self._inner_event_ndims)
        return jnp.concatenate((x, context), axis=-1), logdet
