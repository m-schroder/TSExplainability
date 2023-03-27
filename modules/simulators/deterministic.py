#  Copyright (c) 2023 Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
#  acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS.
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

import numpy as np
import pandas as pd


class FourierSeries:
    """
    Simulates Fourier series signals.

    read on user guide... to be continued

    Parameters
    ----------
    sampled_latents : pd.DataFrame
        `n x p` size Dataframe of latent realizations for n samples and p latent variables.
    dominant_amplitude : positive float, default=1
        amplitude of the dominant frequency in the series.
    amplitude_exp_decay_rate : positive float, default=1
        exponential decay rate for the amplitude of next frequencies
    added_noise_sigma_ratio : float, default=0.0
        determines the added noise sigma s.t. `sigma = dominant_amplitude * added_noise_sigma_ratio`
        if set to `0.0`, then no added noise
    frequency_prefix : str, default='w'
        prefix for the frequency columns in `sampled_latents` dataframe
    phaseshift_prefix : str, default='phi'
        prefix for the phase shift columns in `sampled_latents` dataframe


    Examples
    --------
    Simulating two signals with 2 dominant frequencies for 5 time points.
    First signal has the frequencies :math:`\{\pi/50, \pi/70\}` and phase shifts :math:`\{0, \pi/2\}`.
    Second signal has the frequencies :math:`\{\pi/20, \pi/30\}` and phase shifts :math:`\{0, \pi/8\}`.

    >>> from parcs.simulators.temporal.deterministic import FourierSeries
    >>> import numpy as np
    >>> latents = pd.DataFrame([
    ...     [np.pi / 50, np.pi / 70, 0, np.pi / 2],
    ...     [np.pi / 20, np.pi / 30, 0, np.pi / 8],
    ... ], columns=('w_0', 'w_1', 'phi_0', 'phi_1'))
    >>> fs = FourierSeries(sampled_latents=latents)
    >>> np.round(fs.sample(seq_len=5), 2)
    array([[0.37, 0.43, 0.49, 0.55, 0.61],
           [0.14, 0.33, 0.52, 0.69, 0.85]])
    """

    def __init__(self,
                 sampled_latents: pd.DataFrame = None,
                 dominant_amplitude: float = 1,
                 amplitude_exp_decay_rate: float = 1,
                 added_noise_sigma_ratio: float = 0.0,
                 frequency_prefix: str = 'w',
                 phaseshift_prefix: str = 'phi'):
        self.latents = sampled_latents
        self.noise_sigma = added_noise_sigma_ratio
        self.dominant_amplitude = dominant_amplitude

        # get latent columns
        self.frequency_columns = sorted([
            i for i in sampled_latents.columns if i.split('_')[0] == frequency_prefix
        ])
        self.phaseshift_columns = sorted([
            i for i in sampled_latents.columns if i.split('_')[0] == phaseshift_prefix
        ])
        # latent values
        self.frequencies = self.latents[self.frequency_columns].values
        self.phaseshifts = self.latents[self.phaseshift_columns].values

        # setup amplitude
        num_freqs = len(self.frequency_columns)
        self.amplitudes = dominant_amplitude * np.exp(-amplitude_exp_decay_rate * np.arange(num_freqs)) * \
                          np.ones(shape=(len(self.latents), num_freqs))

    def sample(self, seq_len: int = 10):
        """
        sampled Fourier series

        Parameters
        ----------
        seq_len : int, default=10

        Returns
        -------
        D : array-like with shape `(n,t)`
            matrix where rows are samples and columns are time points.
        """
        t = np.arange(seq_len)
        # reshape [n x w_i] -> [n x w_i x 1], because it will be outer product with time
        freqs = self.frequencies.reshape(*self.frequencies.shape, 1)
        amps = self.amplitudes.reshape(*self.amplitudes.shape, 1)
        phis = self.phaseshifts.reshape(*self.phaseshifts.shape, 1)
        # calculate bucket of sins for samples and reshape again to [ n x w x t]
        decomposed = (amps * np.sin(freqs * t + phis)).reshape(*self.frequencies.shape, -1)
        # add sine waves
        data = decomposed.sum(axis=1)
        # add noise
        data = data + np.random.normal(0, self.dominant_amplitude * self.noise_sigma, size=data.shape)
        return data


