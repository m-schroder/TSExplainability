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

from modules.simulators.basic import *
from modules.simulators.deterministic import *


class FsLogNormalLatent2distYSimulator:
    def __init__(self,
                 dist_0_frequency_mean=None, dist_1_frequency_mean=None,
                 dist_0_frequency_sigma=None, dist_1_frequency_sigma=None,
                 dist_0_next_frequency_ratio=2, dist_1_next_frequency_ratio=2,
                 fs_phi_latent_type='uniform', dist_0_phi_config=None, dist_1_phi_config=None,
                 dist_0_num_sin=None, dist_1_num_sin=None,
                 dist_0_dominant_amplitude=1, dist_1_dominant_amplitude=1,
                 dist_0_amplitude_exp_decay_rate=1, dist_1_amplitude_exp_decay_rate=1,
                 dist_0_added_noise_sigma_ratio=0.2, dist_1_added_noise_sigma_ratio=0.2,
                 class_ratio=0.5):
        self.fs0_config = {
            'dominant_amplitude': dist_0_dominant_amplitude,
            'added_noise_sigma_ratio': dist_0_added_noise_sigma_ratio,
            'amplitude_exp_decay_rate': dist_0_amplitude_exp_decay_rate
        }
        self.fs1_config = {
            'dominant_amplitude': dist_1_dominant_amplitude,
            'added_noise_sigma_ratio': dist_1_added_noise_sigma_ratio,
            'amplitude_exp_decay_rate': dist_1_amplitude_exp_decay_rate
        }
        self.class_ratio = class_ratio
        # frequencies
        self.l0_freq_simulator = FrequencyLogNormalLatents(
            num_freqs=dist_0_num_sin,
            first_freq_mean=dist_0_frequency_mean,
            next_freq_ratio=dist_0_next_frequency_ratio,
            sigma=dist_0_frequency_sigma
        )
        self.l1_freq_simulator = FrequencyLogNormalLatents(
            num_freqs=dist_1_num_sin,
            first_freq_mean=dist_1_frequency_mean,
            next_freq_ratio=dist_1_next_frequency_ratio,
            sigma=dist_1_frequency_sigma
        )
        # phis
        if fs_phi_latent_type == 'uniform':
            self.l0_phi_simulator = IndependentUniformLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'low': dist_0_phi_config[0], 'high': dist_0_phi_config[1]}
                for i in range(dist_0_num_sin)
            ])
            self.l1_phi_simulator = IndependentUniformLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'low': dist_1_phi_config[0], 'high': dist_1_phi_config[1]}
                for i in range(dist_1_num_sin)
            ])
        elif fs_phi_latent_type == 'normal':
            self.l0_phi_simulator = IndependentNormalLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'mean': dist_0_phi_config[0], 'sigma': dist_0_phi_config[1], 'log': False}
                for i in range(dist_0_num_sin)
            ])
            self.l1_phi_simulator = IndependentNormalLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'mean': dist_1_phi_config[0], 'sigma': dist_1_phi_config[1], 'log': False}
                for i in range(dist_1_num_sin)
            ])
        else:
            raise ValueError('unknown latent type')

    def sample(self, sample_size=None, seq_len=None):
        n1 = int(sample_size * self.class_ratio)
        n0 = sample_size - n1
        l0_freqs = self.l0_freq_simulator.sample(sample_size=n0)
        l1_freqs = self.l1_freq_simulator.sample(sample_size=n1)
        l0_phis = self.l0_phi_simulator.sample(sample_size=n0)
        l1_phis = self.l1_phi_simulator.sample(sample_size=n1)

        fs0 = FourierSeries(
            sampled_latents=pd.concat([l0_freqs, l0_phis], axis=1),
            **self.fs0_config
        )
        fs1 = FourierSeries(
            sampled_latents=pd.concat([l1_freqs, l1_phis], axis=1),
            **self.fs1_config
        )
        signals = np.concatenate([fs0.sample(seq_len=seq_len), fs1.sample(seq_len=seq_len)], axis=0)
        # make labels
        labels = np.concatenate([np.zeros(shape=(n0,)), np.ones(shape=(n1,))]).astype(int)
        # shuffle
        idx = np.arange(signals.shape[0])
        np.random.shuffle(idx)
        signals = signals[idx]
        labels = labels[idx]
        return signals, labels


class FsLogNormalLatentLogisticYSimulator:
    def __init__(self,
                 fs_num_sin=10,
                 fs_frequency_mean=None,
                 fs_frequency_sigma=None,
                 fs_phi_mean=None,
                 fs_phi_sigma=None,
                 fs_next_frequency_ratio=1.5,
                 fs_dominant_amplitude=1,
                 fs_amplitude_exp_decay_rate=1.5,
                 fs_added_noise_sigma_ratio=0,
                 y_beta_coef_range=None,
                 y_sigmoid_offset=None,
                 latent_subset_for_label=None):
        self.l_freq = FrequencyLogNormalLatents(
            num_freqs=fs_num_sin,
            first_freq_mean=fs_frequency_mean,
            next_freq_ratio=fs_next_frequency_ratio,
            sigma=fs_frequency_sigma
        )
        self.l_phi = IndependentNormalLatents().set_nodes([
            {'name': 'phi_{}'.format(i), 'mean': fs_phi_mean, 'sigma': fs_phi_sigma, 'log': False}
            for i in range(fs_num_sin)
        ])
        self.dominant_amplitude = fs_dominant_amplitude
        self.amplitude_exp_decay_rate = fs_amplitude_exp_decay_rate
        self.added_noise_sigma_ratio = fs_added_noise_sigma_ratio

        # labels
        self.label_maker = LatentLabelMaker(
            coef_min=y_beta_coef_range[0],
            coef_max=y_beta_coef_range[1],
            sigmoid_offset=y_sigmoid_offset
        )
        self.latent_subset_column = latent_subset_for_label

    def sample(self, sample_size=None, seq_len=None):
        # sample latents
        latents = pd.concat([
                self.l_freq.sample(sample_size=sample_size),
                self.l_phi.sample(sample_size=sample_size)
            ], axis=1)
        signals = FourierSeries(
            sampled_latents=latents,
            dominant_amplitude=self.dominant_amplitude,
            amplitude_exp_decay_rate=self.amplitude_exp_decay_rate,
            added_noise_sigma_ratio=self.added_noise_sigma_ratio
        ).sample(seq_len=seq_len)
        labels = self.label_maker.make_label(sampled_latents=latents[self.latent_subset_column])

        return signals, labels


class FsUniformLatentLogisticYSimulator:
    def __init__(self,
                 fs_num_sin=10,
                 fs_frequency_range=None,
                 fs_phi_range=None,
                 fs_dominant_amplitude=1,
                 fs_amplitude_exp_decay_rate=1.5,
                 fs_added_noise_sigma_ratio=0,
                 y_beta_coef_range=None,
                 y_sigmoid_offset=None,
                 latent_subset_for_label=None):
        self.l_simulator = IndependentUniformLatents().set_nodes(
            [
                {'name': 'w_{}'.format(i), 'low': fs_frequency_range[0], 'high': fs_frequency_range[1]}
                for i in range(fs_num_sin)
            ] + [
                {'name': 'phi_{}'.format(i), 'low': fs_phi_range[0], 'high': fs_phi_range[1]}
                for i in range(fs_num_sin)
            ]
        )
        self.dominant_amplitude = fs_dominant_amplitude
        self.amplitude_exp_decay_rate = fs_amplitude_exp_decay_rate
        self.added_noise_sigma_ratio = fs_added_noise_sigma_ratio

        # labels
        self.label_maker = LatentLabelMaker(
            coef_min=y_beta_coef_range[0],
            coef_max=y_beta_coef_range[1],
            sigmoid_offset=y_sigmoid_offset
        )
        self.latent_subset_column = latent_subset_for_label

    def sample(self, sample_size=None, seq_len=None):
        latents = self.l_simulator.sample(sample_size=sample_size)
        signals = FourierSeries(
            sampled_latents=latents,
            dominant_amplitude=self.dominant_amplitude,
            amplitude_exp_decay_rate=self.amplitude_exp_decay_rate,
            added_noise_sigma_ratio=self.added_noise_sigma_ratio
        ).sample(seq_len=seq_len)
        labels = self.label_maker.make_label(sampled_latents=latents[self.latent_subset_column])

        return signals, labels


class FsShapeletYSimulator:     
    def __init__(self,
                 length_ts = 300,
                 fs_latent_type='uniform',
                 fs_num_sin=None,
                 fs_frequency_range=None,
                 fs_phi_range=None,
                 fs_dominant_amplitude=1,
                 fs_amplitude_exp_decay_rate=1,
                 fs_noise_sigma=0.1,
                 shapelet_type = None,
                 shapelet_window_ratio = 0.2,
                 shapelet_starting_index = 1,
                 shapelet_num_sin=5,
                 shapelet_added_noise=None,
                 class_ratio=None):
        if fs_latent_type == 'uniform':
            self.latent_simulator = IndependentUniformLatents()
        else:
            raise ValueError('latent type unknown')
        self.latent_simulator.set_nodes(
            [
                {'name': 'w_{}'.format(i), 'low': fs_frequency_range[0], 'high': fs_frequency_range[1]}
                for i in range(fs_num_sin)
            ] + [
                {'name': 'phi_{}'.format(i), 'low': fs_phi_range[0], 'high': fs_phi_range[1]}
                for i in range(fs_num_sin)
            ]
        )
        self.fs_config = {
            'dominant_amplitude': fs_dominant_amplitude,
            'added_noise_sigma_ratio': fs_noise_sigma,
            'amplitude_exp_decay_rate': fs_amplitude_exp_decay_rate
        }

        if shapelet_type == "random":
            self.shapelet_type = "random"
            self.label_maker = ShapeletPlacementLabelMaker(
                window_ratio=shapelet_window_ratio,
                class_ratio=class_ratio,
                shapelet_num_sin=shapelet_num_sin,
                shapelet_added_noise=shapelet_added_noise
            )
        elif shapelet_type == "fixed":
            self.shapelet_type = "fixed"
            self.label_maker = DetShapeletPlacementLabelMaker(
                shapelet_length = int(shapelet_window_ratio * length_ts), 
                shapelet_index = int(shapelet_starting_index* length_ts -1),
                class_ratio=class_ratio,
                shapelet_num_sin=shapelet_num_sin,
                shapelet_added_noise=shapelet_added_noise
            )
        else:
            raise ValueError('shapelet type unknown') 

    def sample(self, sample_size=None, seq_len=None):
        # make fourier series
        # sample latents
        l = self.latent_simulator.sample(sample_size=sample_size)
        # initiate fourier series
        fs = FourierSeries(
            sampled_latents=l, **self.fs_config
        )
        raw_signal = fs.sample(seq_len=seq_len)
        if self.shapelet_type == "random":
            signals, labels = self.label_maker.make_label(signals=raw_signal)
            return signals, labels
        elif self.shapelet_type == "fixed":
            signals, labels = self.label_maker.make_label(signals=raw_signal)
            return signals, labels
        



class FsUniformLatent2distYSimulator:
    def __init__(self,
                 dist_0_frequency_range=None, dist_1_frequency_range=None,
                 dist_0_phi_range=None, dist_1_phi_range=None,
                 dist_0_num_sin=None, dist_1_num_sin=None,
                 dist_0_dominant_amplitude=1, dist_1_dominant_amplitude=1,
                 dist_0_amplitude_exp_decay_rate=1, dist_1_amplitude_exp_decay_rate=1,
                 dist_0_added_noise_sigma_ratio=0.2, dist_1_added_noise_sigma_ratio=0.2,
                 class_ratio=0.5):
        self.fs0_config = {
            'dominant_amplitude': dist_0_dominant_amplitude,
            'added_noise_sigma_ratio': dist_0_added_noise_sigma_ratio,
            'amplitude_exp_decay_rate': dist_0_amplitude_exp_decay_rate
        }
        self.fs1_config = {
            'dominant_amplitude': dist_1_dominant_amplitude,
            'added_noise_sigma_ratio': dist_1_added_noise_sigma_ratio,
            'amplitude_exp_decay_rate': dist_1_amplitude_exp_decay_rate
        }
        self.class_ratio = class_ratio

        self.l0_simulator = IndependentUniformLatents().set_nodes(
            [
                {'name': 'w_{}'.format(i), 'low': dist_0_frequency_range[0], 'high': dist_0_frequency_range[1]}
                for i in range(dist_0_num_sin)
            ] + [
                {'name': 'phi_{}'.format(i), 'low': dist_0_phi_range[0], 'high': dist_0_phi_range[1]}
                for i in range(dist_0_num_sin)
            ]
        )
        self.l1_simulator = IndependentUniformLatents().set_nodes(
            [
                {'name': 'w_{}'.format(i), 'low': dist_1_frequency_range[0], 'high': dist_1_frequency_range[1]}
                for i in range(dist_1_num_sin)
            ] + [
                {'name': 'phi_{}'.format(i), 'low': dist_1_phi_range[0], 'high': dist_1_phi_range[1]}
                for i in range(dist_1_num_sin)
            ]
        )


    def sample(self, sample_size=None, seq_len=None):
        n1 = int(sample_size * self.class_ratio)
        n0 = sample_size - n1
        l0 = self.l0_simulator.sample(sample_size=n0)
        l1 = self.l1_simulator.sample(sample_size=n1)

        fs0 = FourierSeries(
            sampled_latents=l0, **self.fs0_config
        )
        fs1 = FourierSeries(
            sampled_latents=l1, **self.fs1_config
        )
        signals = np.concatenate([fs0.sample(seq_len=seq_len), fs1.sample(seq_len=seq_len)], axis=0)
        # make labels
        labels = np.concatenate([np.zeros(shape=(n0,)), np.ones(shape=(n1,))]).astype(int)
        # shuffle
        idx = np.arange(signals.shape[0])
        np.random.shuffle(idx)
        signals = signals[idx]
        labels = labels[idx]
        return signals, labels