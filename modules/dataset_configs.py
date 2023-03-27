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

import os, random
import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import modules.simulators.demos as sim
from modules.simulators.deterministic import FourierSeries
from modules.simulators.basic import IndependentUniformLatents, ShapeletPlacementLabelMaker, DetShapeletPlacementLabelMaker
from modules.helpers import seed


seed(1)



# Dataset configurations per experiment 


def config_Experiment1():
    simulator = sim.FsShapeletYSimulator(
        length_ts = 300,
        fs_num_sin=10,
        fs_frequency_range=[np.pi/300, np.pi/60],
        fs_phi_range=[-np.pi/4, np.pi/4],
        fs_dominant_amplitude = 1,
        fs_amplitude_exp_decay_rate=0.3,
        fs_noise_sigma=0.1,
        shapelet_type = "random",
        shapelet_window_ratio=0.2,
        shapelet_num_sin=5,
        shapelet_added_noise=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


def config_Experiment2():
    simulator = sim.FsShapeletYSimulator(
        length_ts = 300,
        fs_num_sin=10,
        fs_frequency_range=[np.pi/300, np.pi/60],
        fs_phi_range=[-np.pi/4, np.pi/4],
        fs_dominant_amplitude = 1,
        fs_amplitude_exp_decay_rate=0.3,
        fs_noise_sigma=0.1,
        shapelet_type = "fixed",
        shapelet_window_ratio=0.2,
        shapelet_starting_index = 0.8,
        shapelet_num_sin=5,
        shapelet_added_noise=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


def config_Experiment3():
    simulator = sim.FsShapeletYSimulator(
        length_ts = 300,
        fs_num_sin=10,
        fs_frequency_range=[np.pi/300, np.pi/60],
        fs_phi_range=[-np.pi/4, np.pi/4],
        fs_dominant_amplitude = 1,
        fs_amplitude_exp_decay_rate=0.3,
        fs_noise_sigma=0.1,
        shapelet_type = "fixed",
        shapelet_window_ratio=0.2,
        shapelet_starting_index = 0.4,
        shapelet_num_sin=5,
        shapelet_added_noise=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config



def config_Experiment4():
    simulator = sim.FsShapeletYSimulator(
        length_ts = 300,
        fs_num_sin=10,
        fs_frequency_range=[np.pi/300, np.pi/60],
        fs_phi_range=[-np.pi/4, np.pi/4],
        fs_dominant_amplitude = 1,
        fs_amplitude_exp_decay_rate=0.3,
        fs_noise_sigma=0.1,
        shapelet_type = "fixed",
        shapelet_window_ratio=0.2,
        shapelet_starting_index = 1/300,
        shapelet_num_sin=5,
        shapelet_added_noise=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config




def config_Experiment5():
    simulator = sim.FsUniformLatent2distYSimulator(
        dist_0_frequency_range=[np.pi/300, np.pi/20],
        dist_1_frequency_range=[np.pi/100, np.pi/2],
        dist_0_phi_range=[-np.pi/4, np.pi/4],
        dist_1_phi_range=[-np.pi/4, np.pi/4],
        dist_0_num_sin=10,
        dist_1_num_sin=10,
        dist_0_dominant_amplitude=1,
        dist_1_dominant_amplitude=1,
        dist_0_amplitude_exp_decay_rate=0.3,
        dist_1_amplitude_exp_decay_rate=0.3,
        dist_0_added_noise_sigma_ratio=0.1,
        dist_1_added_noise_sigma_ratio=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


def config_Experiment6():
    simulator = sim.FsUniformLatent2distYSimulator(
        dist_0_frequency_range=[np.pi/300, np.pi/20],
        dist_1_frequency_range=[np.pi/100, np.pi/2],
        dist_0_phi_range=[-np.pi/4, np.pi/4],
        dist_1_phi_range=[-np.pi/4, np.pi/4],
        dist_0_num_sin=1,
        dist_1_num_sin=1,
        dist_0_dominant_amplitude=1,
        dist_1_dominant_amplitude=1,
        dist_0_amplitude_exp_decay_rate=0.3,
        dist_1_amplitude_exp_decay_rate=0.3,
        dist_0_added_noise_sigma_ratio=0.1,
        dist_1_added_noise_sigma_ratio=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


def config_Experiment7():
    simulator =sim.FsUniformLatent2distYSimulator(
        dist_0_frequency_range=[np.pi/300, np.pi/20],
        dist_1_frequency_range=[np.pi/300, np.pi/20],
        dist_0_phi_range=[0, np.pi/4],
        dist_1_phi_range=[np.pi/4, np.pi/2],
        dist_0_num_sin=1,
        dist_1_num_sin=1,
        dist_0_dominant_amplitude=1,
        dist_1_dominant_amplitude=1,
        dist_0_amplitude_exp_decay_rate=0.3,
        dist_1_amplitude_exp_decay_rate=0.3,
        dist_0_added_noise_sigma_ratio=0.1,
        dist_1_added_noise_sigma_ratio=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


def config_Experiment8():
    simulator = sim.FsUniformLatent2distYSimulator(
        dist_0_frequency_range=[np.pi/300, np.pi/20],
        dist_1_frequency_range=[np.pi/300, np.pi/20],
        dist_0_phi_range=[0, np.pi/4],
        dist_1_phi_range=[np.pi/4, np.pi/2],
        dist_0_num_sin=10,
        dist_1_num_sin=10,
        dist_0_dominant_amplitude=1,
        dist_1_dominant_amplitude=1,
        dist_0_amplitude_exp_decay_rate=0.3,
        dist_1_amplitude_exp_decay_rate=0.3,
        dist_0_added_noise_sigma_ratio=0.1,
        dist_1_added_noise_sigma_ratio=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


def config_Experiment9():
    simulator = sim.FsUniformLatent2distYSimulator(
        dist_0_frequency_range=[np.pi/300, np.pi/20],
        dist_1_frequency_range=[np.pi/300, np.pi/20],
        dist_0_phi_range=[-np.pi/4, np.pi/4],
        dist_1_phi_range=[-np.pi/4, np.pi/4],
        dist_0_num_sin=10,
        dist_1_num_sin=10,
        dist_0_dominant_amplitude=1,
        dist_1_dominant_amplitude=3,
        dist_0_amplitude_exp_decay_rate=0.3,
        dist_1_amplitude_exp_decay_rate=0.3,
        dist_0_added_noise_sigma_ratio=0.1,
        dist_1_added_noise_sigma_ratio=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


def config_Experiment10():
    simulator = sim.FsUniformLatent2distYSimulator(
        dist_0_frequency_range=[np.pi/300, np.pi/20],
        dist_1_frequency_range=[np.pi/300, np.pi/20],
        dist_0_phi_range=[-np.pi/4, np.pi/4],
        dist_1_phi_range=[-np.pi/4, np.pi/4],
        dist_0_num_sin=1,
        dist_1_num_sin=1,
        dist_0_dominant_amplitude=1,
        dist_1_dominant_amplitude=3,
        dist_0_amplitude_exp_decay_rate=0.3,
        dist_1_amplitude_exp_decay_rate=0.3,
        dist_0_added_noise_sigma_ratio=0.1,
        dist_1_added_noise_sigma_ratio=0.1,
        class_ratio=0.5
    )
    config = dict(
        simulator = simulator,
        n_features = 1,
        seq_len = 300,
        train_size = 2048,
        val_size = 256,
        test_size = 256,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config



def config_CWRU():
    config = dict(        
        n_features = 1,
        seq_len = 2048,
        mask_factor = 0.25,
        kl_weight = 0.2,
        attention_hops = 50
    )
    return config


