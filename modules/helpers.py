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
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from matplotlib.collections import LineCollection
import seaborn as sns


import os, random

from captum.attr import IntegratedGradients, KernelShap, Lime, DeepLift 
import captum.attr


#-------------------------------------------------------------General helper functions--------------------------------------------------------

def seed(seed):

    pl.seed_everything(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

seed(1)




def get_sampled_data(datamodule, model):
    seed(1)

    train_instances = []
    train_labels = []
    train_pred = []

    val_instances = []
    val_labels = []
    val_pred = []

    test_instances = []
    test_labels = []
    test_pred = []

    with torch.no_grad():

        for i in datamodule.train_dataloader():
            if "Attention" in str(model):
                pred = model.forward(i[0])[0]
            else:
                pred = model.forward(i[0])
            pred = pred.numpy().transpose()[0]
            train_instances = np.append(train_instances, i[0])
            train_labels = np.append(train_labels, i[1].numpy().transpose()[0])
            train_pred = np.append(train_pred, torch.round(torch.Tensor(pred)))
        size = (len(train_instances)+1)/(i[0].size()[1])
        train_instances = train_instances.reshape((int(size), i[0].size()[1]))
        train_df = pd.DataFrame(train_instances)
        train_df["label"] = pd.DataFrame(train_labels)

        for i in datamodule.val_dataloader():
            if "Attention" in str(model):
                pred = model.forward(i[0])[0]
            else:
                pred = model.forward(i[0])
            pred = pred.numpy().transpose()[0]
            val_instances = np.append(val_instances, i[0])
            val_labels = np.append(val_labels, i[1].numpy().transpose()[0])
            val_pred = np.append(val_pred, torch.round(torch.Tensor(pred)))
        size = (len(val_instances)+1)/(i[0].size()[1])
        val_instances = val_instances.reshape((int(size), i[0].size()[1]))
        val_df = pd.DataFrame(val_instances)
        val_df["label"] = pd.DataFrame(val_labels)

        for i in datamodule.test_dataloader():
            if "Attention" in str(model):
                pred = model.forward(i[0])[0]
            else:
                pred = model.forward(i[0])
            pred = pred.numpy().transpose()[0]
            test_instances = np.append(test_instances, i[0])
            test_labels = np.append(test_labels, i[1].numpy().transpose()[0])
            test_pred = np.append(test_pred, torch.round(torch.Tensor(pred)))
        size = (len(test_instances)+1)/(i[0].size()[1])
        test_instances = test_instances.reshape((int(size), i[0].size()[1]))
        test_df = pd.DataFrame(test_instances)
        test_df["label"] = pd.DataFrame(test_labels)

    return train_df, val_df, test_df, train_pred, val_pred, test_pred




def get_predictions(data, model):
    seed(1)
    """
        data has to have shape (number time series, ts_length)
    """
    with torch.no_grad():
        tensor_data = torch.Tensor(data)
        preds = model.forward(tensor_data.view(tensor_data.size()[0], tensor_data.size()[1], 1)).numpy().transpose()[0]
        classification_result = torch.round(torch.Tensor(preds))

    return classification_result.numpy().astype(int)




#----------------------------------------------------------Attribution helpers------------------------------------------------------

def attention_forward_wrapper(model_of_interest):
    def forward(x):
        out = model_of_interest(x)
        return(out[0])
    return forward



def get_feature_attribution(x, classifier, method):
    """
        x = instance to be explained (tensor)
        classifier = trained classifier
        method = Feature attribution method to be used for calculating importance score (str)

        returns numpy array of importance scores
    """
    x = x.requires_grad_()

    if method == "IG":
        if "Attention" in str(classifier):
            FA = IntegratedGradients(attention_forward_wrapper(classifier))
            score = FA.attribute(x, baselines = 0)
        else:
            FA = IntegratedGradients(classifier)
            score = FA.attribute(x, baselines = 0)       

    elif method == "SHAP":
        if "Attention" in str(classifier):
            FA = KernelShap(attention_forward_wrapper(classifier))
            score = FA.attribute(x, baselines = 0)
        else:
            FA = KernelShap(classifier)
            score = FA.attribute(x, baselines = 0)      
    
    elif method == "LIME":
        if "Attention" in str(classifier):
            FA = Lime(attention_forward_wrapper(classifier))
            score = FA.attribute(x)
        else:
            FA = Lime(classifier)
            score = FA.attribute(x) 

    elif method == "DL":
        if "Attention" in str(classifier):
            classifier.forward = classifier.forward_DL
            FA = DeepLift(classifier)
            score = FA.attribute(x)
        else:
            FA = DeepLift(classifier)
            score = FA.attribute(x) 
   
    score.detach().numpy()

    return score




def plot_heatmap_background(data, saliency, title=None, path = None):   
    """
        data, saliency as numpy array
    """ 

    fig, ax1 = plt.subplots(1,1)
    fig.set_size_inches(9.5, 5.5)
    plt.grid(True)
    plt.xlabel("Time")
    plt.ylabel("Values")
    if title != None:
        plt.title(title)
    plt.plot(data, lw=2, c="w")
    ax1 = plt.imshow(saliency.reshape([1, -1]), cmap=plt.cm.magma, aspect="auto", 
                    extent=[0, len(saliency) - 1, float(np.min([np.min(data), np.min(saliency)])),
                            float(np.max([np.max(data), np.max(saliency)])) ]
                    )
    bar = fig.add_axes(rect = [0.28, 0.05, 0.5, 0.05])
    fig.colorbar(ax1, cax=bar, orientation="horizontal", label = "Attribution")
    plt.tight_layout(pad=5)

    if path != None:
        fig.savefig(path)

    plt.show()



def plot_multi_attribution_heatmap(df, x_label_width = 10, y_label_width = False, title = None, ylabel = None, path = None):
    """
        Used for plotting heatmaps of multiple time series below each other to get overview of distribution of importance in time
        (does not plot behaviour of time series)

        df = dataframe of attribution vectors; rows = different time series, columns = time steps
        title, ylabel = str
        y_label_width = False or 1 -> 1 plots labels/indices of all classes
    """
    fig, ax = plt.subplots(figsize = (9.5, 5.5))
    heat = sns.heatmap(df, xticklabels = x_label_width, yticklabels = y_label_width, cbar = True)   
    heat.set_xlabel('Time')

    if title != None:
        plt.title(title)
    if ylabel != None:
        plt.ylabel(ylabel)

    plt.yticks(rotation=0) 

    if path != None:
        fig.savefig(path)