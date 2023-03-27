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

from tslearn.neighbors import KNeighborsTimeSeries
from tslearn.barycenters import dtw_barycenter_averaging

from captum.attr import Saliency
from skimage.metrics import structural_similarity as ssim

from modules import helpers

helpers.seed(1)

#------------------------------------------------- functions Saliency guided training------------------------------------------------------------


def masked_input(saliency, input, mask_factor):                        

    # Get sorted index I for the gradient of output with respect to the input
    gradients = []
    for i in range(input.size()[0]):
            grad = saliency.attribute(input[i].view(1, input.size()[1], input.size()[2]), abs=False).detach().cpu().numpy()
            gradients.append(grad[0])    

    sorted_indices = np.argsort(gradients, axis = 1)                  
    X_masked = input.clone().detach().cpu().numpy()

    # Mask bottom k features of the original input
    k = int(mask_factor * input.size()[1])
    mask = np.random.uniform(low=0, high=1,size=(input.size()[0], k))                            

    for i in range(input.size()[0]):
           for ind in range(k):
                   X_masked[i][sorted_indices[i][ind]] = mask[i,ind]

    return torch.Tensor(X_masked.reshape(input.size()[0], input.size()[1], input.size()[2])).cuda()





#-----------------------------------------------------------------Saliency evaluation metrics--------------------------------------------------    


# Sanity (SSIM)

def evaluate_sanity_ssim(model, classifier, saliency, method, test_data):

    randomized_model = model
    similarities = []
    accuracies = []

    if "CNNClf" in classifier.__name__:     
        layers = [randomized_model.cnn[0], randomized_model.cnn[3], randomized_model.linear]
    elif "LSTMClf" in classifier.__name__:     
        layers = [randomized_model.lstm, randomized_model.linear]
    elif "AttentionLSTM" in classifier.__name__:     
        layers = ["adapted_lstm_cell", randomized_model.lstm.linear_first, randomized_model.lstm.linear_second, randomized_model.linear]
    elif "TCN" in classifier.__name__:     
        layers = [randomized_model.cnn[0], randomized_model.cnn[4], randomized_model.linear]


    for i in range(len(layers)):

        layer = layers[-(i+1)]

        similarity = []
        acc = []

        if layer == "adapted_lstm_cell":
            randomized_model.reset_parameters()

            with torch.no_grad():
                test_instances = test_data.loc[:,test_data.columns != "label"]
                instances = test_instances.to_numpy().reshape(np.shape(test_instances)[0], np.shape(test_instances)[1], -1) 
                instances = torch.Tensor(instances)

                attributions_random = helpers.get_feature_attribution(instances, randomized_model, method).view(np.shape(test_instances)).numpy()
                acc.append(test_data["label"].to_numpy() - torch.round(torch.Tensor(randomized_model(instances))).numpy().squeeze())

            sim = ssim(saliency, attributions_random)
            similarity.append(sim)

        elif layer == randomized_model.lstm:
            random_indices_ih = np.arange(len(layer.weight_ih_l0.detach().numpy().flatten()))
            np.random.shuffle(random_indices_ih)
            random_indices_hh = np.arange(len(layer.weight_hh_l0.detach().numpy().flatten()))
            np.random.shuffle(random_indices_hh)
            array_hh_list = np.array_split(random_indices_hh, 4)
            count = 0

            for array in np.array_split(random_indices_ih, 4):
                array_hh = array_hh_list[count]

                with torch.no_grad():
                    layer.weight_ih_l0.flatten()[array] = torch.Tensor(2*np.random.random_sample(len(array))-1)   # random from interval [-1,1]
                    layer.weight_hh_l0.flatten()[array_hh] = torch.Tensor(2*np.random.random_sample(len(array_hh))-1) 
                    test_instances = test_data.loc[:,test_data.columns != "label"]
                    instances = test_instances.to_numpy().reshape(np.shape(test_instances)[0], np.shape(test_instances)[1], -1) 
                    instances = torch.Tensor(instances)

                    attributions_random = helpers.get_feature_attribution(instances, randomized_model, method).view(np.shape(test_instances)).numpy()
                    acc.append(test_data["label"].to_numpy() - torch.round(torch.Tensor(randomized_model(instances))).numpy().squeeze())

                sim = ssim(saliency, attributions_random)
                similarity.append(sim)

                count = count + 1

        else:
            random_indices = np.arange(len(layer.weight.detach().numpy().flatten()))
            np.random.shuffle(random_indices)

            for array in np.array_split(random_indices, 4):

                with torch.no_grad():
                    layer.weight.flatten()[array] = torch.Tensor(2*np.random.random_sample(len(array))-1)  
                    test_instances = test_data.loc[:,test_data.columns != "label"]
                    instances = test_instances.to_numpy().reshape(np.shape(test_instances)[0], np.shape(test_instances)[1], -1) 
                    instances = torch.Tensor(instances)

                    attributions_random = helpers.get_feature_attribution(instances, randomized_model, method).view(np.shape(test_instances)).numpy()
                    acc.append(test_data["label"].to_numpy() - torch.round(torch.Tensor(randomized_model(instances))).numpy().squeeze())

                sim = ssim(saliency, attributions_random)
                similarity.append(sim)

        similarities.append(similarity)
        accuracies.append(acc)   

    sanity = -1/(test_data.shape[0]*len(layers)) * np.sum(similarities)
    return(sanity, accuracies, similarities)



# Faithfulness (TI)

def evaluate_faithfulnes_single_TI(saliency, input, L, model, mean_signal, label):                        

    sorted_indices = np.argsort(saliency)      

    preds_list = []
    for i in range(L):
        X_masked = input       
        for ind in range(i):
            X_masked[sorted_indices[ind]] = mean_signal[ind]
        out = model(torch.Tensor(X_masked).reshape(1,len(X_masked),1))
        if label == 1:
            preds_list.append(out.detach().numpy())
        elif label == 0:
            preds_list.append(1 - out.detach().numpy())
    
    return preds_list


def faithfulness_TI(correct_test_data, model, attributions, L, mean_signal):

    perturbed_preds = []
    for i in range(correct_test_data.shape[0]): 
        out = evaluate_faithfulnes_single_TI(attributions[i], correct_test_data.iloc[i,correct_test_data.columns != "label"], L, model, mean_signal, correct_test_data["label"].iloc[i])
        perturbed_preds.append(out)

    faithfulness = -1/(correct_test_data.shape[0] * L) * np.sum(perturbed_preds)

    return faithfulness







#--------------------------------------------------functions Native Guide--------------------------------------------------------------------------

# Finding the nearest unlike neighbour (NUN)
    
def native_guide_retrieval(model, instance, predicted_label, train_instances, distance = 'dtw', n_neighbors = 1):
    
    """
    returns distance to NUN and index of NUN in train dataset

    instance, train instances need to be pandas DataFrame including column "label"
    """
    
    knn = KNeighborsTimeSeries(n_neighbors=n_neighbors, metric = distance)
    train_predictions = pd.DataFrame(helpers.get_predictions(train_instances.values, model))
    train_instances["preds"] = train_predictions
    
    out_of_class = train_instances.loc[train_instances["preds"] != int(predicted_label)]
    ooc = out_of_class.reset_index(drop=True)
    ooc = ooc.drop("preds", axis = 1)
    
    knn.fit(ooc.drop("label", axis=1))
    
    dist, idx_ooc = knn.kneighbors(np.array(instance.drop("label", axis = 1)).reshape(1,-1), return_distance=True)       
 
    nun = ooc.iloc[idx_ooc[0]]
    nun_class = helpers.get_predictions(pd.DataFrame(nun).values, model)

    return dist, nun, nun_class
    


# Generate perturbed counterfactual 

def generate_counterfactual_multiclass(model, instance, predicted_label, native_guide, w = 0):
    helpers.seed(1)

    instance = instance.loc[:,instance.columns != "label"].to_numpy()
    native_guide = native_guide.loc[:,native_guide.columns != "label"].to_numpy()
    generated_cf =torch.Tensor(dtw_barycenter_averaging(X=[instance, native_guide], weights=np.array([w, (1-w)])))
    
    #cf_class = model(generated_cf.view(generated_cf.size()[0], generated_cf.size()[1], 1)).detach().numpy().transpose()[0]
    #cf_class = torch.round(torch.Tensor(cf_class)).detach().numpy()
    cf_class = torch.Tensor([1-predicted_label])
    
    while int(cf_class) != int(predicted_label):
        w +=0.01 
        generated_cf_old = generated_cf
        cf_class_old =cf_class
        generated_cf = torch.Tensor(dtw_barycenter_averaging(X=[instance, native_guide], weights=np.array([w, (1-w)])))
        cf_class = model(generated_cf.view(generated_cf.size()[0], generated_cf.size()[1], 1)).detach().numpy().transpose()[0]
        cf_class = torch.round(torch.Tensor(cf_class))
    
    return generated_cf_old, cf_class_old 



# Counterfactual generation (perturbation) method 2: Use weight vectors generated by feature attribution method 

def get_subarray(weights_array, vector_length):

    # returns starting index of most influential subarray

    subarrays = []

    for i in range(len(weights_array)-vector_length+1):
        temp = []

        for j in range(i, i + vector_length):
            temp.append(weights_array[j])

        subarrays.append(temp)
    
    sums = []
    for array in subarrays:
        sums.append(np.sum(array))

    return np.argmax(sums)


def generate_counterfactual_feature_importace(model, instance, predicted_label, native_guide, weights_array, vector_length):    
    # importance weights of NUN

    index_start_of_permutation = get_subarray(weights_array, vector_length)

    instance = instance.loc[:,instance.columns != "label"].to_numpy()[0]
    native_guide = native_guide.loc[:,native_guide.columns != "label"].to_numpy()[0]

    if index_start_of_permutation == 0:
        generated_cf = np.concatenate((
            native_guide[0:vector_length], instance[vector_length:])
            ).reshape(1,-1)
    else: 
        generated_cf = np.concatenate((
            instance[0:index_start_of_permutation ], native_guide[index_start_of_permutation: (index_start_of_permutation + vector_length)], 
            instance[(index_start_of_permutation + vector_length):])
            ).reshape(1,-1)

    cf_class = helpers.get_predictions(generated_cf, model)

    while (int(cf_class) == int(predicted_label)):

        vector_length = vector_length + 1
        index_start_of_permutation = get_subarray(weights_array, vector_length)
        if index_start_of_permutation == 0:
            generated_cf = np.concatenate((
                native_guide[0:vector_length], instance[vector_length:])
                ).reshape(1,-1)
        else: 
            generated_cf = np.concatenate((
                instance[0:index_start_of_permutation], native_guide[index_start_of_permutation: (index_start_of_permutation + vector_length)], 
                instance[(index_start_of_permutation + vector_length):])
                ).reshape(1,-1)
        cf_class = helpers.get_predictions(generated_cf, model)

    return generated_cf, cf_class




# plot TS + counterfactual

def plot_cf(instance, generated_cf, predicted_label, cf_class, method, path = None):

    data_to_plot = pd.DataFrame(instance.loc[:,instance.columns != "label"].transpose())
    data_to_plot.columns = ["Predicted class: {}".format(int(predicted_label))]
    data_to_plot["Counterfactual class: {}".format(int(cf_class[0]))] = pd.DataFrame(generated_cf[0])
    fig, ax = plt.subplots(figsize = (9.5, 5.5))
    data_to_plot.plot(ax = ax, title = 'Instance together with perturbed counterfactual (method: %s)' %str(method), lw = 2, colormap = "brg", grid = True, xlabel="Time", ylabel="Value")
    if path == None:
        pass
    else:
        fig.savefig(path)




