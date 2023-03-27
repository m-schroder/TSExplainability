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

from scipy.special import expit
import torch
import torch.nn as nn
from torch.nn import Parameter
from typing import *
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import pytorch_lightning as pl
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

from captum.attr import Saliency

import pickle 
from random import shuffle

from modules import helpers, functions


helpers.seed(1)

        

class BaseTorchDataset(Dataset):
    def __init__(self, simulator=None, sample_size=None, time_points=None):
        self.sample_size = sample_size
        self.time_points = time_points
        self.simulator = simulator
        
        #x, y = simulator.sample(sample_size=self.sample_size, time_points=self.time_points)
        x, y = simulator.sample(sample_size=self.sample_size, seq_len=self.time_points)
        
        x = x.reshape(*x.shape, 1)
        self.x = torch.tensor(x).float()
        y = y.reshape(*y.shape, 1)
        self.y = torch.tensor(y).float()
    
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    
class RealWorldDataset(Dataset):
    def __init__(self, df):
        data = df
        self.length = df.shape[0]
        labels = df.iloc[:, data.columns == "label"].values
        sequence = data.iloc[:, data.columns != "label"].values
        
        sequence = sequence.reshape(*sequence.shape,1)
        self.sequence = torch.tensor(sequence).float()
        #labels = labels.reshape(*labels.shape,1)
        self.labels = torch.tensor(labels).float()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.sequence[idx], self.labels[idx])

    

class UnivarTSClfDataModule(pl.LightningDataModule):
    def __init__(self,
                 seq_len=1,
                 train_size=3000,
                 val_size=100,
                 test_size=100,
                 batch_size=64,
                 simulator=None,
                 num_workers=0):
        super().__init__()
        # simulator params
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.seq_len = seq_len
        self.simulator = simulator
        
        # dataset params
        self.batch_size = batch_size
        self.num_workers = num_workers

        # is sampled once
        self.train_is_sampled = False
        self.val_is_sampled = False
        self.test_is_sampled = False

        # loaders
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

    def train_dataloader(self):
        if self.train_is_sampled:
            pass
        else:
            train_dataset = BaseTorchDataset(
                sample_size=self.train_size,
                time_points=self.seq_len,
                simulator=self.simulator
            )
            self.train_loader = DataLoader(train_dataset, 
                                    batch_size = self.batch_size, 
                                    shuffle = True, 
                                    num_workers = self.num_workers)
        
        return self.train_loader

    def val_dataloader(self):
        if self.val_is_sampled:
            pass
        else:
            val_dataset = BaseTorchDataset(
                sample_size=self.val_size,
                time_points=self.seq_len,
                simulator=self.simulator
            )
            self.val_loader = DataLoader(val_dataset, 
                                    batch_size = self.batch_size, 
                                    shuffle = False, 
                                    num_workers = self.num_workers)

        return self.val_loader

    def test_dataloader(self):
        if self.test_is_sampled:
            pass
        else:
            test_dataset = BaseTorchDataset(
                sample_size=self.test_size,
                time_points=self.seq_len,
                simulator=self.simulator
            )
            self.test_loader = DataLoader(test_dataset, 
                                    batch_size = self.batch_size, 
                                    shuffle = False, 
                                     num_workers = self.num_workers)

        return self.test_loader

    def predict_dataloader(self):
        if self.test_is_sampled:
            pass
        else:
            test_dataset = BaseTorchDataset(
                sample_size=self.test_size,
                time_points=self.seq_len,
                simulator=self.simulator
            )
            self.test_loader = DataLoader(test_dataset, 
                                    batch_size = self.batch_size, 
                                    shuffle = False, 
                                     num_workers = self.num_workers)

        return self.test_loader



class CWRUDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def setup(self, stage: str):
        with open("../data/CWRU/CWRU_full_dataframe", "rb") as input:
            full_data = pickle.load(input)
        
        full_data = shuffle(full_data)
        self.train_data = RealWorldDataset(full_data.iloc[:int(0.6*full_data.shape[0]),:])
        self.val_data = RealWorldDataset(full_data.iloc[int(0.6*full_data.shape[0]):int(0.8*full_data.shape[0]),:])
        self.test_data = RealWorldDataset(full_data.iloc[int(0.8*full_data.shape[0]):,:])

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle = True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, shuffle = True)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle = False)

    def predict_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, shuffle = False)
    

    

    
##---------------------------------------------------------LSTM models-------------------------------------------------------------------

#LSTM
class LSTMClf(pl.LightningModule):
    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate,
                 criterion):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        y_pred = self.sigmoid(y_pred)
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train_loss", loss, logger = True, prog_bar=True)
        return {"loss": loss}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss, logger = True, prog_bar=True)
        return {"loss": loss}

    def validation_epoch_end(self, outputs):            
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss, logger = True, prog_bar=True)
        return {"loss": loss}
    


#----------LSTM + Saliency guided training-------------------

class LSTMClf_SaliencyGuidedTraining(pl.LightningModule):
    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate,
                 criterion,
                 mask_factor,
                 kl_weight):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.criterionKL = nn.KLDivLoss(reduction="batchmean") 
        self.learning_rate = learning_rate

        self.mask_factor = mask_factor
        self.kl_weight = kl_weight
        self.save_hyperparameters()

        self.lstm = nn.LSTM(input_size=n_features, 
                            hidden_size=hidden_size,
                            num_layers=num_layers, 
                            dropout=dropout, 
                            batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        y_pred = self.sigmoid(y_pred)

        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        saliency = Saliency(self)
        input_masked = functions.masked_input(saliency, x, self.mask_factor)
        y_hat_masked = self(input_masked)
        loss = self.criterion(y_hat, y)
        loss_KL = self.criterionKL(y_hat_masked, y_hat)
        loss = loss + self.kl_weight*loss_KL
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("val_loss", loss, logger = True, prog_bar=True)
        self.log("val_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_epoch_end(self, outputs):            
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        return loss


#-------------LSTM + input cell attention--------------------------------

# adapted code from https://github.com/ayaabdelsalam91/Input-Cell-Attention

class LSTMWithInputCellAttention(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int,r:int,d_a:int):
        super().__init__()
        self.r=r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_iBarh = Parameter(torch.Tensor(input_sz,  hidden_sz* 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.r=r
        self.linear_first = torch.nn.Linear(input_sz,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.init_weights()
     
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)


    def getMatrixM(self, pastTimeSteps):        
        x = self.linear_first(pastTimeSteps)
        x = torch.tanh(x)
        x = self.linear_second(x) 
        x = self.softmax(x,1)
        attention = x.transpose(1,2) 
        matrixM = attention@pastTimeSteps 
        matrixM = torch.sum(matrixM,1)/self.r      
        return matrixM, attention

    def softmax(self,input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d, dim = 1)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)


    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device), 
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states
         
        HS = self.hidden_size

        M=torch.zeros(bs , self.input_sz).double()

        for t in range(seq_sz):
            if(t==0):

                H=x[:, 0, :].view(bs,1,self.input_sz)
                M, attention = self.getMatrixM(H)

            elif(t>0):

                H=x[:, :t+1, :]
                M, attention = self.getMatrixM(H)

        
            gates = M @ self.weight_iBarh + h_t @ self.weight_hh + self.bias

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :HS]), 
                torch.sigmoid(gates[:, HS:HS*2]), 
                torch.tanh(gates[:, HS*2:HS*3]),
                torch.sigmoid(gates[:, HS*3:]), 
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0)) 

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        
        hidden_seq = hidden_seq.squeeze(1)

        return hidden_seq, (h_t, c_t), attention




class AttentionLSTM(pl.LightningModule):

    # r = attention hops (number of time steps the model will attend to)
    # d_a = hidden size for attention matrix computation

    def __init__(self, 
                 n_features, 
                 hidden_size, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 dropout, 
                 learning_rate,
                 criterion,
                 d_a,
                 r):
        super().__init__()
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.d_a = d_a
        self.r = r
        self.save_hyperparameters()

        self.lstm = LSTMWithInputCellAttention(n_features, hidden_size, r, d_a)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        lstm_out, states, attention = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        y_pred = self.sigmoid(y_pred)
        return y_pred, attention
    
    def forward_DL(self, x):
        # This alternative forward function is needed only for computing the saliency scores from the method DeepLift and can be ignored otherwise
        lstm_out, states, attention = self.lstm(x)
        y_pred = self.linear(lstm_out[:,-1])
        y_pred = self.sigmoid(y_pred)
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[0]
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[0]
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("val_loss", loss, logger = True, prog_bar=True)
        self.log("val_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_epoch_end(self, outputs):            
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[0]
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}





#--------------------------------------------------------------CNN models---------------------------------------------------------------------

class CNNClf(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 learning_rate,
                 criterion):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.cnn = nn.Sequential(                                       
            nn.Conv1d(1, 4, kernel_size = 5),                          
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                     
            nn.Conv1d(4, 8, kernel_size = 5),                            
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                      
        )
        self.linear = nn.Linear(2*(seq_len-12), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x.transpose(1,2)
        cnn_out = self.cnn(x_in)
        cnn_out = cnn_out.view(-1, 2*(self.seq_len-12))
        y_pred = self.linear(cnn_out)
        y_pred = self.sigmoid(y_pred)
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("val_loss", loss, logger = True, prog_bar=True)
        self.log("val_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_epoch_end(self, outputs):            
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}


#-----------------CNN + Saliency guided training-------------------------------

class CNNClf_SaliencyGuidedTraining(pl.LightningModule):
    def __init__(self, 
                 n_features, 
                 seq_len, 
                 batch_size,
                 num_layers, 
                 learning_rate,
                 criterion,
                 mask_factor,
                 kl_weight):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.criterion = criterion
        self.criterionKL = nn.KLDivLoss(reduction="batchmean") 
        self.learning_rate = learning_rate

        self.mask_factor = mask_factor
        self.kl_weight = kl_weight
        self.save_hyperparameters()

        self.cnn = nn.Sequential(                                       
            nn.Conv1d(1, 4, kernel_size = 5),                           
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                      
            nn.Conv1d(4, 8, kernel_size = 5),                           
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                      
        )
        self.linear = nn.Linear(2*(seq_len-12), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x.transpose(1,2)
        cnn_out = self.cnn(x_in)
        cnn_out = cnn_out.view(-1, 2*(self.seq_len-12))
        y_pred = self.linear(cnn_out)
        y_pred = self.sigmoid(y_pred)
        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        saliency = Saliency(self)
        input_masked = functions.masked_input(saliency, x, self.mask_factor)
        y_hat_masked = self(input_masked)
        loss = self.criterion(y_hat, y)
        loss_KL = self.criterionKL(y_hat_masked, y_hat)
        loss = loss + self.kl_weight*loss_KL
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("val_loss", loss, logger = True, prog_bar=True)
        self.log("val_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_epoch_end(self, outputs):            
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        return loss



#-----------------------Temporal CNN------------------------------------------

# Code source: https://github.com/locuslab/TCN 

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



#----------adapted TCN implementation----------------------

# Direct comparison to normal CNN
class BasicTCN(pl.LightningModule):                                      
    def __init__(self, 
                 n_features, 
                 seq_len, 
                 batch_size,
                 learning_rate,
                 criterion):
        super().__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.cnn = nn.Sequential(                                       
            weight_norm(nn.Conv1d(1, 4, kernel_size = 5, padding=4, dilation = 1)),   # padding = (kernel_size -1)*dilation_size
            Chomp1d(4),                        
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                      
            weight_norm(nn.Conv1d(4, 8, kernel_size = 5, padding=8, dilation = 2)), 
            Chomp1d(8),                           
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),                      
        )
        self.linear = nn.Linear(2*seq_len, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x_in = x.transpose(1,2)
        cnn_out = self.cnn(x_in)
        cnn_out = cnn_out.view(-1, 2*self.seq_len)
        y_pred = self.linear(cnn_out)
        y_pred = self.sigmoid(y_pred)
        return y_pred
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("train_loss", loss, logger = True, prog_bar=True)
        self.log("train_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def training_epoch_end(self, outputs):    
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Train loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Train accuracy per epoch", avg_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("val_loss", loss, logger = True, prog_bar=True)
        self.log("val_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}

    def validation_epoch_end(self, outputs):            
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["accuracy"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Validation loss per epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Validation accuracy per epoch", avg_acc, self.current_epoch)
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        correct=torch.round(y_hat).eq(y).sum().item()
        total=y.size()[0]
        step_accuracy = torch.Tensor([correct/total])
        self.log("test_loss", loss, logger = True, prog_bar=True)
        self.log("test_accuracy", step_accuracy, logger = True, prog_bar=True)
        return {"loss": loss, "accuracy": step_accuracy}



