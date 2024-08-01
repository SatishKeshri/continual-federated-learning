from typing import Any, Dict, List, Optional, Tuple
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import math
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
# from custom_memory_buffers import prop_memory_scheme
from dirichlet_sampler import sampler
from utils import arg_parser, average_weights
import gc
import random
from types import SimpleNamespace

import pickle
import os
import shutil

from avalanche.benchmarks.classic import SplitMNIST, PermutedMNIST
from avalanche.training.storage_policy import ClassBalancedBuffer


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, n_classes: int):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def format_dict_with_precision(data, precision=4):
    return {key: tuple(f'{val:.{precision}f}' for val in value) for key, value in data.items()}

class CFLAG:
    """Proposed algorithm"""

    def __init__(self, args: Dict[str, Any], benchmark, num_tasks):
        self.args = args
        self.bm = benchmark
        self.num_tasks = num_tasks
        
        self.device = torch.device(
            f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
        )

    
    def server(self, ):
        """Implementation of the server strategy"""
        if self.args.model_name == "MLP":
            self.global_model = MLP(input_size=28*28, hidden_size=512, n_classes=10).to(self.device).to(self.device) 
        else:
            raise NotImplementedError
        print(f"Training using model: {type(self.global_model)}")

        self.num_clients = max(int(self.args.frac * self.args.n_clients),1)
        self.client_buffer_size = [self.args.initial_buffer_size]*self.num_clients
        if self.args.memory_scheme == "class_balanced":
            self.memory_buffer = [ClassBalancedBuffer(max_size=self.client_buffer_size[0],adaptive_size=True,
                                                    total_num_classes=None,
                                                    ) for _ in range(self.num_clients)
                                                    ]
        else:
            raise NotImplementedError
        
        self.cl_models_list = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]

        for task_id in range(self.num_tasks):
            self.task_id = task_id
            exp_train = self.bm.train_stream[task_id]
            self.mini_server(exp_train)


    def mini_server(self, exp_train):
        """This funtion implements one task for the server"""
        print(f"Task id {self.task_id}, training on {exp_train.classes_in_this_experience}")
        cl_data_indices = sampler(exp_train.dataset, n_clients=self.num_clients, n_classes=exp_train.classes_in_this_experience, alpha=self.args.alpha)
        
        save_train_losses = []
        save_train_accs = []
        save_test_losses = []
        save_test_accs = []

        train_losses = []
        train_accs = []
        
        if self.task_id == 0 :
            self.opt_lists = [torch.optim.SGD(self.cl_models_list[i].parameters(),
                                        lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        )
                                        for i in range(len(self.cl_models_list)) ]
        else :
            self.opt_lists = [torch.optim.SGD(self.cl_models_list[i].parameters(),
                                        lr=self.args.lr_retrain,
                                        momentum=self.args.momentum_retrain,
                                        )
                                        for i in range(len(self.cl_models_list)) ]
        for epoch in tqdm(range(self.args.n_epochs),):
            self.global_epoch = epoch
            clients_losses = []
            clients_accs = []
            clients_test_losses = []
            clients_test_accs = []
            idx_clients = [i for i in range(self.num_clients)]
            for cl_id in idx_clients:
                sub_exp_train = copy.deepcopy(exp_train)
                sub_exp_train.dataset = sub_exp_train.dataset.subset(cl_data_indices[cl_id])
                cl_loss, cl_acc = self._train_client_adap(
                client_idx=cl_id,
                sub_exp_train=sub_exp_train,
                )
                clients_losses.append(cl_loss)
                clients_accs.append(cl_acc)
                cl_test_loss, cl_test_acc = self.test(model=self.cl_models_list[cl_id])
                clients_test_losses.append(cl_test_loss)
                clients_test_accs.append(cl_test_acc)

            # Update server model based on client models and then transmit to clients
            updated_weights = average_weights(self.cl_models_list, state_dict=False)
            self.global_model.load_state_dict(updated_weights)
            self.cl_models_list = [copy.deepcopy(self.global_model) for _ in range(self.num_clients)]   

            avg_loss = sum(clients_losses)/ len(clients_losses)
            avg_acc = sum(clients_accs)/ len(clients_accs)
            train_losses.append(avg_loss)
            train_accs.append(avg_acc)
            

            save_train_accs.append(clients_accs)
            save_train_losses.append(clients_losses)
            save_test_accs.append(clients_test_accs)
            save_test_losses.append(clients_test_losses)
        print(f"At the end of task {self.task_id} length of memory buffers:")
        print([self.client_buffer_size[client_idx] for client_idx in range(self.num_clients)])
        ## Update memory buffer for each client
        for cl_id in idx_clients:
            sub_exp_train = copy.deepcopy(exp_train)
            sub_exp_train.dataset = sub_exp_train.dataset.subset(cl_data_indices[cl_id])
            self.memory_buffer_updator(cl_id, sub_exp_train, name=self.args.memory_scheme)
        print(f"\nResults after Task: {self.task_id}, Epoch: {epoch + 1} global rounds of training:")
        print(f"---> Avg Training Loss: {sum(train_losses) / len(train_losses)}")
        print(f"---> Avg Training Accuracy (with older scores, not updated after this epoch): {sum(train_accs) / len(train_accs)}")
        # save results
        with open(result_path+'Train_loss_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_train_losses, f)
        with open(result_path+'Train_accuracy_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_train_accs, f)
        with open(result_path+'Test_loss_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_test_losses, f)
        with open(result_path+'Test_accuracy_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(save_test_accs, f)

        # Get the test accuracies
        test_loss, test_acc = self.test()
        print(f"After full training on task {self.task_id}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

        # Get avaluation on past tasks
        if self.task_id >0:
            task_metric_dict = {task_id: self.test_with_task_id(task_id) for task_id in range(self.task_id + 1) }
            print(f"At the end of task_id: {self.task_id} the previous metrics are as follows: \n {task_metric_dict}")
        
        #saving results 
        task_metric_dict = {task_id: self.test_with_task_id(task_id) for task_id in range(self.task_id + 1)}
        with open(global_result_path+'test_accuracy_task'+str(self.task_id)+'.pkl', 'wb') as f:
            pickle.dump(task_metric_dict, f)
    

    def _normalize_grad(self, model, grad_list):
        """Mutates the "grad_list" in-place"""
        # Accumulate and normalize gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = copy.deepcopy((param.grad))
                old_grad = grad_list[name]
                old_grad += grad
                norm = old_grad.norm()
                if norm != 0:  # Avoid division by zero
                    normalized_grad = old_grad / norm
                    grad_list[name] = normalized_grad
                
    
    def _get_client_memory_grads(self, client_idx):
        """Compute grads on the memory data and return the grads"""
        # model = copy.deepcopy(self.global_model)
        model = self.cl_models_list[client_idx]
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = self.opt_lists[client_idx]
        # Get the train loader
        storage_p = self.memory_buffer[client_idx]
        train_loader_memory, _ = self._get_dataloader(storage_p.buffer, only_train=True)
        
        grads_f = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
        grads_f_full = [param.grad.clone().detach() if param.grad is not None else torch.zeros_like(param) for n, param in model.named_parameters() ]
        samples = 0
        for idx, (data, target, _) in enumerate(train_loader_memory):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            with torch.no_grad():
                grads_f_full = [param.grad.clone().detach()+ grad_f for ((n, param),grad_f)  in zip(model.named_parameters(), grads_f_full)]  
                _ = self._normalize_grad(model, grads_f)
                samples += data.size(0)
        grads_f = {name: grad / len(train_loader_memory) for name, grad in grads_f.items()}
        grads_f_full = [grad/len(train_loader_memory) for grad in grads_f_full]
        optimizer.zero_grad()
        return list(grads_f.values()), grads_f_full
    
    
    def _inner_prod_sum(self, grad_list_1, grad_list_2):
        """Calculate inner product"""
        inn = torch.tensor(0.0).to(self.device)
        for grad1, grad2 in zip(grad_list_1, grad_list_2):
            product = torch.dot(grad1.view(-1), grad2.view(-1))
            inn += product
        return inn
    
    def calc_new_grads(self, grads_f, grads_g):
        """Calculate the gradients for new task"""
       
        inn = self._inner_prod_sum(grads_f, grads_g)
        norm_2 = self._inner_prod_sum(grads_f, grads_f)
        new_grads = [torch.where(inn <= 0, grads_g[i] - torch.mul(torch.div(inn, norm_2), grads_f[i]), grads_g[i]) for i in range(len(grads_g))] #for resnet
        return new_grads
    
    def _sigmoid(self, x):
        return 1/ (1 + math.exp(-x))
    
    def _fetch_memory_data(self, client_idx):
        """Returns data for training from the memory buffer"""
        if self.args.memory_scheme != "prop":
            # strategy_state = SimpleNamespace(experience=sub_exp_train)
            storage_p = self.memory_buffer[client_idx]
            memory_data = storage_p.buffer
        elif self.args.memory_scheme == "prop":
            memory_data = self.memory_buffer[client_idx]
        return memory_data

    
    def _train_on_memory_restore_model(self, client_idx, criterion,):
        """Funtion to train model on memory data"""
        model = copy.deepcopy(self.cl_models_list[client_idx])
        model.train()
        if self.args.optim == "sgd":
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=self.args.lr,
                                        momentum=self.args.momentum,
                                        )
        elif self.args.optim == "adam":
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=self.args.lr,
                                        
            )
        # Fetch data
        memory_data = self._fetch_memory_data(client_idx)
        train_loader_memory, _ = self._get_dataloader(memory_data, only_train=True)
        for idx, (data, target, _) in enumerate(train_loader_memory):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, target)
            loss.backward()
            clipped_val = clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            optimizer.zero_grad()

        new_f_data = [param.data.clone().detach() for param in model.parameters()]
        old_client_model_data = [param.data.clone().detach() for param in self.cl_models_list[client_idx].parameters()]
        delta_f_data = [new-old for (new,old) in zip(new_f_data, old_client_model_data)]
        return delta_f_data

    def _train_client_curr(self, client_idx, sub_exp_train, grads_f=None):
        """Train a client on current task data"""
        
        model = self.cl_models_list[client_idx]
        model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = self.opt_lists[client_idx]
        lr_lmbda = lambda epoch: 1/math.sqrt(self.args.n_client_epochs)
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lmbda, last_epoch = -1)
        # Get the train loader and test loader
        train_loader, test_loader = self._get_dataloader(sub_exp_train.dataset)
        for epoch in range(self.args.n_client_epochs):
            for idx, (data, target, _) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                logits = model(data)
                loss = criterion(logits, target)
                loss.backward()
                if grads_f:
                    with torch.no_grad():
                        grads_g = [param.grad.clone().detach() for param in model.parameters()]
                        model_params_dict = copy.deepcopy(dict(model.named_parameters()))
                        _ = self._normalize_grad(model, model_params_dict)
                        new_grads = self.calc_new_grads(grads_f, grads_g)
                        # Set new grads in the model parameters
                        for (n,p), new_grad in zip(model.named_parameters(), new_grads) :
                            p.grad = new_grad
                    _ = clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.step()
                    optimizer.zero_grad()
            if self.task_id >0:
                scheduler.step()
            # Check test accuracy
            task_metric_dict = {task_id: self.test_with_task_id(task_id, model=model) for task_id in range(self.task_id+1) }
            formatted_results = format_dict_with_precision(task_metric_dict)
            print(f"Task: {self.task_id}, client: {client_idx}, Server/local_epoch: {self.global_epoch}/{epoch}, test_stats on local model: {formatted_results}")

        #save results
        with open(client_result_path+'test_accuracy_model'+str(client_idx)+'task'+str(self.task_id)+'.pkl', 'wb') as f:
             pickle.dump(task_metric_dict, f)

        print(f"--------"*25)
    
    def memory_buffer_updator(self, client_idx, sub_exp_train, name=None ):
        """Function to update memory buffer"""
        strategy_state = SimpleNamespace(experience=sub_exp_train)
        storage_p = self.memory_buffer[client_idx]
        if self.task_id >0:
            self.client_buffer_size[client_idx] += self.args.initial_buffer_size
            storage_p.resize(strategy_state, self.client_buffer_size[client_idx])
        _ = storage_p.update(strategy_state)
    
    def _train_client_adap(self,
                client_idx,
                sub_exp_train,
                ):
        """Trains a client for one global server round on both the memory data and current task data"""
        if self.task_id == 0:
            _ = self._train_client_curr(client_idx, sub_exp_train,)        
        else:
            grads_f_norm, grads_f_full = self._get_client_memory_grads(client_idx)
            criterion = nn.CrossEntropyLoss()
            delta_f_data = self._train_on_memory_restore_model(client_idx, criterion)
            _ = self._train_client_curr(client_idx, sub_exp_train, grads_f_full)

            local_model = copy.deepcopy(self.cl_models_list[client_idx])
            # Merge the memory data gradients with the updated model
            for (n,p), new_data in zip(local_model.named_parameters(), delta_f_data):
                p.data += new_data
            self.cl_models_list[client_idx] = local_model
        ## Fetch scores
        train_loader, _ = self._get_dataloader(sub_exp_train.dataset, only_train=True)
        train_loss, train_acc = self.test(train_loader, model=self.cl_models_list[client_idx])
        return train_loss, train_acc


    def _get_dataloader(self, training_dataset, only_train=False):
        """Retuns dataloader"""
        train_loader = DataLoader(training_dataset, batch_size=self.args.batch_size, shuffle=True)
        if not only_train:
            test_set = self.bm.test_stream[self.task_id].dataset
            test_loader = DataLoader(test_set, batch_size=128)
        else:
            test_loader = None
        
        return train_loader, test_loader
    
    def test(self, test_loader=None, model=None) -> Tuple[float, float]:
        """Test on the model. If no model passed as an argument then test on the server model."""
        if model == None:
            model = copy.deepcopy(self.global_model)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        if test_loader == None:
            test_set = self.bm.test_stream[self.task_id].dataset
            test_loader = DataLoader(test_set, batch_size=128)
        # test_loader = 
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for idx, (data, target, _) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                logits = model(data)
                # logits = logits[self.task_id] #Alexnet
                loss = criterion(logits, target)

                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)
        # calculate average accuracy and loss
        total_loss /= (idx+1)
        total_acc = total_correct / total_samples

        return total_loss, total_acc
    
    def test_with_task_id(self,task_id, model=None) -> Tuple[float, float]:
        if model ==  None:
            model = copy.deepcopy(self.global_model)
        model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        # Fetch the test_loader based on task_id
        test_set = self.bm.test_stream[task_id].dataset
        test_loader = DataLoader(test_set, batch_size=128)
        with torch.no_grad():
            for idx, (data, target, _) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                logits = model(data)
                loss = criterion(logits, target)
    
                total_loss += loss.item()
                total_correct += (logits.argmax(dim=1) == target).sum().item()
                total_samples += data.size(0)

        total_loss /= (idx+1)
        total_acc = total_correct / total_samples

        return total_loss, total_acc
    
if __name__== "__main__":
    parser = arg_parser()

    result_path = f'./saved_results/results_{parser.dataset}_{parser.model_name}/buffer_{parser.initial_buffer_size}/alpha_{parser.alpha}/lrs_{parser.lr}-{parser.lr_retrain}/momen_{parser.momentum}-{parser.momentum_retrain}/seed_{parser.seed}/'
    if os.path.exists(result_path):
        shutil.rmtree(result_path)
    os.makedirs(result_path)

    global_result_path = result_path + 'global_model/'
    if os.path.exists(global_result_path):
        shutil.rmtree(global_result_path)
    os.makedirs(global_result_path)

    client_result_path = result_path + 'local_model/'
    if os.path.exists(client_result_path):
        shutil.rmtree(client_result_path)
    os.makedirs(client_result_path)
    

    if parser.dataset == "MNIST":
        num_tasks = parser.num_tasks
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))  # Normalize inputs
                                        ])
        bm = SplitMNIST(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    train_transform=transform, eval_transform=transform
                    )
    elif parser.dataset == "PermutedMNIST":
        num_tasks = parser.num_tasks
        transform = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))  # Normalize inputs
                                        ])
        bm = PermutedMNIST(n_experiences=num_tasks,
                    return_task_id=True,
                    seed=parser.seed,
                    train_transform=transform, eval_transform=transform
                    )
    
    else:
        raise NotImplementedError
    print(f"Dataset: {parser.dataset} with splits: {num_tasks}")
    algo = CFLAG(parser, bm, num_tasks)
    algo.server()
    print(parser)