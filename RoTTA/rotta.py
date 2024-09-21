import torch
import torch.nn as nn
import memory
from base_adapter import BaseAdapter
from copy import deepcopy
from base_adapter import softmax_entropy
from bn_layers import RobustBN1d, RobustBN2d

class RoTTA(BaseAdapter):
    def __init__(self, model, optimizer):
        print("-" * 30)
        print("Initializing RoTTA and BaseAdapter with default Values")
        print("Model : Resnet-18 | Optimizer : Adam")
        print("Memory Bank : Size = 64, Lambda_T = 1, Lambda_U = 1")
        print("NU : 0.0001 | Update Frequency for Memory Bank : 64")
        super(RoTTA, self).__init__(model, optimizer)
        self.mem = memory.CSTU(capacity=64, num_class=10, lambda_t=1.0, lambda_u=1.0)
        print(f"Building the Teacher Model")
        self.model_ema = self.build_ema(self.model)
        self.nu = 0.001
        self.update_frequency = 64  # actually the same as the size of memory bank
        self.current_instance = 0
        print("-" * 30)

    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        print("-" * 30)
        print("Entering the Forward and Adapt Function")
        with torch.no_grad():
            model.eval()
            self.model_ema.eval()
            ema_out = self.model_ema(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            print(f"Adding the Current Instance to Memory : (Data, Pseudo Label, Uncertainty) \n{current_instance}")
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                print("Updating the Student and Teacher Models")
                self.update_model(model, optimizer)

        print(f"Exiting forward_and_adapt function with Teacher Output : {ema_out}")
        print("-" * 30)
        return ema_out

    def update_model(self, model, optimizer):
        print("-" * 30)
        print(f"Entering the update_model function")
        model.train()
        self.model_ema.train()
        # get memory data
        print("Getting the Memory Bank Data")
        sup_data, ages = self.mem.get_memory()
        l_sup = None
        if len(sup_data) > 0:
            sup_data = torch.stack(sup_data)
            ema_sup_out = self.model_ema(sup_data)
            stu_sup_out = model(sup_data)
            instance_weight = timeliness_reweighting(ages)
            l_sup = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()
            print(f"Loss calculated from Teacher and Student predictions, instance_weight : \n{l_sup}")
        l = l_sup
        if l is not None:
            optimizer.zero_grad()
            l.backward()
            optimizer.step()

        print(f"Updating the Teacher Model using EMA")
        self.update_ema_variables(self.model_ema, self.model, self.nu)
        print(f"Exiting the update_model function")
        print("-" * 30)

    @staticmethod
    def update_ema_variables(ema_model, model, nu):
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data[:] = (1 - nu) * ema_param[:].data[:] + nu * param[:].data[:]
        return ema_model

    def configure_model(self, model: nn.Module):

        model.requires_grad_(False)
        normlayer_names = []

        for name, sub_module in model.named_modules():
            if isinstance(sub_module, nn.BatchNorm1d) or isinstance(sub_module, nn.BatchNorm2d):
                normlayer_names.append(name)

        for name in normlayer_names:
            bn_layer = get_named_submodule(model, name)
            if isinstance(bn_layer, nn.BatchNorm1d):
                NewBN = RobustBN1d
            elif isinstance(bn_layer, nn.BatchNorm2d):
                NewBN = RobustBN2d
            else:
                raise RuntimeError()

            momentum_bn = NewBN(bn_layer, 0.05)
            momentum_bn.requires_grad_(True)
            set_named_submodule(model, name, momentum_bn)
        return model


def get_named_submodule(model, sub_name: str):
    names = sub_name.split(".")
    module = model
    for name in names:
        module = getattr(module, name)

    return module


def set_named_submodule(model, sub_name, value):
    names = sub_name.split(".")
    module = model
    for i in range(len(names)):
        if i != len(names) - 1:
            module = getattr(module, names[i])

        else:
            setattr(module, names[i], value)

def timeliness_reweighting(ages):
    if isinstance(ages, list):
        ages = torch.tensor(ages).float().cuda()
    return torch.exp(-ages) / (1 + torch.exp(-ages))
