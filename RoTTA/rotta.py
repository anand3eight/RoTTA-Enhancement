import torch
import torch.nn as nn
import torch.nn.functional as F
import memory
from base_adapter import BaseAdapter
from base_adapter import softmax_entropy
from bn_layers import RobustBN1d, RobustBN2d

class RoTTA(BaseAdapter):
    def __init__(self, student, teacher, optimizer):
        super(RoTTA, self).__init__(student, optimizer)
        self.mem = memory.CSTU(capacity=64, num_class=10, lambda_t=1.0, lambda_u=1.0)
        self.teacher = teacher
        self.nu = 0.001
        self.update_frequency = 64  # actually the same as the size of memory bank
        self.current_instance = 0
        self.fitness_lambda = 0.4
        self.ADaAD_Alpha = 0.5


    @torch.enable_grad()
    def forward_and_adapt(self, batch_data, model, optimizer):
        # batch data
        with torch.no_grad():
            model.eval()
            self.teacher.eval()
            ema_out = self.teacher(batch_data)
            predict = torch.softmax(ema_out, dim=1)
            pseudo_label = torch.argmax(predict, dim=1)
            entropy = torch.sum(- predict * torch.log(predict + 1e-6), dim=1)

        # add into memory
        for i, data in enumerate(batch_data):
            p_l = pseudo_label[i].item()
            uncertainty = entropy[i].item()
            current_instance = (data, p_l, uncertainty)
            self.mem.add_instance(current_instance)
            self.current_instance += 1

            if self.current_instance % self.update_frequency == 0:
                self.update_model(model, optimizer)

        return ema_out

    def update_model(self, student, optimizer):
        x_nat, _ = self.mem.get_memory()
        x_nat = torch.stack(x_nat)
        x_adv = self.adaad_inner_loss(student, self.teacher, x_nat)

        student.train()
        optimizer.zero_grad()

        ori_outputs = student(x_nat)
        adv_outputs = student(x_adv)

        with torch.no_grad():
            self.teacher.eval()
            t_ori_outputs = self.teacher(x_nat)
            t_adv_outputs = self.teacher(x_adv)

        kl_loss1 = nn.KLDivLoss()(F.log_softmax(adv_outputs, dim=1),
                                    F.softmax(t_adv_outputs.detach(), dim=1))
        kl_loss2 = nn.KLDivLoss()(F.log_softmax(ori_outputs, dim=1),
                                    F.softmax(t_ori_outputs.detach(), dim=1))
        
        loss = self.ADaAD_Alpha*kl_loss1 + (1-self.ADaAD_Alpha)*kl_loss2
        loss.backward()
        optimizer.step()

        # train_loss += loss.data

        # correct_ori += torch.max(ori_outputs, 1)[1].eq(targets.data).cpu().sum()
        # total += targets.size(0)
        # # get memory data
        # l = self.calculate_loss()
        # if l is not None:
        #     optimizer.zero_grad()
        #     l.backward()
        #     optimizer.step()

        # self.update_ema_variables(self.teacher, self.model, self.nu)

    def calculate_loss(self):
        sup_data, ages = self.mem.get_memory()
        if len(sup_data) == 0 :
            return None
                
        sup_data = torch.stack(sup_data)
        features = self.feature_extractor(sup_data)
        features = torch.flatten(features, 1)
        batch_std, batch_mean = torch.std_mean(features, dim=0)     

        # Forward pass through student and teacher models
        ema_sup_out = self.model_ema(sup_data)  # Teacher model output (f_T)
        stu_sup_out = self.model(sup_data)      # Student model output (f_S)

        # Timeliness reweighting (assuming ages is defined elsewhere)
        instance_weight = timeliness_reweighting(ages)  # Should be defined or passed as input

        # Entropy loss between student and teacher model (softmax entropy)
        entropy_loss = (softmax_entropy(stu_sup_out, ema_sup_out) * instance_weight).mean()

        # Discrepancy loss between batch statistics (MSE)
        criterion_mse = torch.nn.MSELoss(reduction='none').cuda()
        std_mse = criterion_mse(batch_std, self.train_info['std'])
        mean_mse = criterion_mse(batch_mean, self.train_info['mean'])
        discrepancy_loss = self.fitness_lambda * (std_mse.sum() + mean_mse.sum()) * sup_data.shape[0] / 64

        # Total supervised loss
        l_sup = discrepancy_loss + entropy_loss

        return l_sup

    def adaad_inner_loss(self, 
                     model,
                     teacher_model,
                     x_natural,
                     step_size=2/255,
                     steps=10,
                     epsilon=8/255,
                     BN_eval=True,
                     random_init=True,
                     clip_min=0.0,
                     clip_max=1.0):
        # define KL-loss
        criterion_kl = nn.KLDivLoss(reduction='none')
        if BN_eval:
            model.eval()

        # set eval mode for teacher model
        teacher_model.eval()
        # generate adversarial example
        if random_init:
            x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).cuda().detach()
        else:
            x_adv = x_natural.detach()
        for _ in range(steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                    F.softmax(teacher_model(x_adv), dim=1))
                loss_kl = torch.sum(loss_kl)
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural -
                            epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, clip_min, clip_max)

        x_adv = torch.clamp(x_adv, clip_min, clip_max)

        return x_adv
    
    # @staticmethod
    # def update_ema_variables(student_param, teacher_param, nu):
    #     for teacher_param, student_param in zip(teacher_param.parameters(), student_param.parameters()):
    #         teacher_param.data[:] = (1 - nu) * teacher_param[:].data[:] + nu * student_param[:].data[:]
    #         return student_param

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

    def obtain_origin_stat(self, train_loader):
        print('===> begin calculating mean and variance for ResNet18')
        self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1]).cuda()
        features = []
        with torch.no_grad():
            for images, _ in train_loader:
                images = images.cuda()
                # Pass images through the model to get features
                feature = self.feature_extractor(images)
                feature = torch.flatten(feature, 1)
                features.append(feature)  # Flatten the features

            features = torch.cat(features, dim=0)
            std, mean = torch.std_mean(features, dim=0)
            self.train_info = {'std': std, 'mean': mean}
        del features
        print('===> calculating mean and variance for ResNet18 end')


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
