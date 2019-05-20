import torch
from torch.optim import Optimizer
import math
import copy

class SUG(Optimizer):
    def __init__(self, params, l_0, d_0=0, prob=1., eps=1e-4, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if l_0 < 0.0:
            raise ValueError("Invalid Lipsitz constant of gradient: {}".format(l_0))
        if d_0 < 0.0:
            raise ValueError("Invalid disperion of gradient: {}".format(d_0))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(L=l_0, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        self.Lips = l_0
        self.prev_Lips = l_0
        self.D_0 = d_0
        self.eps = eps
        self.prob = prob
        self.start_param = params
        self.upd_sq_grad_norm = None
        self.sq_grad_norm = None
        self.loss = torch.tensor(0.)
        self.closure = None
        super(SUG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SUG, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def comp_batch_size(self):
        """Returns optimal batch size for given d_0, eps and l_0;

        """
        return math.ceil(2 * self.D_0 * self.eps / self.prev_Lips)

    def step(self, loss, closure):
        """Performs a single optimization step.

        Arguments:
            loss : current loss

            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self.start_params = []
        self.loss = loss
        self.sq_grad_norm = 0
        self.closure = closure
        for gr_idx, group in enumerate(self.param_groups):
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            self.start_params.append([])
            for p_idx, p in enumerate(group['params']):
                self.start_params[gr_idx].append([p.data])
                if p.grad is None:
                    continue
                self.start_params[gr_idx][p_idx].append(p.grad.data)
                d_p = self.start_params[gr_idx][p_idx][1]
                p_ = self.start_params[gr_idx][p_idx][0]
                self.sq_grad_norm += torch.sum(p.grad.data * p.grad.data)

                if weight_decay != 0:
                   d_p.add_(weight_decay, p.data)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                self.start_params[gr_idx][p_idx][1] = d_p
        i = 0
        difference = -1
        while difference < 0:
            self.Lips = max(self.prev_Lips * 2 ** (i - 1), 2.)
            for gr_idx, group in enumerate(self.param_groups):
                for p_idx, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    start_param_val = self.start_params[gr_idx][p_idx][0]
                    start_param_grad = self.start_params[gr_idx][p_idx][1]
                    p.data = start_param_val - 1/(2*self.Lips) * start_param_grad
            difference, upd_loss = self.stop_criteria()
            i += 1
        self.prev_Lips = self.Lips

        return self.Lips, i

    def stop_criteria(self):
        """Checks if the Lipsitz constant of gradient is appropriate

           <g(x_k), w_k - x_k> + 2L_k / 2 ||x_k - w_k||^2 = - 1 / (2L_k)||g(x_k)||^2 + 1 / (4L_k)||g(x_k)||^2 = -1 / (4L_k)||g(x_k)||^2
        """
        cur_loss = self.loss.item()
        upd_loss = self.closure().item()
        major =  cur_loss - 1 / (4 * self.Lips) * self.sq_grad_norm
        return major - upd_loss + self.eps / 10, upd_loss

    def get_lipsitz_const(self):
        """Returns current Lipsitz constant of the gradient of the loss function
        """
        return self.Lips

    def get_sq_grad(self):
        """Returns the current second norm of the gradient of the loss function
           calculated by the formula

           ||f'(p_1,...,p_n)||_2^2 ~ \sum\limits_{i=1}^n ((df/dp_i) * (df/dp_i))(p1,...,p_n))

        """
        self.upd_sq_grad_norm = 0
        for gr_idx, group in enumerate(self.param_groups):
            for p_idx, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                self.upd_sq_grad_norm += torch.sum(p.grad.data * p.grad.data)

        return self.upd_sq_grad_norm
