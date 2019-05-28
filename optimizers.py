import torch
from torch.optim.optimizer import Optimizer, required
import math
import copy

class A2GradUni(Optimizer):

    def __init__(self, params, beta=10, lips=10):
        defaults = dict(beta=beta, lips=lips)
        super(A2GradUni, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(A2GradUni, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['alpha_k'] = 1
                    state['v_k'] = 0
                    state['avg_grad'] = copy.deepcopy(grad)
                    state['x_k'] = copy.deepcopy(p.data)

                gamma_k = 2 * group['lips'] / (state['step'] + 1)

                avg_grad = state['avg_grad']
                avg_grad.mul_(state['step'])
                avg_grad.add_(grad)
                avg_grad.div_(state['step']+1)

                delta_k = torch.add(grad, -1, avg_grad)

                state['v_k'] += torch.sum(delta_k * delta_k).item()

                h_k = math.sqrt(state['v_k'])
                alpha_k_1 = 2 / (state['step'] + 3)
                coef = 1 / (gamma_k+group['beta'] * h_k)
                x_k_1 = state['x_k']
                x_k_1.add_(-coef, grad)

                p.data.mul_(1 - alpha_k_1)
                p.data.add_(alpha_k_1, x_k_1)
                p.data.add_(-(1 - alpha_k_1) * state['alpha_k'] * coef, grad)

                state['alpha_k'] = alpha_k_1
                state['step'] += 1

        return loss

class A2GradInc(Optimizer):

    def __init__(self, params, beta=10, lips=10):
        defaults = dict(beta=beta, lips=lips)
        super(A2GradInc, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(A2GradInc, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]


                if len(state) == 0:
                    state['step'] = 0
                    state['alpha_k'] = 1
                    state['v_k'] = 0
                    state['avg_grad'] = copy.deepcopy(grad)
                    state['x_k'] = copy.deepcopy(p.data)

                gamma_k = 2*group['lips']/(state['step']+1)

                avg_grad = state['avg_grad']
                avg_grad.mul_(state['step'])
                avg_grad.add_(grad)
                avg_grad.div_(state['step']+1)

                delta_k = torch.add(grad, -1, avg_grad)

                state['v_k'] *= (state['step'] / (state['step'] + 1)) ** 2
                state['v_k'] += torch.sum(delta_k * delta_k).item()

                h_k = math.sqrt(state['v_k'])
                alpha_k_1 = 2/(state['step'] + 3)
                coef = 1 / (gamma_k + group['beta'] * h_k)
                x_k_1 = state['x_k']
                x_k_1.add_(-coef, grad)

                p.data.mul_(1 - alpha_k_1)
                p.data.add_(alpha_k_1, x_k_1)
                p.data.add_(-(1 - alpha_k_1) * state['alpha_k'] * coef, grad)

                state['alpha_k'] = alpha_k_1
                state['step'] += 1

        return loss

class A2GradExp(Optimizer):

    def __init__(self, params, beta=10, lips=10, rho=0.5):
        defaults = dict(beta=beta, lips=lips, rho=rho)
        super(A2GradExp, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(A2GradExp, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]


                if len(state) == 0:
                    state['step'] = 0
                    state['alpha_k'] = 1
                    state['v_k'] = 0
                    state['avg_grad'] = copy.deepcopy(grad)
                    state['x_k'] = copy.deepcopy(p.data)

                gamma_k = 2*group['lips']/(state['step']+1)

                avg_grad = state['avg_grad']
                avg_grad.mul_(state['step'])
                avg_grad.add_(grad)
                avg_grad.div_(state['step']+1)

                delta_k = torch.add(grad, -1, avg_grad)

                if state['step'] == 0:
                    state['v_kk'] = torch.sum(delta_k*delta_k).item()
                else:
                    state['v_kk']*=group['rho']
                    state['v_kk']+=(1-group['rho'])*torch.sum(delta_k*delta_k).item()
                state['v_k'] = max([state['v_kk'], state['v_k']])

                h_k = math.sqrt((state['step']+1)*state['v_k'])

                alpha_k_1 = 2/(state['step']+3)


                coef = -1/(gamma_k+group['beta']*h_k)
                x_k_1 = state['x_k']
                x_k_1.add_(coef, grad)

                p.data.mul_(1-alpha_k_1)
                p.data.add_(alpha_k_1, x_k_1)
                p.data.add_((1 - alpha_k_1)*state['alpha_k']*coef, grad)

                state['alpha_k'] = alpha_k_1
                state['step'] += 1

        return loss

# accelerated SGD
class AccSGD(Optimizer):

    def __init__(self, params, lr=required, kappa = 1000.0, xi = 10.0, smallConst = 0.7, weight_decay=0):
        defaults = dict(lr=lr, kappa=kappa, xi=xi, smallConst=smallConst,
                        weight_decay=weight_decay)
        super(AccSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AccSGD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            large_lr = (group['lr']*group['kappa'])/(group['smallConst'])
            Alpha = 1.0 - ((group['smallConst']*group['smallConst']*group['xi'])/group['kappa'])
            Beta = 1.0 - Alpha
            zeta = group['smallConst']/(group['smallConst']+Beta)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    param_state['momentum_buffer'] = copy.deepcopy(p.data)
                buf = param_state['momentum_buffer']
                buf.mul_((1.0/Beta)-1.0)
                buf.add_(-large_lr,d_p)
                buf.add_(p.data)
                buf.mul_(Beta)

                p.data.add_(-group['lr'],d_p)
                p.data.mul_(zeta)
                p.data.add_(1.0-zeta,buf)

        return loss


# adaptive SGD
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
