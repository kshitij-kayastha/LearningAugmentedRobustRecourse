import tqdm
import torch
import numpy as np
import torch.optim as optim
from typing import List, Callable
from torch.autograd import grad
from scipy.optimize import linprog

from copy import deepcopy
from abc import ABC, abstractmethod

from utils import *


class RecourseCost:
    def __init__(self, x_0: np.ndarray, lamb: float, cost_fn: Callable = l1_cost) -> None:
        self.x_0 = x_0
        self.lamb = lamb
        self.cost_fn = cost_fn
    
    def __call__(self, x: np.ndarray, weights: np.ndarray, bias: np.ndarray, return_breakdowns: bool = False):
        f_x = 1 / (1 + np.exp(-(np.matmul(x, weights) + bias)))
        bce_loss = -np.log(f_x)
        cost = self.cost_fn(self.x_0, x)
        loss = bce_loss + self.lamb*cost
        if return_breakdowns:
            return bce_loss, cost, loss
        else:
            return loss
        
    def eval_nonlinear(self, x, model, breakdown: bool = False):
        if isinstance(x, np.ndarray):
            x = torch.tensor(deepcopy(x)).float()
        f_x = model(x)
        loss_fn = torch.nn.BCELoss(reduction='mean')
        bce_loss = loss_fn(f_x, torch.ones(f_x.shape).float())
        cost = torch.dist(x, torch.tensor(self.x_0).float(), 1)
        recourse_cost = bce_loss + self.lamb*cost
        if breakdown:
            return bce_loss.detach().item(), cost.detach().item(), recourse_cost.detach().item()
        return recourse_cost.detach().item()

class RobustRecourse(ABC):
    def __init__(self) -> None:
        super().__init__()
        
    @abstractmethod
    def get_recourse(self, x):
        pass
    
    @abstractmethod
    def calc_theta_adv(self, x):
        pass
    

class LAROAR(RobustRecourse):
    def __init__(self, weights: np.ndarray, bias: np.ndarray = None, alpha: float = 0.1, lamb: float = 0.1, y_target: float = 1.):
        self.weights = weights
        self.bias = bias
        self.alpha = alpha
        self.lamb = lamb
        self.y_target = y_target

    def calc_delta(self, w: float, c: float):
        if (w > self.lamb):
            delta = ((np.log((w - self.lamb)/self.lamb) - c) / w)
            if delta < 0: delta = 0.
        elif (w < -self.lamb):
            delta = (np.log((-w - self.lamb)/self.lamb) - c) / w
            if delta > 0: delta = 0.
        else:
            delta = 0.
        return delta   
    
    def calc_augmented_delta(self, x: np.ndarray, i: int, theta: tuple[np.ndarray, np.ndarray], theta_p: tuple[np.ndarray, np.ndarray], beta: float, J: RecourseCost):
        n = 201
        delta = 10
        deltas = np.linspace(-delta, delta, n)
        
        x_rs = np.tile(x, (n, 1))
        x_rs[:, i] += deltas
        vals = beta*J(x_rs, *theta) + (1-beta)*J(x_rs, *theta_p)
        min_i = np.argmin(vals)
        return deltas[min_i]
            
    def can_change_sign(self, w: float):
        return np.sign(w+self.alpha) != np.sign(w-self.alpha)

    def get_max_idx(self, weights: np.ndarray, changed: List):
        weights_copy = deepcopy(weights)
        while True:
            idx = np.argmax(np.abs(weights_copy))
            if not changed[idx]:
                return idx
            else:
                weights_copy[idx] = 0.
        
    def calc_theta_adv(self, x: np.ndarray):
        weights_adv = self.weights - (np.sign(x) * self.alpha)
        for i in range(len(x)):
            if np.sign(x[i]) == 0:
                weights_adv[i] = weights_adv[i] - (np.sign(weights_adv[i]) * self.alpha)
        bias_adv = self.bias - self.alpha
        
        return weights_adv, bias_adv
    
    
    def get_robust_recourse(self, x_0: np.ndarray):
        x = deepcopy(x_0)
        
        weights, bias = self.calc_theta_adv(x)
        changed = [False] * len(weights)
        while True:
            if np.all(changed):
                break
    
            i = self.get_max_idx(weights, changed)
            x_i, w_i = x[i], weights[i]
            
            c = np.matmul(x, weights) + bias
            delta = self.calc_delta(w_i, c[0])

            if delta == 0:
                break        
            if (np.sign(x_i+delta) == np.sign(x_i)) or (x_i == 0):
                x[i] = x_i + delta
                changed[i] = True
            else:
                x[i] = 0
                weights[i] = self.weights[i] + (np.sign(x_i) * self.alpha)
                if self.can_change_sign(self.weights[i]):
                    changed[i] = True
        
        return x
        
    def get_consistent_recourse(self, x_0: np.ndarray, theta_p: tuple[np.ndarray, np.ndarray]):
        x = deepcopy(x_0)
        weights, bias = theta_p
        
        i = np.argmax(np.abs(weights))
        x_i, w_i = x[i], weights[i]
        c = np.matmul(x, weights) + bias
        delta = self.calc_delta(w_i, c)
        x[i] = x_i + delta
        
        return x
        
    def get_augmented_recourse(self, x_0: np.ndarray, theta_p: tuple[np.ndarray, np.ndarray], beta: float):
        x = deepcopy(x_0)
        J = RecourseCost(x_0, self.lamb)
        
        weights, bias = self.calc_theta_adv(x)
        weights_p, bias_p = theta_p
        
        changed = [False] * len(weights)
        while True:
            if np.all(changed):
                break
            
            min_val = 1e6
            min_i = 0
            for i in range(len(x)):
                if changed[i]: continue
                delta = self.calc_augmented_delta(x, i, (weights, bias), (weights_p, bias_p), beta, J)
                x_new = deepcopy(x)
                x_new[i] += delta
                val = beta*J(x_new, weights, bias) + (1-beta)*J(x_new, weights_p, bias_p)
                if val < min_val:
                    min_val = val
                    min_i = i
                    min_delta = delta
                    
            i = min_i
            delta = min_delta
            x_i = x[i]
        
            if (np.sign(x_i+delta) == np.sign(x_i)) or (x_i == 0.):
                if (x_i == 0) and np.sign(x_0[i]) == np.sign(delta):
                    delta = 0
                x[i] = x_i + delta
                changed[i] = True
            else:
                x[i] = 0
                weights[i] = self.weights[i] + (np.sign(x_i) * self.alpha)
                if self.can_change_sign(self.weights[i]):
                    changed[i] = True
        
        return x
        
    def get_recourse(self, x_0: np.ndarray, beta: float, theta_p: tuple[np.ndarray, np.ndarray] = None):
        
        if beta == 1.:
            return self.get_robust_recourse(x_0)
        elif beta == 0.:
            return self.get_consistent_recourse(x_0, theta_p)
        else:
            return self.get_augmented_recourse(x_0, theta_p, beta)
        
    def choose_lambda(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        lambdas = np.arange(0.1, 1.1, 0.1).round(1)
        v_old = 0
        print('Choosing lambda')
        # for i in tqdm.trange(len(lambdas)):
        for i in range(len(lambdas)):
            lamb = lambdas[i]
            self.lamb = lamb
            recourses = []
            for xi in tqdm.trange(len(recourse_needed_X), desc=f'lambda={lamb}'):
                if xi >= 145:
                    continue
                x = recourse_needed_X[xi]
                if self.weights is None and self.bias is None:
                    # set seed for lime
                    np.random.seed(xi)
                    weights, bias = lime_explanation(predict_fn, X_train, x)
                    weights, bias = np.round(weights, 4), np.round(bias, 4)
                    self.weights = weights
                    self.bias = bias

                    x_r = self.get_recourse(x, 1.)

                    self.weights = None
                    self.bias = None
                else:
                    x_r = self.get_recourse(x, 1.)
                recourses.append(x_r)

            if predict_proba_fn:
                v = recourse_expectation(predict_proba_fn, recourses)
            else:
                v = recourse_validity(predict_fn, recourses, self.y_target)
            if v >= v_old:
                v_old = v
            else:
                li = max(0, i - 1)
                return lambdas[li]
        return lamb
    

class ROAR(RobustRecourse):
    def __init__(self, weights: np.ndarray, bias: np.ndarray = None, alpha: float = 0.1, lamb: float = 0.1, y_target: float = 1.):
        self.set_weights(weights)
        self.set_bias(bias)
        self.alpha = alpha
        self.lamb = lamb
        self.y_target = torch.tensor(y_target).float()
    
    def set_weights(self, weights: np.ndarray):
        if weights is not None:
            self.weights = torch.from_numpy(weights).float()
        else:
            self.weights = None
        
    def set_bias(self, bias: np.ndarray):
        if bias is not None:
            self.bias = torch.from_numpy(bias).float()
        else:
            self.bias = None
    
    def l1_cost(self, x_new, x):
        return torch.dist(x_new, x, 1)
        
    def calc_theta_adv(self, x):
        theta = torch.cat((self.weights, self.bias), 0)
        x = torch.cat((x, torch.ones(1)), 0)

        loss_fn = torch.nn.BCELoss()

        A_eq = np.empty((0, len(theta)), float)

        b_eq = np.array([])

        theta.requires_grad = True 
        f_x = torch.nn.Sigmoid()(torch.matmul(theta, x))
        w_loss = loss_fn(f_x, self.y_target)
        gradient_w_loss = grad(w_loss, theta)[0]

        c = list(np.array(gradient_w_loss) * np.array([-1] * len(gradient_w_loss)))
        bound = (-self.alpha, self.alpha)
        bounds = [bound] * len(gradient_w_loss)

        res = linprog(c, bounds=bounds, A_eq=A_eq, b_eq=b_eq, method='simplex')
        alpha_opt = res.x  # the delta value that maximizes the function
        weights_alpha, bias_alpha = torch.from_numpy(alpha_opt[:-1]).float(), torch.from_numpy(alpha_opt[[-1]]).float()
        
        weights = self.weights + weights_alpha
        bias = self.bias + bias_alpha
        
        return weights.detach().numpy(), bias.detach().numpy()
        

    def get_recourse(self, x, lr=1e-3, abstol=1e-4):
        torch.manual_seed(0)
         
        x_0 = torch.from_numpy(x).float()
        x_r = x_0.clone().requires_grad_()
            
        weights = self.weights
        bias = self.bias
    
        optimizer = optim.Adam([x_r], lr=lr)
        loss_fn = torch.nn.BCELoss()

        # Placeholders
        loss = torch.tensor(1)
        loss_diff = 1

        i = 0
        while loss_diff > abstol:
            loss_prev = loss.clone().detach()
            
            weights, bias = self.calc_theta_adv(x_r.clone().detach())
            weights, bias = torch.from_numpy(weights).float(), torch.from_numpy(bias).float()
            optimizer.zero_grad()
            
            f_x = torch.nn.Sigmoid()(torch.matmul(weights, x_r) + bias)[0]
            bce_loss = loss_fn(f_x, self.y_target)
            cost = self.l1_cost(x_r, x_0)
            loss = bce_loss + self.lamb*cost
            
            loss.backward()
            optimizer.step()
            
            loss_diff = torch.dist(loss_prev, loss, 1)

            i += 1
        
        return x_r.detach().numpy(), np.hstack((weights.detach().numpy(), bias.detach().numpy()))
    
    def choose_lambda(self, recourse_needed_X, predict_fn, X_train=None, predict_proba_fn=None):
        lambdas = np.arange(0.1, 1.1, 0.1).round(1)
        v_old = 0
        print('Choosing lambda')
        for i in tqdm.trange(len(lambdas)):
            lamb = lambdas[i]
            self.lamb = lamb
            recourses = []
            for xi, x in enumerate(recourse_needed_X):
                if self.weights is None and self.bias is None:
                    # set seed for lime
                    np.random.seed(xi)
                    weights, bias = lime_explanation(predict_fn, X_train, x)
                    weights, bias = np.round(weights, 4), np.round(bias, 4)
                    self.set_weights(weights)
                    self.set_bias(bias)

                    x_r, _ = self.get_recourse(x)

                    self.weights = None
                    self.bias = None
                else:
                    x_r, _ = self.get_recourse(x)
                recourses.append(x_r)
            
            if predict_proba_fn:
                v = recourse_expectation(predict_proba_fn, recourses)
            else:
                v = recourse_validity(predict_fn, recourses, self.y_target.item())
        
            if v >= v_old:
                v_old = v
            else:
                li = max(0, i - 1)
                return lambdas[li]
        return lamb