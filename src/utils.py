import lime
import lime.lime_tabular
import numpy as np
from sklearn.linear_model import LogisticRegression

# find x's that need recourse
def recourse_needed(predict_fn, X, y_target=1):
    indices = np.where(predict_fn(X) == 1-y_target)[0]
    return X[indices]

# recourse validity
def recourse_validity(predict_fn, recourses, y_target=1):
    return sum(predict_fn(recourses)==y_target)/len(recourses)

def recourse_expectation(predict_proba_fn, recourses):
    return sum(predict_proba_fn(recourses)[:,1]) / len(recourses)


def lime_explanation(pred_proba_fn, X_train, x):
    explainer = lime.lime_tabular.LimeTabularExplainer(training_data=X_train, mode='regression', discretize_continuous=False, feature_selection='none')
    exp = explainer.explain_instance(x, pred_proba_fn, num_features=X_train.shape[1], model_regressor=LogisticRegression())
    weights = exp.local_exp[1][0][1]
    bias = exp.intercept[1]
    return weights, bias

def sigmoid(x, weights, bias):
    f = 1 / (1 + np.exp(-(np.matmul(x, weights) + bias)))
    return f

def l1_cost(x1, x2):
    return np.linalg.norm(x1-x2, 1, -1)

def l2_cost(x1, x2):
    return np.linalg.norm(x1-x2, 2, -1)

def linf_cost(x1, x2):
    return np.linalg.norm(x1-x2, np.inf)

def generate_grid(center, delta, n, ord=None):
    linspaces = [np.linspace(center[i]-delta, center[i]+delta, n) for i in range(len(center))]
    grids = np.meshgrid(*linspaces)
    points = np.stack([grid.reshape(-1) for grid in grids]).T
    if ord != None:
        mask = np.linalg.norm(points-center, ord=ord, axis=1) <= delta
        points = points[mask]
    return points    

def hex2rgba(color, a=1):
    h = color.lstrip('#')
    c = f'rgba{tuple(int(h[i:i+2], 16) for i in (0, 2, 4))}'[:-1] + f', {a})'
    return c

def argmaxs(a):
    max_i = np.argmax(np.abs(a))
    sorted_is = np.argsort(np.abs(a))
    i = np.where(sorted_is==max_i)[0][0]
    return sorted_is[i:]

def pareto_frontier(A, B):
    temp = np.column_stack((A, B))
    is_dominated = np.ones(temp.shape[0], dtype=bool)
    
    for i, c in enumerate(temp):
        if is_dominated[i]:
            is_dominated[is_dominated] = np.any(temp[is_dominated] < c, axis=1)
            is_dominated[i] = True
    return is_dominated