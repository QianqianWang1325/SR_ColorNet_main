import torch

# SGD

def SGD_Momentum_Optimizer(model, learn_rate=1e-2):
    return torch.optim.SGD(model, lr=learn_rate, momentum=0.9, dampening=0, weight_decay=0, nesterov=False)


def SGD_Momentum_Nesterov_Optimizer(model, learn_rate=1e-2):
    # 采用nesterov（牛顿动量法），dampening必须为 0
    # 与Momentum唯一区别就是，计算梯度的不同，Nesterov先用当前的速度v更新一遍参数，在用更新的临时参数计算梯度。
    return torch.optim.SGD(model, lr=learn_rate, momentum=0.9, dampening=0, weight_decay=0, nesterov=True)


def SGD_Optimizer(model, learn_rate=1e-2):
    # 不带动量的SGD
    return torch.optim.SGD(model, lr=learn_rate, momentum=0, dampening=0, weight_decay=0, nesterov=False)


def ASGD_Optimizer(model, learn_rate=1e-2):
    # 随机平均梯度下降，用空间换时间的一种SGD
    return torch.optim.ASGD(model, lr=learn_rate, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

# SGD

def Rprop_Optimizer(model, learn_rate=1e-2):
    # 弹性反向传播，该优化方法适用于full-batch，不适用于mini-batch
    return torch.optim.Rprop(model, lr=learn_rate, etas=(0.5, 1.2), step_sizes=(1e-06, 50))


def Adam_Oprimizer(model, learn_rate=1e-4):
    # 结合了Momentum和RMSprop，并进行了偏差修正
    return torch.optim.Adam(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


def Adamax_Oprimizer(model, learn_rate=1e-2):
    # 对Adam增加了一个学习率上限的概念
    return torch.optim.Adamax(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)


def SparseAdam_Oprimizer(model, learn_rate=1e-2):
    # 针对稀疏张量的一种“阉割版”Adam优化方法。
    return torch.optim.SparseAdam(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08)


def AdamW_Oprimizer(model, learn_rate=1e-3):
    # 优点：比Adam收敛得更快
    # 缺点：只有fastai使用，缺乏广泛的框架
    return torch.optim.AdamW(model, lr=learn_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01,
                             amsgrad=False)


def Adagrad_Oprimizer(model, learn_rate=1e-2):
    # 自适应优化方法，是自适应的为各个参数分配不同的学习率。这个学习率的变化，会受到梯度的大小和迭代次数的影响。梯度越大，学习率越小；梯度越小，学习率越大。缺点是训练后期，学习率过小，因为Adagrad累加之前所有的梯度平方作为分母
    return torch.optim.Adagrad(model, lr=learn_rate, lr_decay=0, weight_decay=0,
                               initial_accumulator_value=0)


def Adadelta_Oprimizer(model, learn_rate=1e-2):
    # Adagrad的改进,分母中采用距离当前时间点比较近的累计项，这可以避免在训练后期，学习率过小
    return torch.optim.Adadelta(model, lr=learn_rate, rho=0.9, eps=1e-06, weight_decay=0)


def RMSprop_Optimizer(model, learn_rate=1e-2):
    # Adagrad的改进,RMSprop采用均方根作为分母，可缓解Adagrad学习率下降较快的问题。并且引入均方根，可以减少摆动
    return torch.optim.RMSprop(model, lr=learn_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                               centered=False)


def LBFGS_Oprimizer(model, learn_rate=1):
    # 拟牛顿算法,节省内存
    return torch.optim.LBFGS(model, lr=learn_rate, max_iter=20, max_eval=None, tolerance_grad=1e-05,
                             tolerance_change=1e-09, history_size=100, line_search_fn=None)


def get_optimizer(name, model, learn_rate):
    if name == 'SGD_ND':
        return SGD_Momentum_Nesterov_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'SGD_M':
        return SGD_Momentum_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'SGD':
        return SGD_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'ASGD':
        return ASGD_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'RPROP':
        return Rprop_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAM':
        return Adam_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAMX':
        return Adamax_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'SPARSEADAM':
        return SparseAdam_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAMW':
        return AdamW_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADAGRAD':
        return Adagrad_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'ADADELTA':
        return Adadelta_Oprimizer(model=model, learn_rate=learn_rate)
    elif name == 'RMSPROP':
        return RMSprop_Optimizer(model=model, learn_rate=learn_rate)
    elif name == 'LBFGS':
        return LBFGS_Oprimizer(model=model, learn_rate=learn_rate)
