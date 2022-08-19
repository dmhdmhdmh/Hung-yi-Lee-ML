# Baseline
from utils import *


# SI
class si(object):
    """
    @article{kirkpatrick2017overcoming,
        title={Overcoming catastrophic forgetting in neural networks},
        author={Kirkpatrick, James and Pascanu, Razvan and Rabinowitz, Neil and Veness, Joel and Desjardins, Guillaume and Rusu, Andrei A and Milan, Kieran and Quan, John and Ramalho, Tiago and Grabska-Barwinska, Agnieszka and others},
        journal={Proceedings of the national academy of sciences},
        year={2017},
        url={https://arxiv.org/abs/1612.00796}
    }
    """

    def __init__(self, model, dataloader, epsilon, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.epsilon = epsilon
        # extract all parameters in models
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        self._n_p_prev, self._n_omega = self._calculate_importance()
        self.W, self.p_old = self._init_()

    def _init_(self):
        W = {}
        p_old = {}
        for n, p in self.model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                W[n] = p.data.clone().zero_()
                p_old[n] = p.data.clone()
        return W, p_old

    def _calculate_importance(self):
        n_p_prev = {}
        n_omega = {}

        if self.dataloader != None:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:
                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                    W = getattr(self.model, '{}_W'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W / (p_change ** 2 + self.epsilon)
                    try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = omega + omega_add
                    n_omega[n] = omega_new
                    n_p_prev[n] = p_current

                    # Store these new values in the model
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p_current)
                    self.model.register_buffer('{}_SI_omega'.format(n), omega_new)

        else:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:
                    n_p_prev[n] = p.detach().clone()
                    n_omega[n] = p.detach().clone().zero_()
                    self.model.register_buffer('{}_SI_prev_task'.format(n), p.detach().clone())
        return n_p_prev, n_omega

    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]
                _loss = omega * (p - prev_values) ** 2
                loss += _loss.sum()
        return loss

    def update(self, model):
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                if p.grad is not None:
                    self.W[n].add_(-p.grad * (p.detach() - self.p_old[n]))
                    self.model.register_buffer('{}_W'.format(n), self.W[n])
                self.p_old[n] = p.detach().clone()
        return


# SI
print("RUN SI")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object = si(model=model, dataloader=None, epsilon=0.1, device=device)
lll_lambda = 1
si_acc = []
task_bar = tqdm.auto.trange(len(train_dataloaders), desc="Task   1")

for train_indexes in task_bar:
    # Train Each Task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object,
                               lll_lambda, evaluate=evaluate, device=device,
                               test_dataloaders=test_dataloaders[:train_indexes + 1])

    # get model weight and calculate guidance for each weight
    lll_object = si(model=model, dataloader=train_dataloaders[train_indexes], epsilon=0.1, device=device)

    # New a Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect average accuracy in each epoch
    si_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes + 2:2}")

# average accuracy in each task per epoch!
print(si_acc)
print("==================================================================================================")