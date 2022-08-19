# Baseline
from utils import *


class rwalk(object):
    def __init__(self, model, dataloader, epsilon, device, prev_guards=[None]):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.epsilon = epsilon
        self.update_ewc_parameter = 0.4
        # extract model parameters and store in dictionary
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        # initialize the guidance matrix
        self._means = {}

        self.previous_guards_list = prev_guards

        # Generate Fisher (F) Information Matrix
        self._precision_matrices = self._calculate_importance_ewc()

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

        if self.dataloader is not None:
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                if p.requires_grad:
                    # Find/calculate new values for quadratic penalty on parameters
                    p_prev = getattr(self.model, '{}_SI_prev_task'.format(n))
                    W = getattr(self.model, '{}_W'.format(n))
                    p_current = p.detach().clone()
                    p_change = p_current - p_prev
                    omega_add = W / (1.0 / 2.0 * self._precision_matrices[n] * p_change ** 2 + self.epsilon)
                    try:
                        omega = getattr(self.model, '{}_SI_omega'.format(n))
                    except AttributeError:
                        omega = p.detach().clone().zero_()
                    omega_new = 0.5 * omega + 0.5 * omega_add
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

    def _calculate_importance_ewc(self):
        precision_matrices = {}
        for n, p in self.params.items():
            # initialize Fisher (F) matrix（all fill zero）
            n = n.replace('.', '__')
            precision_matrices[n] = p.clone().detach().fill_(0)
            for i in range(len(self.previous_guards_list)):
                if self.previous_guards_list[i]:
                    precision_matrices[n] += self.previous_guards_list[i][n]

        self.model.eval()
        if self.dataloader is not None:
            number_data = len(self.dataloader)
            for n, p in self.model.named_parameters():
                n = n.replace('.', '__')
                precision_matrices[n].data *= (1 - self.update_ewc_parameter)
            for data in self.dataloader:
                self.model.zero_grad()
                input = data[0].to(self.device)
                output = self.model(input)
                label = data[1].to(self.device)

                # generate Fisher(F) matrix for RWALK
                loss = F.nll_loss(F.log_softmax(output, dim=1), label)
                loss.backward()

                for n, p in self.model.named_parameters():
                    n = n.replace('.', '__')
                    precision_matrices[n].data += self.update_ewc_parameter * p.grad.data ** 2 / number_data

            precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            n = n.replace('.', '__')
            if p.requires_grad:
                prev_values = self._n_p_prev[n]
                omega = self._n_omega[n]
                # Generate regularization term  _loss by omega and Fisher Matrix
                _loss = (omega + self._precision_matrices[n]) * (p - prev_values) ** 2
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

# RWalk
print("RUN Rwalk")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object=rwalk(model=model, dataloader=None, epsilon=0.1, device=device)
lll_lambda=100
rwalk_acc = []
task_bar = tqdm.auto.trange(len(train_dataloaders),desc="Task   1")
prev_guards = []

for train_indexes in task_bar:
  model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object, lll_lambda, evaluate=evaluate,device=device, test_dataloaders=test_dataloaders[:train_indexes+1])
  prev_guards.append(lll_object._precision_matrices)
  lll_object=rwalk(model=model, dataloader=train_dataloaders[train_indexes], epsilon=0.1, device=device, prev_guards=prev_guards)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
  rwalk_acc.extend(acc_list)
  task_bar.set_description_str(f"Task  {train_indexes+2:2}")

# average accuracy in each task per epoch!
print(rwalk_acc)
print("==================================================================================================")

