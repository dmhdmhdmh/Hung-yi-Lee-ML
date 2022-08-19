# Baseline
from utils import *

class baseline(object):
    """
    baseline technique: do nothing in regularization term [initialize and all weight is zero]
    """

    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        # extract all parameters in models
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        # store current parameters
        self.p_old = {}

        # generate weight matrix
        self._precision_matrices = self._calculate_importance()

        for n, p in self.params.items():
            # keep the old parameter in self.p_old
            self.p_old[n] = p.clone().detach()

    def _calculate_importance(self):
        precision_matrices = {}
        # initialize weight matrix（fill zero）
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)

        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self.p_old[n]) ** 2
            loss += _loss.sum()
        return loss

    def update(self, model):
        # do nothing
        return


# Baseline
print("RUN BASELINE")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# initialize lifelong learning object (baseline class) without adding any regularization term.
lll_object = baseline(model=model, dataloader=None, device=device)
lll_lambda = 0.0
baseline_acc = []
task_bar = tqdm.auto.trange(len(train_dataloaders), desc="Task   1")

# iterate training on each task continually.
for train_indexes in task_bar:
    # Train each task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task,
                               lll_object, lll_lambda, evaluate=evaluate, device=device,
                               test_dataloaders=test_dataloaders[:train_indexes + 1])

    # get model weight to baseline class and do nothing!
    lll_object = baseline(model=model, dataloader=train_dataloaders[train_indexes], device=device)

    # new a optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect average accuracy in each epoch
    baseline_acc.extend(acc_list)

    # display the information of the next task.
    task_bar.set_description_str(f"Task  {train_indexes + 2:2}")

# average accuracy in each task per epoch!
print(baseline_acc)
print("==================================================================================================")