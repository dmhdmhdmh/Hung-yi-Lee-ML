# Baseline
from utils import *


class mas(object):
    """
    @article{aljundi2017memory,
        title={Memory Aware Synapses: Learning what (not) to forget},
        author={Aljundi, Rahaf and Babiloni, Francesca and Elhoseiny, Mohamed and Rohrbach, Marcus and Tuytelaars, Tinne},
        booktitle={ECCV},
        year={2018},
        url={https://eccv2018.org/openaccess/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf}
    }
    """

    def __init__(self, model: nn.Module, dataloader, device, prev_guards=[None]):
        self.model = model
        self.dataloader = dataloader
        # extract all parameters in models
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        # initialize parameters
        self.p_old = {}

        self.device = device

        # save previous guards
        self.previous_guards_list = prev_guards

        # generate Omega(Ω) matrix for MAS
        self._precision_matrices = self.calculate_importance()

        # keep the old parameter in self.p_old
        for n, p in self.params.items():
            self.p_old[n] = p.clone().detach()

    def calculate_importance(self):
        precision_matrices = {}
        # initialize Omega(Ω) matrix（all filled zero）
        for n, p in self.params.items():
            precision_matrices[n] = p.clone().detach().fill_(0)
            for i in range(len(self.previous_guards_list)):
                if self.previous_guards_list[i]:
                    precision_matrices[n] += self.previous_guards_list[i][n]

        self.model.eval()
        if self.dataloader is not None:
            num_data = len(self.dataloader)
            for data in self.dataloader:
                self.model.zero_grad()
                output = self.model(data[0].to(self.device))
                ################################################################
                #####  TODO: generate Omega(Ω) matrix for MAS.  #####
                ################################################################
                pass
                ################################################################

            precision_matrices = {n: p for n, p in precision_matrices.items()}
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


# MAS
print("RUN MAS")
model = Model()
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

lll_object = mas(model=model, dataloader=None, device=device)
lll_lambda = 0.1
mas_acc = []
task_bar = tqdm.auto.trange(len(train_dataloaders), desc="Task   1")
prev_guards = []

for train_indexes in task_bar:
    # Train Each Task
    model, _, acc_list = train(model, optimizer, train_dataloaders[train_indexes], args.epochs_per_task, lll_object,
                               lll_lambda, evaluate=evaluate, device=device,
                               test_dataloaders=test_dataloaders[:train_indexes + 1])

    # get model weight and calculate guidance for each weight
    prev_guards.append(lll_object._precision_matrices)
    lll_object = mas(model=model, dataloader=train_dataloaders[train_indexes], device=device, prev_guards=prev_guards)

    # New a Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Collect average accuracy in each epoch
    mas_acc.extend(acc_list)
    task_bar.set_description_str(f"Task  {train_indexes + 2:2}")

# average accuracy in each task per epoch!
print(mas_acc)
print("==================================================================================================")