import torch
import torch.nn.functional as F
from torch import nn


class EWC(nn.Module):
    def __init__(self, ewc_lambda=0.5):
        super().__init__()
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = {}
        self.optpar_dict = {}
        self.num_samples = 512

    def before_task(self, model, dataloader, task_id=0):
        self.fisher_dict[task_id] = {}
        self.optpar_dict[task_id] = {}
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        sample_counter = 0
        losses = torch.tensor(0.0).to(model.device)
        for inputs in dataloader:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            output = model(**inputs["tokenized_query"])

            logits = output.logits  # [bz, smtid_length, vocab_size]

            bz, smtid_length = inputs["labels"].size()
            losses += F.cross_entropy(
                logits.view(bz * smtid_length, -1), inputs["labels"].view(-1)
            )

            sample_counter += bz
            if sample_counter >= self.num_samples:
                print("Total number of samples for ewc:", self.num_samples)
                break
        from pdb import set_trace as st; st()
        losses.backward()
        # gradients accumulated can be used to calculate fisher
        for name, param in model.named_parameters():
            self.optpar_dict[task_id][name] = param.data.clone()
            self.fisher_dict[task_id][name] = param.grad.data.clone().pow(2)
        from pdb import set_trace as st; st()
        model.zero_grad()

    def forward(self, model, task_id):
        ewc_loss = torch.tensor(0.0).to(model.device)
        for task in range(1, task_id):
            for name, param in model.named_parameters():
                fisher = self.fisher_dict[task][name]
                optpar = self.optpar_dict[task][name]
                ewc_loss += (fisher * (optpar - param).pow(2)).sum() * self.ewc_lambda
        return ewc_loss