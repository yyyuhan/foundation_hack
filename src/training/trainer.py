import time

import torch
from tqdm import tqdm

from src.utils.comm_util import save_checkpoint
from src.utils.types import Constants


class Trainer:
    def __init__(
        self,
        model,
        optimizer=None,
        batch_size=None,
        training_data_provider=None,
        validation_data_provider=None,
        lr_scheduler=None,
        device="cpu",
        eval_fns=[],
    ):
        self.device = device
        self.model = model.to(device)
        self.optim = optimizer
        self.batch_size = batch_size
        self.training_data_provider = training_data_provider
        self.lr_scheduler = lr_scheduler

    def train(self, n_iters):
        begin_time = time.perf_counter()
        results_iter = {}

        for iter in range(n_iters):
            print(f"***iteration {iter}")
            state_batch, action_batch = self.training_data_provider.get_batch()
            # state_batch, action_batch = state_batch.to(self.device), action_batch.to(self.device)
            # required by BEiT: we must pass list[tensor] as input
            state_batch = [state.to(self.device) for state in state_batch[:]]
            action_batch = action_batch.to(self.device)
            self.model.train()  # training mode
            batch = (state_batch, action_batch)
            loss_ep = self.train_iteration(batch)
            results_iter[iter] = loss_ep.detach().cpu().item()
            print(f"loss: {results_iter}")

            if 0 == iter % Constants.CHECKPOINT_FREQUENCY:
                save_checkpoint(self.model, f"bc_{iter}")

        dur = time.perf_counter() - begin_time
        print(f"===training timecost: {dur}")

    def train_iteration(self, batch):
        begin_time = time.perf_counter()

        self.optim.zero_grad()
        state_batch, real_action_batch = batch
        pred_action_batch = self.model(state_batch)  # feed the state batch into our model
        loss = self.model.loss_on_batch(pred_action_batch, real_action_batch)
        loss.backward()
        # update parameters
        self.optim.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        dur = time.perf_counter() - begin_time
        print(f"***iteration timecost: {dur}")
        return loss

    def train_epoch(self, batches):
        pbar = tqdm(batches)
        loss_ep, n_batch = 0, 0
        for x_batch, y_batch in pbar:
            x_batch = x_batch.type(torch.FloatTensor).to(self.device)  # TODO type?
            y_batch = y_batch.type(torch.FloatTensor).to(self.device)
            loss = self.train_iteration((x_batch, y_batch))
            loss_ep += loss.detach().item()
            n_batch += 1
            pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
        return loss_ep

    def pred_action(self, state):
        state = torch.from_numpy(state.transpose(2, 0, 1))
        logits = self.model([state.to(self.device)])
        preds = torch.argmax(logits, dim=-1)
        return preds.item()
