import time

from tqdm import tqdm
import torch
from src.utils.comm_util import get_device


class Trainer:
    def __init__(self,
        model,
        optimizer,
        batch_size,
        training_data_provider,
        validation_data_provider,
        lr_scheduler=None,
        eval_fns=[]):
        self.model = model
        self.optim = optimizer
        self.training_data_provider = training_data_provider
        self.lr_scheduler = lr_scheduler
        self.device = get_device()

    def train(self, n_iters):
        begin_time = time.perf_counter()
        results_iter = {}

        for iter in range(n_iters):
            print(f'***iteration {iter}')
            state_batch, action_batch = self.training_data_provider.get_batch()
            self.model.train() # training mode
            batch = (state_batch, action_batch)
            loss_ep = self.train_iteration(batch)
            results_iter[iter] = loss_ep
            print(f"loss: {results_iter}")

        dur = time.perf_counter() - begin_time
        print(f"===training timecost: {dur}")

    def train_iteration(self, batch):
        begin_time = time.perf_counter()

        self.optim.zero_grad()
        state_batch, real_action_batch = batch
        pred_action_batch = self.model(**state_batch) # feed the state batch into our model
        loss = self.model.loss_on_batch(real_action_batch, pred_action_batch)
        self.optim.zero_grad()
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
            x_batch = x_batch.type(torch.FloatTensor).to(self.device) # TODO type?
            y_batch = y_batch.type(torch.FloatTensor).to(self.device)
            loss = self.train_iteration((x_batch, y_batch))
            loss_ep+=loss.detach().item()
            n_batch+=1
            pbar.set_description(f"train loss: {loss_ep/n_batch:.4f}")
        return loss_ep

    # Do a training iteration:
    # 1) Evaluate the model through the pre-supplied eval functions.
    # 2) Train on #num_steps batches on the training data
    # 3) Do validation on #num_valid_steps batches of data from the validaiton data.
    # def train_iteration(self, num_train_steps, num_valid_steps, iter_num=0):
    #     print("=" * 80)
    #     print(f"Iteration {iter_num}")

    #     logs = dict()

    #     # train for fixed number of timesteps
    #     train_outputs = []
    #     train_start = time.time()
    #     self.model.train()

    #     for _ in tqdm(range(num_train_steps)):
    #         try:
    #             train_batch = self.training_data.get_batch()
    #         except (OutOfDataException, StopIteration):
    #             # We have decided to train on num_steps batches for training. Reset the
    #             # data provider and keep going (losing a single iteration)
    #             self.training_data.reset()
    #         else:
    #             train_output = self.step(train_batch, is_training=True)
    #             train_outputs.append(train_output)
    #             if self.lr_scheduler is not None:
    #                 self.lr_scheduler.step()
    #             pass

    #         # _, _, _, _, _, timesteps, _ = train_batch
    #         # sequence_traj_tag = timesteps[:, 0]
    #         # for tag in sequence_traj_tag:
    #         #     hist_stats[tag] += 1

    #     print(train_output)

    #     logs["time/training"] = time.time() - train_start
    #     training_outputs_dict = self._concat_dicts(train_outputs)
    #     self._log_means_stds(training_outputs_dict, prefix="training")

    #     # validation
    #     val_outputs = []
    #     valid_start = time.time()
    #     self.model.eval()
    #     for _ in tqdm(range(num_valid_steps)):
    #         try:
    #             val_batch = self.validation_data.get_batch()
    #         except (OutOfDataException, StopIteration):
    #             # We have utilised all the validation data we have, stop doing validation
    #             self.validation_data.reset()
    #             continue
    #         else:
    #             torch.cuda.empty_cache()
    #             valid_output = self.step(val_batch, is_training=False)  # train=False => we're doing validation
    #             val_outputs.append(valid_output)
    #     print(valid_output)

    #     logs["time/offline_eval"] = time.time() - valid_start
    #     validation_outputs_dict = self._concat_dicts(val_outputs)
    #     self._log_means_stds(validation_outputs_dict, prefix="offline_eval")

    #     if self.online_eval:
    #         # if self.env_type in {"mujoco", "mujoco_maze2d_large", "mujoco_maze2d_medium", "mujoco_maze2d_umaze", "mujoco_antmaze_umaze", "mujoco_antmaze_umaze_diverse", "mujoco_antmaze_medium", "mujoco_antmaze_medium_diverse", "mujoco_antmaze_large_diverse", "mujoco_antmaze_large"}: # add online evaluation
    #         if self.env_type in {"mujoco", "mujoco_maze2d_large", "mujoco_maze2d_medium", "mujoco_maze2d_umaze"}: # add online evaluation
    #             online_eval_start = time.time()
    #             self.model.eval()
    #             eval_info = self.online_evaluate_mujoco(self.training_data.env_id, self.training_data)

    #             logs["time/online_eval"] = time.time() - online_eval_start
    #             self._log_means_stds(eval_info, prefix="online_eval")
    #             print(eval_info)

    #         logs["time/total"] = time.time() - self.start_time

    #     for k, v in logs.items():
    #         mlflow.log_metric(k, v)
    #     print("=" * 80)

