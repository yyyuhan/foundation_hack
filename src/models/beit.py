import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BeitFeatureExtractor, BeitModel


class ImageBeit(nn.Module):
    def __init__(self, num_actions=0, device="cpu") -> None:
        super().__init__()
        self.fc1 = nn.Linear(768, 384)
        self.fc2 = nn.Linear(384, num_actions)

        self.loss_fn = nn.CrossEntropyLoss()

        self.beit_feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        self.beit_model = BeitModel.from_pretrained("microsoft/beit-base-patch16-224-pt22k")
        self.device = device
        # num_params = sum(param.numel() for param in self.beit_model.parameters())
        # 85.6m

    def forward(self, x):
        # x: a list of tensor (3*H*W)
        x = self.beit_feature_extractor(x, return_tensors="pt")
        x.to(self.device)
        with torch.no_grad():
            x = self.beit_model(**x).last_hidden_state  # the size is always (b*197*768)
        x = x[
            :, 0, :
        ]  # (b*768) for each image inside the batch, select the vector in position 0 as a representation of the image.
        x = F.relu(self.fc1(x.view(x.size(0), -1)))  # map x from (b*768) to (b*512)
        x = self.fc2(x)  # map x from  (b*512) to (b*num_actions)
        return x  # our job is finished. we do not need to touch the RL algorithm and training part. We only need to make sure the input x is in the corroct format

    def loss_on_batch(self, real_batch, pred_batch):
        loss = self.loss_fn(real_batch, pred_batch)
        return loss

    def load_checkpoint(self, model_path):
        self.load_state_dict(state_dict=torch.load(model_path))
