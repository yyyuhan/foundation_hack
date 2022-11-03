
from tests.test_dataloader import DATA_DIR
from src.utils.types import GameName
from src.pipeline.load_atari import MspacmanDataProvider
from src.models.beit import ImageBeit
from src.utils.comm_util import get_device
from src.training.trainer import Trainer
from torch import optim

# pytest -s tests/test_trainer.py::test_trainer
def test_trainer():
    batch_size = 128
    dataloader = MspacmanDataProvider(DATA_DIR, GameName.ATARI_MSPACMAN, batch_size=batch_size)
    
    batch_size = 128
    model = ImageBeit(num_actions=18) # TODO dataset
    # model = model.to(device=get_device())
    optimizer = optim.AdamW(model.parameters(), lr=1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda steps: 1)
    trainer = Trainer(model, optimizer=optimizer, batch_size=batch_size, lr_scheduler=scheduler, training_data_provider=dataloader)
    trainer.train(n_iters=1)
    # model, optimizer, batch_size, training_data_provider, validation_data_provider, lr_scheduler=None, eval_fns=[]
    