# foundation_hack

This is a toy project of applying pretrained models to games.

The current implementation (Nov 3rd 2022) uses BEiT on Atari mspacman.

The setup is easy for plugging in other games or models:
- To use another game:
    - register the game environment under src/env
    - rewrite the data loader under src/pipeline
- To use another model:
    - register the model under src/models