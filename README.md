<!--
SPDX-FileCopyrightText: 2026 Alexandre Gomes Gaigalas <alganet@gmail.com>

SPDX-License-Identifier: ISC
-->

# Minimal Word Generator

Tiny character-level Transformer that trains a simple word generator from a word list. Good for learning basic training loops, checkpointing, and sampling.

- Generates words that resemble english
- Does the whole cycle super fast (seconds or minutes)
- Small Transformer LM (character-level)
- Epoch checkpoints and vocab saved to `out/`

## Requirements

Install dependencies:

```sh
pip install -r requirements.txt
```

## Quickstart

Run the trainer (it downloads data on first run):

```sh
python train.py
```

Once it trains, you can run it again to just re-generate samples:

```sh
python train.py
```

```
Device: cuda
Loading text...
Building/Loading vocab...
Loaded vocab from out/vocab.json
Splitting data...
Creating DataLoaders...
Initializing model...
Model already trained up to epoch 2, skipping training
Generating samples...
Generated word 1: coales
Generated word 2: tereed
Generated word 3: healable
Generated word 4: thines
Generated word 5: unitlerable
Generated word 6: loteroformated
Generated word 7: rearmaz
Generated word 8: Debrowing
Generated word 9: unpatates
Generated word 10: unatined
```

If you change hyperparameters, remove the generated files before re-training:

```sh
rm -rf out/
python train.py
```
