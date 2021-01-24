#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS224N 2018-19: Homework 3
run.py: Run the dependency parser.
Sahil Chopra <schopra8@stanford.edu>
"""
import argparse
import sys
from datetime import datetime
import os
import pickle
import math
import time

from torch import nn, optim
import torch
from tqdm import tqdm

from parser_model import ParserModel
from utils.parser_utils import minibatches, load_and_preprocess_data, AverageMeter

# -----------------
# Primary Functions
# -----------------
def train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005, device="cpu"):
    """ Train the neural dependency parser.

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param output_path (str): Path to which model weights and results are written.
    @param batch_size (int): Number of examples in a single batch
    @param n_epochs (int): Number of training epochs
    @param lr (float): Learning rate
    """
    best_dev_UAS = 0


    ### TODO:
    ###      1) Construct Adam Optimizer in variable `optimizer`
    ###      2) Construct the Cross Entropy Loss Function in variable `loss_func`
    ###
    ### Hint: Use `parser.model.parameters()` to pass optimizer
    ###       necessary parameters to tune.
    ### Please see the following docs for support:
    ###     Adam Optimizer: https://pytorch.org/docs/stable/optim.html
    ###     Cross Entropy Loss: https://pytorch.org/docs/stable/nn.html#crossentropyloss
    ### YOUR CODE HERE (~2-7 lines)
    optimizer = torch.optim.Adam(params=parser.model.parameters(), lr=lr)
    loss_func = nn.CrossEntropyLoss()

    ### END YOUR CODE

    for epoch in range(n_epochs):
        print("Epoch {:} out of {:}".format(epoch + 1, n_epochs))
        dev_UAS = train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size, device=device)
        if dev_UAS > best_dev_UAS:
            best_dev_UAS = dev_UAS
            print("New best dev UAS! Saving model.")
            torch.save(parser.model.state_dict(), output_path)
        print("")


def train_for_epoch(parser, train_data, dev_data, optimizer, loss_func, batch_size, device="cpu"):
    """ Train the neural dependency parser for single epoch.

    Note: In PyTorch we can signify train versus test and automatically have
    the Dropout Layer applied and removed, accordingly, by specifying
    whether we are training, `model.train()`, or evaluating, `model.eval()`

    @param parser (Parser): Neural Dependency Parser
    @param train_data ():
    @param dev_data ():
    @param optimizer (nn.Optimizer): Adam Optimizer
    @param loss_func (nn.CrossEntropyLoss): Cross Entropy Loss Function
    @param batch_size (int): batch size
    @param lr (float): learning rate

    @return dev_UAS (float): Unlabeled Attachment Score (UAS) for dev data
    """
    parser.model.train() # Places model in "train" mode, i.e. apply dropout layer
    n_minibatches = math.ceil(len(train_data) / batch_size)
    loss_meter = AverageMeter()

    with tqdm(total=(n_minibatches)) as prog:
        for i, (train_x, train_y) in enumerate(minibatches(train_data, batch_size)):
            optimizer.zero_grad()   # remove any baggage in the optimizer
            loss = 0. # store loss for this batch here
            train_x = torch.from_numpy(train_x).long().to(device)
            train_y = torch.from_numpy(train_y.nonzero()[1]).long().to(device)

            ### TODO:
            ###      1) Run train_x forward through model to produce `logits`
            ###      2) Use the `loss_func` parameter to apply the PyTorch CrossEntropyLoss function.
            ###         This will take `logits` and `train_y` as inputs. It will output the CrossEntropyLoss
            ###         between softmax(`logits`) and `train_y`. Remember that softmax(`logits`)
            ###         are the predictions (y^ from the PDF).
            ###      3) Backprop losses
            ###      4) Take step with the optimizer
            ### Please see the following docs for support:
            ###     Optimizer Step: https://pytorch.org/docs/stable/optim.html#optimizer-step
            ### YOUR CODE HERE (~5-10 lines)
            predictions = parser.model(train_x)
            loss = loss_func(predictions, train_y)
            loss.backward()
            optimizer.step()
            ### END YOUR CODE
            prog.update(1)
            loss_meter.update(loss.item())

    print ("Average Train Loss: {}".format(loss_meter.avg))

    print("Evaluating on dev set",)
    parser.model.eval() # Places model in "eval" mode, i.e. don't apply dropout layer
    dev_UAS, _ = parser.parse(dev_data, device=device)
    print("- dev UAS: {:.2f}".format(dev_UAS * 100.0))
    return dev_UAS

def start(args):
    debug = args.debug
    device = 0


    print(80 * "=")
    print("INITIALIZING")
    print(80 * "=")
    parser, embeddings, train_data, dev_data, test_data = load_and_preprocess_data(debug)

    start = time.time()
    model = ParserModel(embeddings).to(device)
    parser.model = model
    print("took {:.2f} seconds\n".format(time.time() - start))

    print(80 * "=")
    print("TRAINING")
    print(80 * "=")
    output_dir = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
    output_path = output_dir + "model.weights"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train(parser, train_data, dev_data, output_path, batch_size=1024, n_epochs=10, lr=0.0005, device=device)

    if not debug:
        print(80 * "=")
        print("TESTING")
        print(80 * "=")
        print("Restoring the best model weights found on the dev set")
        parser.model.load_state_dict(torch.load(output_path, map_location=device if device=="cpu" else f"cuda:{device}"))
        print("Final evaluation on test set",)
        parser.model.eval()
        UAS, dependencies = parser.parse(test_data, device=device)
        print("- test UAS: {:.2f}".format(UAS * 100.0))
        print("Done!")

def main(arguments_str):  
    args = sys.argv[1:]
    if arguments_str:
        args = arguments_str.split()

    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action="store_true", help="Debug")
    parser.add_argument('--device', type=str, default="cpu", help="Device to use")
    start(parser.parse_args(args))

if __name__ == "__main__":
    main(None)



"""
C:\ProgramData\Anaconda3\python.exe C:/Users/soki/PycharmProjects/NLP_HW4/run.py --device cuda
================================================================================
INITIALIZING
================================================================================
Loading data...
took 2.26 seconds
Building parser...
took 1.41 seconds
Loading pretrained embeddings...
took 3.57 seconds
Vectorizing data...
took 1.79 seconds
Preprocessing training data...
took 69.51 seconds
took 2.55 seconds

================================================================================
TRAINING
================================================================================
Epoch 1 out of 10
100%|██████████| 1848/1848 [00:33<00:00, 55.14it/s]
Average Train Loss: 0.17826719019255596
Evaluating on dev set
1445850it [00:00, 30122562.06it/s]      
- dev UAS: 84.54
New best dev UAS! Saving model.
  0%|          | 0/1848 [00:00<?, ?it/s]
Epoch 2 out of 10
100%|██████████| 1848/1848 [00:29<00:00, 62.05it/s]
Average Train Loss: 0.11351320215650064
Evaluating on dev set
1445850it [00:00, 24929435.33it/s]      
- dev UAS: 86.45
New best dev UAS! Saving model.

Epoch 3 out of 10
100%|██████████| 1848/1848 [00:30<00:00, 60.36it/s]
Average Train Loss: 0.09998108191189892
Evaluating on dev set
1445850it [00:00, 25367203.65it/s]      
- dev UAS: 87.28
New best dev UAS! Saving model.
  0%|          | 0/1848 [00:00<?, ?it/s]
Epoch 4 out of 10
100%|██████████| 1848/1848 [00:30<00:00, 61.29it/s]
Average Train Loss: 0.09139955503809633
Evaluating on dev set
1445850it [00:00, 26519794.46it/s]      
- dev UAS: 87.69
New best dev UAS! Saving model.

Epoch 5 out of 10
100%|██████████| 1848/1848 [00:36<00:00, 49.95it/s]
Average Train Loss: 0.085175301093463
Evaluating on dev set
1445850it [00:00, 21906823.25it/s]      
- dev UAS: 87.67

Epoch 6 out of 10
100%|██████████| 1848/1848 [00:36<00:00, 50.07it/s]
Average Train Loss: 0.08035307952497171
Evaluating on dev set
1445850it [00:00, 20364603.14it/s]      
- dev UAS: 88.09
New best dev UAS! Saving model.
  0%|          | 0/1848 [00:00<?, ?it/s]
Epoch 7 out of 10
100%|██████████| 1848/1848 [00:32<00:00, 57.50it/s]
Average Train Loss: 0.07632694575629193
Evaluating on dev set
1445850it [00:00, 27805165.67it/s]      
- dev UAS: 88.20
New best dev UAS! Saving model.
  0%|          | 0/1848 [00:00<?, ?it/s]
Epoch 8 out of 10
100%|██████████| 1848/1848 [00:29<00:00, 62.06it/s]
Average Train Loss: 0.07281426161334112
Evaluating on dev set
1445850it [00:00, 20081575.04it/s]      
- dev UAS: 88.37
New best dev UAS! Saving model.
  0%|          | 0/1848 [00:00<?, ?it/s]
Epoch 9 out of 10
100%|██████████| 1848/1848 [00:28<00:00, 64.78it/s]
Average Train Loss: 0.06947017599147184
Evaluating on dev set
1445850it [00:00, 22240400.33it/s]      
  0%|          | 0/1848 [00:00<?, ?it/s]- dev UAS: 88.36

Epoch 10 out of 10
100%|██████████| 1848/1848 [00:28<00:00, 64.47it/s]
Average Train Loss: 0.06689222400112972
Evaluating on dev set
1445850it [00:00, 15715474.48it/s]      
- dev UAS: 88.27

================================================================================
TESTING
================================================================================
Restoring the best model weights found on the dev set
Final evaluation on test set
2919736it [00:00, 39456209.19it/s]      
- test UAS: 89.00
Done!

Process finished with exit code 0
"""