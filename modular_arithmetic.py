import os
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import time

from models import Transformer
from neels_visualizations import lines


def main():
    root = r"C:\Users\Avery\Projects\ModularitySERI\saved_models\grokking"
    lr = 1e-3
    weight_decay = 1.0
    p = 113
    d_model = 128
    fn_name = 'add'  # ['add', 'subtract', 'x2xyy2','rand']
    frac_train = 0.3
    num_epochs = 500
    save_models = True
    save_every = 100
    stopping_thresh = -1
    seed = 0

    num_layers = 1
    batch_style = 'full'
    d_vocab = p + 1
    n_ctx = 3
    d_mlp = 4 * d_model
    num_heads = 4
    assert d_model % num_heads == 0
    d_head = d_model // num_heads
    act_type = 'ReLU'  # ['ReLU', 'GeLU']
    # batch_size = 512
    use_ln = False
    random_answers = np.random.randint(low=0, high=p, size=(p, p))
    fns_dict = {'add': lambda x, y: (x + y) % p, 'subtract': lambda x, y: (x - y) % p,
                'x2xyy2': lambda x, y: (x ** 2 + x * y + y ** 2) % p, 'rand': lambda x, y: random_answers[x][y]}
    fn = fns_dict[fn_name]

    train_model = True
    train, test = gen_train_test(frac_train, p, seed)
    print(len(train), len(test))

    # Creates an array of Boolean indices according to whether each data point is in
    # train or test
    # Used to index into the big batch of all possible data
    is_train = []
    is_test = []
    for x in range(p):
        for y in range(p):
            if (x, y, 113) in train:
                is_train.append(True)
                is_test.append(False)
            else:
                is_train.append(False)
                is_test.append(True)

    is_train = np.array(is_train)  # TODO Used in loss
    is_test = np.array(is_test)

    model = Transformer(num_layers=num_layers, d_vocab=d_vocab, d_model=d_model, d_mlp=d_mlp, d_head=d_head,
                        num_heads=num_heads, n_ctx=n_ctx, act_type=act_type, use_cache=False, use_ln=use_ln)

    if train_model:
        model.to('cuda')
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98))
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step / 10, 1))
        run_name = f"grok_{int(time.time())}"
        print(f'Run name {run_name}')
        if save_models:
            save_path = f"{root}/{run_name}"
            os.makedirs(save_path, exist_ok=True)
            save_dict = {'model': model.state_dict(), 'train_data': train, 'test_data': test}
            torch.save(save_dict, f"{save_path}/init.pth")
        train_losses = []
        test_losses = []
        for epoch in range(num_epochs):
            train_loss = full_loss(model, train, fn)
            test_loss = full_loss(model, test, fn)
            train_losses.append(train_loss.item())
            test_losses.append(test_loss.item())
            if epoch % 100 == 0:
                # print(f"{epoch}_{np.log(train_loss.item()):.4f}_{np.log(test_loss.item()):.4f}")  # _{train_acc.item():.4f}_{test_acc.item():.4f}")
                print(f"Epoch {epoch} Train Loss: {round(np.log(train_loss.item()),5)} Test Loss: {round(np.log(test_loss.item()), 5)}")
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if test_loss.item() < stopping_thresh:
                break
            if save_models and (epoch % save_every == 0):
                if test_loss.item() < stopping_thresh:
                    break
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss': train_loss,
                    'test_loss': test_loss,
                    'epoch': epoch,
                }
                torch.save(save_dict, f"{save_path}/{epoch}.pth")
                print(f"Saved model to {save_path}/{epoch}.pth")
        # if not save_models:
        #     os.mkdir(root / run_name)
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'train_loss': train_loss,
            'test_loss': test_loss,
            'train_losses': train_losses,
            'test_losses': test_losses,
            'epoch': epoch,
        }
        torch.save(save_dict, f"{save_path}/final.pth")
        print(f"Saved model to {save_path}/final.pth")
        lines([train_losses, test_losses], labels=['train', 'test'], log_y=True)


def gen_train_test(frac_train, num, seed=0):
    pairs = [(i, j, num) for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]


# Helper functions
def cuda_memory():
    print(torch.cuda.memory_allocated()/1e9)


def cross_entropy_high_precision(logits, labels):
    # Shapes: batch x vocab, batch
    # Cast logits to float64 because log_softmax has a float32 underflow on overly
    # confident data and can only return multiples of 1.2e-7 (the smallest float x
    # such that 1+x is different from 1 in float32). This leads to loss spikes
    # and dodgy gradients
    logprobs = F.log_softmax(logits.to(torch.float64), dim=-1)
    prediction_logprobs = torch.gather(logprobs, index=labels[:, None], dim=-1)
    loss = -torch.mean(prediction_logprobs)
    return loss


def full_loss(model, data, fn):
    # Take the final position only
    logits = model(data)[:, -1]
    labels = torch.tensor([fn(i, j) for i, j, _ in data]).to('cuda')
    return cross_entropy_high_precision(logits, labels)


# def test_logits(logits, bias_correction=False, original_logits=None, mode='all'):
#     # Calculates cross entropy loss of logits representing a batch of all p^2
#     # possible inputs
#     # Batch dimension is assumed to be first
#     if logits.shape[1]==p*p:
#         logits = logits.T
#     if logits.shape==torch.Size([p*p, p+1]):
#         logits = logits[:, :-1]
#     logits = logits.reshape(p*p, p)
#     if bias_correction:
#         # Applies bias correction - we correct for any missing bias terms,
#         # independent of the input, by centering the new logits along the batch
#         # dimension, and then adding the average original logits across all inputs
#         logits = einops.reduce(original_logits - logits, 'batch ... -> ...', 'mean') + logits
#     if mode=='train':
#         return cross_entropy_high_precision(logits[is_train], labels[is_train])
#     elif mode=='test':
#         return cross_entropy_high_precision(logits[is_test], labels[is_test])
#     elif mode=='all':
#         return cross_entropy_high_precision(logits, labels)


if __name__ == "__main__":
    main()










