import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import pickle as pkl

class MLP(nn.Module):
    def __init__(
        self, input_dim: int, output_dim: int, layers_size:List[int] = [256]*2,
        activation: nn.Module = nn.ReLU, output_activation: Optional[nn.Module] = None
    ):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, layers_size[0]), activation()]
        for i in range(len(layers_size) - 1):
            layers += [
                nn.Linear(layers_size[i], layers_size[i+1]),
                activation()
            ]
        layers.append(nn.Linear(layers_size[-1], output_dim))
        
        if output_activation is not None:
            layers.append(output_activation())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)
    

class MapClassifyDataset(Dataset):
    def __init__(self, labels, vectors):
        self.labels = labels
        self.vectors = vectors
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, index):
        label = self.labels[index]
        vector = self.vectors[index]
        return label, vector
    
def prepare_dataset(val_percentage = 0.2) -> Tuple[MapClassifyDataset]:
    labels, vectors = [], []
    for i in range(9):
        with open("./dataset/"+str(i)+".pkl", "rb") as f:
            (y,x) = pkl.load(f)
        labels.append(y)
        vectors.append(x)
    labels = np.stack(labels).reshape(-1, 1)
    vectors = np.stack(vectors).reshape(-1, 235)
    datasets = np.concatenate([labels, vectors], -1)
    np.random.shuffle(datasets)
    size = datasets.shape[0]
    train_size = int(size * (1-val_percentage))
    labels = datasets[:, 0]
    vectors = datasets[:, 1:]
    train_dataset = MapClassifyDataset(labels[:train_size], vectors[:train_size])
    test_dataset = MapClassifyDataset(labels[train_size:], vectors[train_size:])
    return train_dataset, test_dataset

class TaskSpecObsEncoder(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.n_layer = nn.Sequential(
            nn.Linear(5,  5),
            nn.LayerNorm(5),
            nn.Tanh(),
        )
        self.l_layer = nn.Sequential(
            nn.Linear(70,  25),
            nn.LayerNorm(25),
            nn.Tanh(),
        )
        self.mix_layer = nn.Linear(100, h_dim)
    def forward(self, o):
        batch_size = o.shape[0]
        n_info = o[:, :25].reshape((batch_size, 5, 5))
        l_info = o[:, 25:].reshape((batch_size, 3, 70))
        n_enc = self.n_layer(n_info).reshape((batch_size, -1))
        l_enc = self.l_layer(l_info).reshape((batch_size, -1))
        return F.relu(self.mix_layer(torch.cat([n_enc, l_enc], -1)))
    

class TaskClassifer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TaskSpecObsEncoder(128)
        self.net = MLP(128, 9, [128]*3)
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        
    def forward(self, obs):
        return self.net(self.encoder(obs))

def train_classifier() -> TaskClassifer:
    device = "cuda"
    batch_size = 64
    learning_rate = 3E-4
    epochs = 100
        
    model = TaskClassifer().to(device)
    train_dataset, test_dataset = prepare_dataset()
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    opt = torch.optim.Adam(model.parameters(), lr = learning_rate)
    
    for e in range(epochs):
        avg_loss = 0.0
        avg_acc = 0.0
        for (label, vector) in train_dataloader:
            label = label.to(device).long()
            logits = model(vector.float().to(device))
            loss = F.cross_entropy(logits, label)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            with torch.no_grad():
                avg_acc += (logits.argmax(-1) == label).float().mean()
                avg_loss += loss.detach().item()
        avg_loss /= len(train_dataloader)
        avg_acc /= len(train_dataloader)
        print(f'[Epoch:{e+1}] loss:{loss.detach().item():.2f} acc:{avg_acc.item():.2f}')

        with torch.no_grad():
            acc = 0.0
            counter = 0
            for (label, vector) in test_dataloader:
                counter += 1
                label = label.to(device).long()
                logits = model(vector.float().to(device))
                acc += (logits.argmax(-1) == label).float().mean()
                if counter >= 20:
                    break
            acc /= 20
            print(f'[Epoch:{e+1}] test_acc:{acc:.2f}')
        # wandb.log({"test_acc": acc})
    return model