# ==================== Imports ====================
from __future__ import annotations
import logging
import os
import random as rd
from typing import Any, Optional, Sequence, Mapping, Iterable, Union, overload, Literal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
TorchDevice = Union[str, torch.device]


# ==================== Logging ====================


# Setting up the logger
logger = logging.getLogger("mar_fl")


# ==================== Models ====================


# Simple CNN for MNIST digit classification
class SimpleCNN(nn.Module):
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)    # [batch, 32, 26, 26]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)   # [batch, 64, 24, 24]
        self.pool = nn.MaxPool2d(2)                     # [batch, 64, 12, 12]
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))     
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # CrossEntropyLoss expects raw logits


# Smaller Fine-tuned ModernBERT for 20Newsgroups category classification
class ModernBERTClassifier(nn.Module):
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=20, dropout_rate=0.2):
        super(ModernBERTClassifier, self).__init__()
        
        # Load the pre-trained BERT model and freeze all its parameters
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Define a tuned classification head with about 57k parameters
        intermediate_size = 72 # NOTE this simpler version has a lower intermediate size
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):

        # Get the embeddings from the frozen BERT
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take the CLS token from the embeddings and pass it through our trainable classifier
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_embedding)
        return logits


# Larger Fine-tuned ModernBERT for 20Newsgroups category classification
class AdvancedModernBERTClassifier(nn.Module):
    
    def __init__(self, model_name='distilbert-base-uncased', num_labels=20, dropout_rate=0.2):
        super(AdvancedModernBERTClassifier, self).__init__()
        
        # Load the pre-trained BERT model and freeze all its parameters
        self.bert = AutoModel.from_pretrained(model_name)
        for param in self.bert.parameters():
            param.requires_grad = False
            
        # Define a tuned classification head with about 202k parameters
        intermediate_size = 256 # NOTE this advanced version has a higher intermediate size
        self.head = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.bert.config.hidden_size, intermediate_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_size, num_labels)
        )

    def forward(self, input_ids, attention_mask):

        # Get the embeddings from the frozen BERT
        with torch.no_grad():
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Take the CLS token from the embeddings and pass it through our trainable classifier
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.head(cls_embedding)
        return logits


# ==================== Data Loaders ====================


# Helper function for creating non-IID MNIST splits
def _dirichlet_client_indices(
    labels: np.ndarray,
    num_partitions: int,
    alpha: float,
    rng: np.random.Generator,
) -> list[list[int]]:
    import numpy as np
    num_classes = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_partitions)]
    for c in range(num_classes):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * num_partitions)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)  # floor like the original
        splits = np.split(idx_c, cuts[:-1])
        for k, part in enumerate(splits):
            client_indices[k].extend(part.tolist())
    return client_indices

# Create non-IID MNIST loader
def get_mnist_data_loaders_dirichlet(
    partition: int,
    num_partitions: int,
    seed: int = 42,
    batch_size: int = 64,
    alpha: float = 1.0,
) -> tuple[DataLoader, None, DataLoader]:
    import numpy as np
    from torch.utils.data import Subset
    from torchvision import datasets, transforms
    import torch

    # NOTE align global batch size to 125 peers case:
    # (a) MNIST & 16 peers: 500 (divided by 5 yields 100 => choose batch size 100 and x5 number of mini batches per FL iteration in shell script)
    # (b) MNIST & 64 peers: 62 for peers 0-31 and 63 for peers 32-63 (choose respective batch sizes and x2 number of mini batches per FL iteration in shell script)
    if num_partitions == 16:
        batch_size = 100
    elif num_partitions == 64:
        if partition < 32:
            batch_size = 62
        else:
            batch_size = 63

    rng = np.random.default_rng(seed)
    tfm = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.1307,), (0.3081,))])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, transform=tfm)

    train_clients = _dirichlet_client_indices(np.array(train_ds.targets), num_partitions, alpha, rng)
    test_clients  = _dirichlet_client_indices(np.array(test_ds.targets),  num_partitions, alpha, rng)

    g = torch.Generator().manual_seed(seed + partition)
    train_loader = DataLoader(Subset(train_ds, train_clients[partition]), batch_size=batch_size, shuffle=True,  generator=g)
    test_loader  = DataLoader(Subset(test_ds,  test_clients[partition]),  batch_size=batch_size, shuffle=False)
    return train_loader, None, test_loader


# Create non-IID NEWS loader
def get_text_data_loaders_dirichlet(
    partition: int,
    num_partitions: int,
    seed: int = 42,
    batch_size: int = 16,
    alpha: float = 1.0,
    dataset_name: str = "SetFit/20_newsgroups",
) -> tuple[Any, None, Any]:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # NOTE align global batch size to 125 peers case:
    # (c) NEWS & 16 peers: 125 (divided by 5 yields 25 => choose batch size 25 and x5 number of mini batches per FL iteration in shell script)
    # (d) NEWS & 64 peers: 15 for peers 0-23 and 16 for peers 24-63 (choose respective batch sizes and x2 number of mini batches per FL iteration in shell script)
    if num_partitions == 16:
        batch_size = 25
    elif num_partitions == 64:
        if partition < 24:
            batch_size = 15
        else:
            batch_size = 16

    rng = np.random.default_rng(seed)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # Load raw splits and compute client indices via Dirichlet per class
    raw_train = load_dataset(dataset_name, split="train")
    raw_test  = load_dataset(dataset_name, split="test")

    train_labels = np.array(raw_train["label"])
    test_labels  = np.array(raw_test["label"])

    train_clients = _dirichlet_client_indices(train_labels, num_partitions, alpha, rng)
    test_clients  = _dirichlet_client_indices(test_labels,  num_partitions, alpha, rng)  # same logic as MNIST

    # Tokenize (same tokenizer/settings as the IID loader)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

    tok_train = raw_train.map(tokenize_function, batched=True)
    tok_test  = raw_test.map(tokenize_function, batched=True)
    tok_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tok_test.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Build per-client subsets/loaders
    g = torch.Generator().manual_seed(seed + partition)
    train_subset = Subset(tok_train, train_clients[partition])
    test_subset  = Subset(tok_test,  test_clients[partition])

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader  = DataLoader(test_subset,  batch_size=batch_size, shuffle=False)

    # Wrap to yield (data_dict, label) like your current text loader
    class TextDataLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                data = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
                target = batch["label"]
                yield data, target
        def __len__(self):
            return len(self.loader)

    return TextDataLoaderWrapper(train_loader), None, TextDataLoaderWrapper(test_loader)


# MNIST data loaders designed for federated learning and clients with strict dataset seperation
def get_mnist_data_loaders(
    partition: int,
    num_partitions: int,
    seed: int = 42,
    batch_size: int = 64,
) -> tuple[DataLoader, None, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # NOTE align global batch size to 125 peers case:
    # (a) MNIST & 16 peers: 500 (divided by 5 yields 100 => choose batch size 100 and x5 number of mini batches per FL iteration in shell script)
    # (b) MNIST & 64 peers: 62 for peers 0-31 and 63 for peers 32-63 (choose respective batch sizes and x2 number of mini batches per FL iteration in shell script)
    if num_partitions == 16:
        batch_size = 100
    elif num_partitions == 64:
        if partition < 32:
            batch_size = 62
        else:
            batch_size = 63

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform) # NOTE global and shared test data aligning to FL research

    # Deterministic train/val split
    total_size = len(dataset)
    partition_size = total_size // num_partitions
    start_idx = partition * partition_size
    end_idx = (partition + 1) * partition_size if partition != num_partitions - 1 else total_size
    partition_indices = list(range(start_idx, end_idx))
    train_dataset = Subset(dataset, partition_indices)

    # Create data loaders
    g = torch.Generator()
    g.manual_seed(seed + partition) # NOTE partition-specific seeding to align with FL principles
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, None, test_loader


# Newsgroup data loaders designed for federated learning and clients with strict dataset seperation (smaller default batch size to avoid OOMs due to ModernBERTs huge size)
def get_text_data_loaders(
    partition: int,
    num_partitions: int,
    seed: int = 42,
    batch_size: int = 16,
    dataset_name: str = "SetFit/20_newsgroups",
) -> tuple[Any, None, Any]:

    # NOTE align global batch size to 125 peers case:
    # (c) NEWS & 16 peers: 125 (divided by 5 yields 25 => choose batch size 25 and x5 number of mini batches per FL iteration in shell script)
    # (d) NEWS & 64 peers: 15 for peers 0-23 and 16 for peers 24-63 (choose respective batch sizes and x2 number of mini batches per FL iteration in shell script)
    if num_partitions == 16:
        batch_size = 25
    elif num_partitions == 64:
        if partition < 24:
            batch_size = 15
        else:
            batch_size = 16

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    dataset = load_dataset(dataset_name, split='train') 
    test_dataset = load_dataset(dataset_name, split='test') # NOTE global and shared test data aligning to FL research

    # Define a tokenization function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128) # NOTE the tokenizer handles padding and truncation to a max length

    # Apply the tokenization to the datasets & set the format to PyTorch tensores & select relevant columns
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create a smaller subset of the test dataset (10%) to speed up evaluation.
    subset_ratio = 0.1
    test_subset_size = int(len(tokenized_test_dataset) * subset_ratio)
    test_subset_indices = list(range(test_subset_size))
    test_subset = Subset(tokenized_test_dataset, test_subset_indices)

    # Partition the training data
    total_size = len(tokenized_dataset)
    partition_size = total_size // num_partitions
    start_idx = partition * partition_size
    end_idx = (partition + 1) * partition_size if partition != num_partitions - 1 else total_size
    partition_indices = list(range(start_idx, end_idx))
    train_partition = Subset(tokenized_dataset, partition_indices)

    # Create DataLoaders
    g = torch.Generator()
    g.manual_seed(seed + partition) # NOTE partition-specific seeding to align with FL principles
    train_loader = DataLoader(train_partition, batch_size=batch_size, shuffle=True, generator=g)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    # The training function expects (data, target) so we need a wrapper
    class TextDataLoaderWrapper:
        def __init__(self, loader):
            self.loader = loader
        def __iter__(self):
            for batch in self.loader:
                data = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']} # NOTE group input_ids and attention_mask into a single 'data' dict
                target = batch['label']
                yield data, target
        def __len__(self):
            return len(self.loader)
    return TextDataLoaderWrapper(train_loader), None, TextDataLoaderWrapper(test_loader)


# ==================== Training & Evaluation ====================


# Pass through num_mini_batches starting at next_batch_idx & thereby apply manually implemented sgd updates with momentum to enable exchanging optimizer states between peers
@overload
def train_num_mini_batches_manually(
    model: Union[nn.Module, list[nn.Module]],
    device: TorchDevice,
    train_loader: Iterable[tuple[Any, Tensor]],
    learning_rate: float,
    momentum: float,
    momentum_vector: Sequence[Tensor],
    peer_id: str = "X",
    num_mini_batches: int = 1,
    next_batch_idx: int = 0,
    do_dp: bool = False,
    return_logits: Literal[False] = False,
) -> tuple[float, str, int, list[Tensor], Optional[list[Tensor]]]:
    ...

# Pass through num_mini_batches starting at next_batch_idx & thereby apply manually implemented sgd updates with momentum to enable exchanging optimizer states between peers
@overload
def train_num_mini_batches_manually(
    model: Union[nn.Module, list[nn.Module]],
    device: TorchDevice,
    train_loader: Iterable[tuple[Any, Tensor]],
    learning_rate: float,
    momentum: float,
    momentum_vector: Sequence[Tensor],
    peer_id: str = "X",
    num_mini_batches: int = 1,
    next_batch_idx: int = 0,
    do_dp: bool = False,
    return_logits: Literal[True] = True,
) -> tuple[float, str, int, list[Tensor], Optional[list[Tensor]], list[Tensor]]:
    ...

# Pass through num_mini_batches starting at next_batch_idx & thereby apply manually implemented sgd updates with momentum to enable exchanging optimizer states between peers
def train_num_mini_batches_manually(
    model: Union[nn.Module, list[nn.Module]],
    device: TorchDevice,
    train_loader: Iterable[tuple[Any, Tensor]],
    learning_rate: float,
    momentum: float,
    momentum_vector: Sequence[Tensor],
    peer_id: str = "X",
    num_mini_batches: int = 1,
    next_batch_idx: int = 0,
    do_dp: bool = False,
    return_logits: bool = False,
) -> Any:
    if isinstance(model, list) and len(model) == 1 and isinstance(model[0], torch.nn.Module): # NOTE Auto-correct if model is a list of one item
        model = model[0]
    assert isinstance(model, torch.nn.Module), f"Model type error: {type(model)}" # NOTE Sanity check
    model.train()
    momentum_vector = [m.to(device) for m in momentum_vector]
    total_loss = 0
    start_idx = next_batch_idx
    end_idx = start_idx + num_mini_batches # NOTE end index mini-batch is not passed through the model => it therefore besides defines the start index of the next iteration
    initial_state = {k: v.clone().detach() for k, v in model.state_dict().items()} # NOTE save initial model state for DP's delta calculation

    all_logits = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < start_idx:
            continue
        if batch_idx >= end_idx:
            break

        model.zero_grad()
        target = target.to(device)
        if isinstance(data, dict):
            input_ids = data['input_ids'].to(device) # NOTE data is a dictionary from the BERT loader
            attention_mask = data['attention_mask'].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            data = data.to(device) # NOTE data is from simple CNN
            output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()

        with torch.no_grad(): # NOTE manually implemented optimizer step
            for param, m_vec in zip(model.parameters(), momentum_vector):
                if param.grad is None:
                    continue
                m_vec.mul_(momentum).add_(param.grad, alpha=(1 - momentum)) # NOTE Reddi et al. 2020 momentum update: m_vec = momentum * m_vec + (1 - momentum) * gradient
                param.data.add_(m_vec, alpha=-learning_rate) # NOTE SGD gradient descent step: param = param - learning_rate * m_vec
        
        if return_logits:
            all_logits.append(output.detach().cpu())
        total_loss += loss.item()

    avg_loss = total_loss / num_mini_batches
    training_message = f"Peer {peer_id} for mini-batches {next_batch_idx}-{(end_idx-1)} - Training loss: {avg_loss:.4f}."

    delta_vector = None # NOTE compute the delta vector between the updated model and the previous model
    if do_dp:
        with torch.no_grad():
            delta_vector = [model.state_dict()[k] - initial_state[k].to(device) for k in model.state_dict()]

    if end_idx >= len(train_loader):
        end_idx = 0
    if return_logits:
        return avg_loss, training_message, end_idx, momentum_vector, delta_vector, all_logits
    else:
        return avg_loss, training_message, end_idx, momentum_vector, delta_vector


# Pass through num_mini_batches starting at next_batch_idx
def train_num_mini_batches(
    model: Union[nn.Module, list[nn.Module]],
    device: TorchDevice,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: optim.Optimizer,
    peer_id: str = "X",
    num_mini_batches: int = 1,
    next_batch_idx: int = 0,
) -> tuple[float, str, int]:
    if isinstance(model, list) and len(model) == 1 and isinstance(model[0], torch.nn.Module): # NOTE Auto-correct if model is a list of one item
        model = model[0]
    assert isinstance(model, torch.nn.Module), f"Model type error: {type(model)}" # NOTE Sanity check
    model.train()
    total_loss = 0
    start_idx = next_batch_idx
    end_idx = start_idx + num_mini_batches # NOTE end index mini-batch is not passed through the model => it therefore besides defines the start index of the next iteration

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx < start_idx:
            continue
        if batch_idx >= end_idx:
            break

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / num_mini_batches
    training_message = f"Peer {peer_id} for mini-batches {next_batch_idx}-{(end_idx-1)} - Training loss: {avg_loss:.4f}."
    if end_idx >= len(train_loader):
        end_idx = 0
    return avg_loss, training_message, end_idx


# Pass through entire batch
def train_entire_batch(
    model: nn.Module,
    device: TorchDevice,
    train_loader: Iterable[tuple[Tensor, Tensor]],
    optimizer: optim.Optimizer,
    epoch: int,
    peer_id: str = "X",
) -> tuple[float, str]:
    model.train()
    total_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    training_message = f"Peer {peer_id} at epoch {epoch} - Training loss: {avg_loss:.4f}."
    return avg_loss, training_message


# Validation
def evaluate(
    model: nn.Module,
    device: TorchDevice,
    loader: Iterable[tuple[Union[Tensor, Mapping[str, Tensor]], Tensor]],
    label: str = "Validation",
    peer_id: str = "X",
) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0

    with torch.no_grad():
        i = 0
        for data, target in loader:
            target = target.to(device)
            if isinstance(data, dict):
                input_ids = data['input_ids'].to(device)
                attention_mask = data['attention_mask'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                data = data.to(device)
                output = model(data)
            loss = F.cross_entropy(output, target)
            loss_sum += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
            i = i+1

    acc = 100.0 * correct / total
    avg_loss = loss_sum / len(loader)
    return acc, avg_loss


# ==================== Knowledge Distillation ====================


# Knowledge distillation using the top-k most similar models by training the peer's student model with a hybrid loss of 1) KL-divergence loss between the peer's and the teacher's output distributions and 2) cross-entropy loss using ground truth data
def knowledge_distillation(
    peer_id: Union[int, str],
    model: nn.Module,
    device: TorchDevice,
    train_loader: Iterable[tuple[Union[Tensor, Mapping[str, Tensor]], Tensor]],
    models_collected: Sequence[Mapping[str, Tensor]],
    momentum_vector: Sequence[Tensor],
    learning_rate: float,
    momentum: float,
    iteration: int,
    kn_dist_iters: int,
    kn_dist_no_blending: bool,
    include_ce_loss: bool,
    top_k_ratio: float = 0.4,
    temperature: float = 3.0,
    epochs: int = 1,
) -> tuple[nn.Module, list[Tensor], float]:
    model.eval()
    student_logits_list = []
    ground_truth_labels = []

    try: # NOTE sanity checking whether model is on device
        model_device = next(model.parameters()).device
    except Exception as e:
        logger.warning(f"[{peer_id}] ERROR accessing model device: {e}")

    # 0v4: create data loader
    distill_loader = []
    seen_indices = set()
    for i, (data_batch, y) in enumerate(train_loader):
        if len(distill_loader) >= 8:
            break        
        if isinstance(data_batch, dict):
            batch_indices = tuple(data_batch['input_ids'].view(data_batch['input_ids'].shape[0], -1).sum(dim=1).tolist()) # NOTE for BERT use input_ids to create a unique identifier for the batch
        else:
            batch_indices = tuple(data_batch.view(data_batch.shape[0], -1).sum(dim=1).tolist()) # NOTE original logic for MNIST
        if batch_indices in seen_indices:
            continue
        seen_indices.add(batch_indices)
        if isinstance(data_batch, dict):
            distill_loader.append(({'input_ids': data_batch['input_ids'].to(device), 'attention_mask': data_batch['attention_mask'].to(device)}, y.to(device)))
        else:
            distill_loader.append((data_batch.to(device), y.to(device)))
    if len(distill_loader) < 1:
        logger.warning(f"[{peer_id}] WARNING: distill_loader is empty, skipping distillation.")
        return model, momentum_vector, 0

    # 1v4: get student logits
    with torch.no_grad():
        for i, (data, labels) in enumerate(distill_loader):
            if isinstance(data, dict):
                student_logits = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
            else:
                student_logits = model(data)
            student_logits_list.append(student_logits.cpu())
            ground_truth_labels.append(labels.cpu()) # NOTE move labels to CPU to match logits

    # 2v4: compare student logits to candidate teacher models
    kl_scores = []
    teacher_logits_list_by_model = []
    if isinstance(model, ModernBERTClassifier):
        temp_model = ModernBERTClassifier().to(device)
    else:
        temp_model = SimpleCNN().to(device)
    for peer_state in models_collected:
        if isinstance(model, ModernBERTClassifier):
            temp_model.load_state_dict(peer_state, strict=False)
        else:
            temp_model.load_state_dict(peer_state, strict=True)
        temp_model.eval()
        kl_total = 0.0
        logits_batches = []
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(distill_loader):
                if isinstance(data, dict):
                    t_logits = temp_model(input_ids=data['input_ids'], attention_mask=data['attention_mask']).cpu()
                else:
                    t_logits = temp_model(data).cpu()
                logits_batches.append(t_logits)
                s_logits = student_logits_list[batch_idx]
                kl = F.kl_div( # NOTE compute KL divergence between teacher's softened logits and student's corresponding logits
                    F.log_softmax(s_logits / temperature, dim=1),
                    F.softmax(t_logits / temperature, dim=1),
                    reduction='batchmean'
                )
                kl_total += kl.item()
        kl_scores.append(kl_total) # NOTE collect KL scores to select most informative teacher models
        teacher_logits_list_by_model.append(logits_batches)

    # 3v4: select top-k teacher models
    num_top_k = max(1, int(top_k_ratio * len(models_collected)))
    top_k_indices = sorted(range(len(kl_scores)), key=lambda i: kl_scores[i])[:num_top_k]

    # 4v4: distill knowledge from top-k teachers (=> train our student model using a hybrid loss of 1) KL divergence loss between student model and teacher model output distributions and 2) cross-entropy loss between student predictions and the true labels)
    model.train()
    momentum_vector = [m.to(device) for m in momentum_vector]
    kl_factor = 0
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(distill_loader):
            target = target.to(device)
            if isinstance(data, dict):
                student_logits = model(input_ids=data['input_ids'], attention_mask=data['attention_mask'])
            else:
                student_logits = model(data)
            avg_teacher_logits = torch.mean(torch.stack([teacher_logits_list_by_model[i][batch_idx] for i in top_k_indices]), dim=0).to(device)
            kl_loss = F.kl_div(F.log_softmax(student_logits / temperature, dim=1), F.softmax(avg_teacher_logits / temperature, dim=1), reduction='batchmean') * (temperature ** 2)
            if include_ce_loss:
                ce_loss = F.cross_entropy(student_logits, target)
                if kn_dist_no_blending:
                    loss = kl_loss + ce_loss
                    kl_factor = 1
                else:
                    kl_factor = max(0.0, 1.0 - (iteration / kn_dist_iters))
                    loss = (kl_factor * kl_loss) + ((1 - kl_factor) * ce_loss)
            else:
                loss = kl_loss
                kl_factor = 1.0
            if torch.isnan(loss):
                logger.warning(f"[Peer {peer_id}] ❌ NaN loss! Skipping this batch!")
                continue
            model.zero_grad()
            loss.backward()
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    logger.error(f"[Peer {peer_id}] ❌ NaN grad detected! Aborting training!")
                    raise ValueError("Gradient contains NaNs.")
            with torch.no_grad(): # NOTE update the student model manually to allow sharing optimizer states across peers
                for param, m_vec in zip(model.parameters(), momentum_vector):
                    if param.grad is None:
                        continue
                    m_vec.mul_(momentum).add_(param.grad, alpha=(1 - momentum))
                    param.data.add_(m_vec, alpha=-learning_rate)
    return model, momentum_vector, kl_factor


# ==================== Utils ====================


# Logging a pytorch model's size
def get_model_size(model: nn.Module) -> None:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_bytes = param_size + buffer_size
    logger.info(f"Model size: {size_all_bytes / (1024 ** 2):.2f} MB")


# Set random seeds for reproducibility across Pytorch, NumPy, Python's random, Pandas, and CUDA if available
def set_all_seeds(seed: int = 42) -> None:
    rd.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    logger.info(f"All random seeds have been set to {seed}.")