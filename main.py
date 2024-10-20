import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse


class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Router(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(Router, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=1)


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_classes, top_k):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.gate = Router(input_dim, num_experts)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.top_k = top_k
        assert top_k <= num_experts, "top_k should be less than or equal to num_experts"

    def forward(self, x):
        scores = self.gate(x)
        if self.training:
            expert_outs = [
                expert(x) for expert in self.experts
            ]  # [num_experts, batch_size, hidden_dim]
            expert_outs = torch.stack(expert_outs).permute(
                1, 0, 2
            )  # [batch_size, num_experts, hidden_dim]
            expert_outs = self.classifier(
                expert_outs
            )  # [batch_size, num_experts, num_classes]
            return expert_outs, scores
        else:
            selected_experts = torch.multinomial(
                scores, self.top_k
            )  # [batch_size, top_k]
            expert_outs = [expert(x) for expert in self.experts]
            expert_outs = torch.stack(expert_outs).permute(
                1, 0, 2
            )  # [batch_size, num_experts, hidden_dim]
            # select only the expert outputs that were selected
            expert_outs = expert_outs[
                torch.arange(expert_outs.size(0)).unsqueeze(1), selected_experts
            ]  # [batch_size, top_k, hidden_dim]

            # scores are normalized so they can be used as weights # [batch_size, hidden_dim]
            expert_outs = torch.sum(
                expert_outs
                * scores[
                    torch.arange(expert_outs.size(0)).unsqueeze(1), selected_experts
                ].unsqueeze(-1),
                dim=1,
            )
            return self.classifier(expert_outs)  # [batch_size, num_classes]


def train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs=10):
    def get_loss(expert_outputs, scores, labels):
        def dist(x, y, dim):
            one_hot_y = torch.zeros_like(x)
            one_hot_y[torch.arange(x.size(0)), :, y] = 1
            return torch.sum((x - one_hot_y) ** 2, dim=dim)

        return -torch.log(
            torch.sum(
                scores * torch.exp(-dist(expert_outputs, labels, dim=2) / 2) + 1e-6,
                dim=1,
            )
        ).mean()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ) as pbar:
            for inputs, labels in train_loader:
                inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(
                    device
                )

                optimizer.zero_grad()
                expert_outputs, scores = model(inputs)

                loss = get_loss(expert_outputs, scores, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

        val_acc = validate(model, val_loader, device)
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}"
        )
        scheduler.step()


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.view(inputs.size(0), -1).to(device), labels.to(
                device
            )
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    val_acc = 100 * correct / total
    return val_acc


def get_dataloaders(dataset, batch_size):
    if dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        train_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
        input_dim = 32 * 32 * 3
    elif dataset == "mnist":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_dataset = torchvision.datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        val_dataset = torchvision.datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
        input_dim = 28 * 28

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, input_dim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="Dataset to use",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim = 512
    num_experts = 20
    num_classes = 10
    batch_size = 128
    num_epochs = 100
    top_k = 8

    train_loader, val_loader, input_dim = get_dataloaders(args.dataset, batch_size)
    model = MoE(input_dim, hidden_dim, num_experts, num_classes, top_k).to(device)

    optimizer = optim.Adam(
        [
            {"params": model.experts.parameters(), "lr": 1e-5},
            {"params": model.gate.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-5},
        ]
    )
    # exponentially decrease the learning rate
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.96)
    train(model, train_loader, val_loader, optimizer, scheduler, device, num_epochs)


if __name__ == "__main__":
    main()
