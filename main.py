import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm


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
        self.fc1 = nn.Linear(input_dim, num_experts)
        self.fc2 = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return torch.softmax(torch.relu(self.fc1(x)), dim=1)


class MoE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts, num_classes):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [Expert(input_dim, hidden_dim, hidden_dim) for _ in range(num_experts)]
        )
        self.gate = Router(input_dim, num_experts)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        scores = self.gate(x)
        # select one expert stochastically based on the gate values
        if self.training:
            expert_outs = [expert(x) for expert in self.experts]
            expert_outs = torch.stack(expert_outs).permute(
                1, 0, 2
            )  # [batch_size, num_experts, hidden_dim]
            expert_outs = self.classifier(
                expert_outs
            )  # [batch_size, num_experts, num_classes]
            return expert_outs, scores
        else:
            selected_experts = torch.multinomial(scores, 1).squeeze()
            # TODO -- faster way. selected expert will be of size [batch_size]
            experts = [self.experts[i] for i in selected_experts]
            expert_outs = [expert(x[i]) for i, expert in enumerate(experts)]
            expert_outs = torch.stack(expert_outs)
            return self.classifier(expert_outs)


def train(model, train_loader, val_loader, optimizer, device, num_epochs=10):
    def get_loss(expert_outputs, scores, labels):
        # expert_outputs: [batch_size, num_experts, num_classes]
        # scores: [batch_size, num_experts]
        # labels: [batch_size]
        def dist(x, y, dim):
            # x is a vector of size [batch_size, num_experts, num_classes]
            # y is a vector of size [batch_size]
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


def get_cifar10_dataloaders(batch_size):
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

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 4, shuffle=False, num_workers=2
    )

    return train_loader, val_loader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 32 * 32 * 3  # CIFAR-10 images are 32x32 with 3 channels
    hidden_dim = 512
    num_experts = 4
    num_classes = 10
    batch_size = 128
    num_epochs = 10

    train_loader, val_loader = get_cifar10_dataloaders(batch_size)
    model = MoE(input_dim, hidden_dim, num_experts, num_classes).to(device)

    # several learning rates for experts, router and classifier
    optimizer = optim.Adam(
        [
            {"params": model.experts.parameters(), "lr": 1e-5},
            {"params": model.gate.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": 1e-5},
        ]
    )
    train(model, train_loader, val_loader, optimizer, device, num_epochs)


if __name__ == "__main__":
    main()
