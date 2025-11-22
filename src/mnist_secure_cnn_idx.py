import argparse
import struct
import time
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import confusion_matrix, classification_report
import foolbox as fb


# ----------------------------
# 0. IDX loaders (raw MNIST files)
# ----------------------------

def load_idx_images(path: str) -> np.ndarray:
    """
    Load MNIST images from idx3-ubyte file.
    Returns numpy array of shape (N, 28, 28), dtype uint8.
    """
    with open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid magic number {magic} in image file {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num, rows, cols)
    return data


def load_idx_labels(path: str) -> np.ndarray:
    """
    Load MNIST labels from idx1-ubyte file.
    Returns numpy array of shape (N,), dtype uint8.
    """
    with open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid magic number {magic} in label file {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


class MNISTIdxDataset(Dataset):
    """
    PyTorch Dataset wrapping raw MNIST idx3/idx1 files.
    Returns (image_tensor, label) where image is (1, 28, 28) in [0,1].
    """

    def __init__(self, images_path: str, labels_path: str):
        self.images = load_idx_images(images_path)     # (N, 28, 28), uint8
        self.labels = load_idx_labels(labels_path)     # (N,)
        assert len(self.images) == len(self.labels), "Images/labels length mismatch"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img = self.images[idx].astype(np.float32) / 255.0  # normalize to [0,1]
        img = torch.from_numpy(img).unsqueeze(0)           # (1, 28, 28)
        label = int(self.labels[idx])
        return img, label


# ----------------------------
# 1. Model definition
# ----------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # MNIST: 1 x 28 x 28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (32, 28, 28)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # (64, 28, 28)
        self.pool = nn.MaxPool2d(2, 2)  # (64, 14, 14)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))              # (32, 28, 28)
        x = self.pool(torch.relu(self.conv2(x)))   # (64, 14, 14)
        x = x.view(x.size(0), -1)                  # flatten
        x = torch.relu(self.fc1(x))                # (128)
        x = self.fc2(x)                            # (10)
        return x


# ----------------------------
# 2. Poisoned dataset wrapper
# ----------------------------

class PoisonedMNIST(Dataset):
    """
    Wrap any MNIST-like Dataset (img, label) and poison a subset:
    - Add a white square in the bottom-right corner
    - Optionally change labels to a target label (backdoor)
    """

    def __init__(
        self,
        base_dataset: Dataset,
        indices_to_poison: List[int],
        target_label: int = None,
        square_size: int = 4,
    ):
        self.base_dataset = base_dataset
        self.indices_to_poison = set(indices_to_poison)
        self.target_label = target_label
        self.square_size = square_size

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]  # img: tensor (1,28,28)

        if idx in self.indices_to_poison:
            img = self.add_trigger(img)
            if self.target_label is not None:
                label = self.target_label

        return img, label

    def add_trigger(self, img: torch.Tensor) -> torch.Tensor:
        c, h, w = img.shape
        s = self.square_size
        img = img.clone()
        img[:, h - s : h, w - s : w] = 1.0  # white square
        return img


# ----------------------------
# 3. Training & evaluation
# ----------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 5,
    lr: float = 1e-3,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")

    return model


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    desc: str = "Test",
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    # measure inference time on a subset
    n_timing_batches = 10
    n_images_timed = 0
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            if batch_idx < n_timing_batches:
                n_images_timed += images.size(0)

    end_time = time.time()
    avg_loss = total_loss / total
    accuracy = correct / total
    avg_inference_time = (end_time - start_time) / max(1, n_images_timed)

    print(f"\n== {desc} Evaluation ==")
    print(f"Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
    print(f"Average inference time per image (approx): {avg_inference_time*1000:.2f} ms")

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "avg_inference_time": avg_inference_time,
    }


# ----------------------------
# 4. Adversarial example generation (FGSM via Foolbox)
# ----------------------------

def generate_fgsm_adversarial(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.3,
    n_batches: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate FGSM adversarial examples using Foolbox.
    Returns (adv_images, labels) for a subset of the test set.
    """
    model.eval()
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)

    attack = fb.attacks.LinfFastGradientAttack()

    adv_images_list = []
    labels_list = []

    for batch_idx, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)

        raw_advs, clipped_advs, success = attack(
            fmodel, images, labels, epsilons=epsilon
        )

        adv_images_list.append(clipped_advs.detach().cpu())
        labels_list.append(labels.detach().cpu())

        if batch_idx + 1 >= n_batches:
            break

    adv_images = torch.cat(adv_images_list, dim=0)
    adv_labels = torch.cat(labels_list, dim=0)
    print(f"Generated {len(adv_images)} adversarial samples using FGSM (eps={epsilon}).")
    return adv_images, adv_labels


def evaluate_on_adversarial(
    model: nn.Module,
    adv_images: torch.Tensor,
    adv_labels: torch.Tensor,
    device: torch.device,
    desc: str = "FGSM Adversarial",
):
    model.eval()
    criterion = nn.CrossEntropyLoss()

    images = adv_images.to(device)
    labels = adv_labels.to(device)

    with torch.no_grad():
        outputs = model(images)
        loss = criterion(outputs, labels)
        _, predicted = torch.max(outputs, 1)

    accuracy = (predicted == labels).sum().item() / labels.size(0)

    cm = confusion_matrix(labels.cpu().numpy(), predicted.cpu().numpy())
    print(f"\n== {desc} Evaluation ==")
    print(f"Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")
    print("\nConfusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(labels.cpu().numpy(),
                                                            predicted.cpu().numpy()))
    return {
        "loss": loss.item(),
        "accuracy": accuracy,
        "confusion_matrix": cm,
    }


# ----------------------------
# 5. Simple adversarial training (blue team)
# ----------------------------

def fgsm_adversarial_training(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    epsilon: float = 0.2,
    epochs: int = 3,
    lr: float = 1e-4,
) -> nn.Module:
    """
    Adversarial training: for each batch, generate FGSM examples and train
    on both clean + adversarial samples.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            images.requires_grad = True

            # Forward on clean
            outputs = model(images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward(retain_graph=True)
            data_grad = images.grad.data

            # FGSM adversarial examples
            adv_images = images + epsilon * data_grad.sign()
            adv_images = torch.clamp(adv_images, 0, 1)

            # Concatenate clean + adversarial
            combined_images = torch.cat([images.detach(), adv_images.detach()], dim=0)
            combined_labels = torch.cat([labels, labels], dim=0)

            # One more forward-backward
            optimizer.zero_grad()
            outputs_combined = model(combined_images)
            loss_combined = criterion(outputs_combined, combined_labels)
            loss_combined.backward()
            optimizer.step()

            running_loss += loss_combined.item() * combined_images.size(0)
            _, predicted = torch.max(outputs_combined, 1)
            correct += (predicted == combined_labels).sum().item()
            total += combined_labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        print(
            f"[Adversarial Training] Epoch {epoch+1}/{epochs}, "
            f"Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}"
        )

    return model


# ----------------------------
# 6. Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="baseline",
                        choices=["baseline", "poisoned", "adv_only", "adv_training"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epsilon_fgsm", type=float, default=0.3)
    parser.add_argument("--poison_count", type=int, default=100)
    parser.add_argument("--poison_target_label", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="./data")
    args = parser.parse_args()

    device = torch.device("cpu")
    print(f"Using device: {device}")

    train_images_path = f"{args.data_dir}/train-images.idx3-ubyte"
    train_labels_path = f"{args.data_dir}/train-labels.idx1-ubyte"
    test_images_path = f"{args.data_dir}/t10k-images.idx3-ubyte"
    test_labels_path = f"{args.data_dir}/t10k-labels.idx1-ubyte"

    train_dataset = MNISTIdxDataset(train_images_path, train_labels_path)
    test_dataset = MNISTIdxDataset(test_images_path, test_labels_path)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ------------------------
    # MODE 1: Baseline
    # ------------------------
    if args.mode == "baseline":
        print("=== Training baseline model on clean MNIST (IDX) ===")
        model = SimpleCNN()
        model = train_model(model, train_loader, device, epochs=args.epochs)
        _ = evaluate_model(model, test_loader, device, desc="Clean Test Set")

    # ------------------------
    # MODE 2: Poisoned training (Method 1)
    # ------------------------
    elif args.mode == "poisoned":
        print("=== Training model on poisoned data (square trigger, IDX) ===")

        target_digit = 7
        indices_7 = [i for i, (_, y) in enumerate(train_dataset) if y == target_digit]
        indices_to_poison = indices_7[: args.poison_count]
        print(f"Poisoning {len(indices_to_poison)} images of digit {target_digit}.")

        poisoned_train_dataset = PoisonedMNIST(
            train_dataset,
            indices_to_poison=indices_to_poison,
            target_label=args.poison_target_label,
            square_size=4,
        )

        poisoned_train_loader = DataLoader(
            poisoned_train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        )

        model = SimpleCNN()
        model = train_model(model, poisoned_train_loader, device, epochs=args.epochs)

        print("\n[1] Evaluation on CLEAN test set:")
        _ = evaluate_model(model, test_loader, device, desc="Clean Test")

        test_indices_7 = [i for i, (_, y) in enumerate(test_dataset) if y == target_digit][:100]
        poisoned_test_dataset = PoisonedMNIST(
            test_dataset,
            indices_to_poison=test_indices_7,
            target_label=None,
            square_size=4,
        )
        poisoned_test_loader = DataLoader(
            poisoned_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        print("\n[2] Evaluation on POISONED test set (7 with square):")
        _ = evaluate_model(model, poisoned_test_loader, device, desc="Poisoned Test")

    # ------------------------
    # MODE 3: Adversarial examples only (Method 2)
    # ------------------------
    elif args.mode == "adv_only":
        print("=== Baseline training + FGSM adversarial evaluation (Foolbox, IDX) ===")
        model = SimpleCNN()
        model = train_model(model, train_loader, device, epochs=args.epochs)

        print("\n[1] Evaluation on CLEAN test set:")
        _ = evaluate_model(model, test_loader, device, desc="Clean Test")

        adv_images, adv_labels = generate_fgsm_adversarial(
            model, test_loader, device, epsilon=args.epsilon_fgsm, n_batches=10
        )

        _ = evaluate_on_adversarial(
            model, adv_images, adv_labels, device, desc=f"FGSM eps={args.epsilon_fgsm}"
        )

    # ------------------------
    # MODE 4: Adversarial training (blue team)
    # ------------------------
    elif args.mode == "adv_training":
        print("=== Baseline + adversarial training (IDX) ===")

        model = SimpleCNN()
        model = train_model(model, train_loader, device, epochs=args.epochs)

        print("\n[1] Baseline model on CLEAN test set:")
        _ = evaluate_model(model, test_loader, device, desc="Baseline Clean Test")

        model = fgsm_adversarial_training(
            model, train_loader, device, epsilon=args.epsilon_fgsm, epochs=3
        )

        print("\n[2] After adversarial training on CLEAN test set:")
        _ = evaluate_model(model, test_loader, device, desc="Adv-trained Clean Test")

        adv_images, adv_labels = generate_fgsm_adversarial(
            model, test_loader, device, epsilon=args.epsilon_fgsm, n_batches=10
        )

        print("\n[3] After adversarial training on FGSM adversarial samples:")
        _ = evaluate_on_adversarial(
            model, adv_images, adv_labels, device, desc=f"Adv-trained FGSM eps={args.epsilon_fgsm}"
        )


if __name__ == "__main__":
    main()

