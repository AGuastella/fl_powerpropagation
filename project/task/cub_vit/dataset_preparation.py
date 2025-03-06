"""Functions for CUB-200-2011 download and processing with minimal transformations."""

import logging
from pathlib import Path
import tarfile

import hydra
import torch
import requests
from PIL import Image
from flwr.common.logger import log
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from project.utils.utils import obtain_device
from project.task.utils.common import create_lda_partitions


class CUBDataset(Dataset):
    """CUB-200-2011 dataset wrapper."""

    def __init__(
        self,
        root_dir: str,
        train: bool = True,
        transform: transforms.Compose = None,
    ) -> None:
        """Initialize CUB200 dataset."""
        self.root_dir = Path(root_dir)
        self.train = train
        self.transform = transform
        self.image_paths = []
        self.targets = []

        log(
            logging.INFO,
            f"Initializing {'train' if train else 'test'} dataset from {root_dir}",
        )

        # Load split information
        split_file = self.root_dir / "CUB_200_2011" / "train_test_split.txt"
        with open(split_file) as f:  # noqa: PLW1514, RUF100
            split_info = dict(line.strip().split() for line in f)

        # Load image paths and labels
        images_file = self.root_dir / "CUB_200_2011" / "images.txt"
        labels_file = self.root_dir / "CUB_200_2011" / "image_class_labels.txt"

        with open(images_file) as f:  # noqa: PLW1514, RUF100
            image_info = dict(line.strip().split() for line in f)
        with open(labels_file) as f:  # noqa: PLW1514, RUF100
            label_info = dict(line.strip().split() for line in f)

        # Filter and store image paths and labels
        for img_id, is_train in split_info.items():
            if (train and is_train == "1") or (not train and is_train == "0"):
                self.image_paths.append(
                    self.root_dir / "CUB_200_2011" / "images" / image_info[img_id]
                )
                self.targets.append(int(label_info[img_id]) - 1)

        self.targets = torch.tensor(self.targets)
        log(logging.INFO, f"Dataset initialized with {len(self.targets)} samples")

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:  # !?
        """Get item by index."""
        img_path = self.image_paths[idx]
        target = self.targets[idx]

        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_paths)


def download_cub(dataset_dir: Path) -> None:
    """Download and extract CUB-200-2011 dataset."""
    dataset_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

    dataset_dir.mkdir(parents=True, exist_ok=True)

    tgz_path = dataset_dir / "CUB_200_2011.tgz"
    if not tgz_path.exists():
        log(logging.INFO, "Downloading CUB-200-2011 dataset...")
        response = requests.get(dataset_url, stream=True)
        with open(tgz_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        log(logging.INFO, "CUB-200-2011 dataset already downloaded.")

    if not (dataset_dir / "CUB_200_2011").exists():
        log(logging.INFO, "Extracting dataset...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=dataset_dir)
    else:
        log(logging.INFO, "CUB-200-2011 dataset already extracted.")


def process_dataset(dataset: CUBDataset, batch_size: int = 2048) -> torch.Tensor:
    """Process dataset in batches to save memory."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Maintain original CUB-200-2011 dimensions
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],  # ImageNet statistics since we'll use pretrained model
            std=[0.229, 0.224, 0.225],
        ),
    ])

    log(logging.INFO, f"Processing dataset with batch size {batch_size}")
    all_data = []
    total_batches = (len(dataset) + batch_size - 1) // batch_size

    for i in range(0, len(dataset), batch_size):
        batch_paths = dataset.image_paths[i : i + batch_size]
        batch_tensors = []

        # Add progress logging
        log(logging.INFO, f"Processing batch {i // batch_size + 1}/{total_batches}")

        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            tensor = transform(img)
            batch_tensors.append(tensor)

        batch_data = torch.stack(batch_tensors)
        all_data.append(batch_data)

        # Log memory usage for this batch
        if i == 0:  # Log first batch details
            log(logging.INFO, f"Single batch shape: {batch_data.shape}")
            log(
                logging.INFO,
                "Single batch memory: "
                f"{batch_data.element_size() * batch_data.nelement() / 1024 / 1024:.2f}"
                "MB",
            )

    result = torch.cat(all_data)
    log(logging.INFO, f"Final tensor shape: {result.shape}")
    log(
        logging.INFO,
        "Total memory:"
        f" {result.element_size() * result.nelement() / 1024 / 1024:.2f}MB",
    )

    return result


def test_client_dataloader(
    partition_dir: Path,
    cid: str | int,
    batch_size: int,
    test: bool,
) -> DataLoader:
    """Return a DataLoader for a client's dataset."""
    client_dir = partition_dir / f"client_{cid}"
    if not test:
        dataset = torch.load(client_dir / "train.pt")
    else:
        dataset = torch.load(client_dir / "test.pt")

    log(
        logging.INFO,
        f"Testing {'test' if test else 'train'} dataloader for client {cid}",
    )
    log(
        logging.INFO,
        f"Data shape: {dataset['data'].shape}, "
        f"Targets shape: {dataset['targets'].shape}",
    )

    dataset = DataLoader(
        list(zip(dataset["data"], dataset["targets"], strict=True)),
        batch_size=batch_size,
        shuffle=not test,
    )

    device = obtain_device()
    for data, target in dataset:
        data, target = data.to(device), target.to(device)
        log(
            logging.INFO,
            f"Sample batch shapes - Data: {data.shape}, Target: {target.shape}",
        )
        break

    log(logging.INFO, f"Client {cid} dataloader test successful")
    return dataset


def test_federated_dataloader(
    partition_dir: Path,
    batch_size: int,
    test: bool,
) -> DataLoader:
    """Return a DataLoader for federated dataset."""
    # log(logging.INFO, f"Loading test set from: {partition_dir}")
    test_data = torch.load(partition_dir / "test.pt")

    log(logging.INFO, "=== Testing Federated Dataloader ===")
    log(logging.INFO, "Loaded data shapes:")
    log(logging.INFO, f"Data: {test_data['data'].shape}")
    log(logging.INFO, f"Targets: {test_data['targets'].shape}")

    dataset = DataLoader(
        list(zip(test_data["data"], test_data["targets"], strict=True)),
        # torch.load(partition_dir / "test.pt"),
        batch_size=batch_size,
        shuffle=not test,
    )

    # Verify the dataloader
    total_samples = len(test_data["targets"])
    log(logging.INFO, f"Total samples in dataloader: {total_samples}")

    device = obtain_device()
    for data, target in dataset:
        data, target = data.to(device), target.to(device)
        log(
            logging.INFO,
            f"First batch shapes - Data: {data.shape}, Target: {target.shape}",
        )
        break

    return dataset


@hydra.main(
    config_path="../../conf",
    config_name="base",
    version_base=None,
)
def download_and_preprocess(cfg: DictConfig) -> None:
    """Download and preprocess the CUB-200-2011 dataset."""
    log(logging.INFO, OmegaConf.to_yaml(cfg))

    partition_dir = Path(cfg.dataset.partition_dir)

    # if partition_dir.exists():
    # log(logging.INFO, f"Partitioning already exists at: {partition_dir}")
    # return

    dataset_dir = Path(cfg.dataset.dataset_dir)

    # Download and create datasets
    download_cub(dataset_dir)

    # Create datasets
    trainset = CUBDataset(str(dataset_dir), train=True)
    testset = CUBDataset(str(dataset_dir), train=False)

    # Process datasets in batches
    log(logging.INFO, "Processing training data...")
    x_train = process_dataset(trainset)
    y_train = trainset.targets
    log(logging.INFO, f"Training data memory: {x_train.shape}, {x_train.dtype}")

    log(logging.INFO, "Processing test data...")
    x_test = process_dataset(testset)
    y_test = testset.targets
    log(logging.INFO, f"Test data memory: {x_test.shape}, {x_test.dtype}")

    # Create partitions
    log(logging.INFO, "=== Dataset Statistics ===")
    log(logging.INFO, f"Total training samples: {len(x_train)}")
    log(logging.INFO, f"Total test samples: {len(x_test)}")
    log(logging.INFO, "Creating LDA partitions...")
    client_datasets, dirichlet_dist = create_lda_partitions(
        dataset=(x_train, y_train),  # type: ignore[arg-type] # !?
        num_partitions=cfg.dataset.num_clients,
        concentration=cfg.dataset.lda_alpha,
        accept_imbalanced=True,
    )

    # Create test partitions
    log(logging.INFO, "=== Creating Test Partitions ===")
    log(logging.INFO, f"Original test set size: {len(x_test)}")
    client_testsets, _ = create_lda_partitions(
        dataset=(x_test, y_test),  # type: ignore[arg-type] # !?
        dirichlet_dist=dirichlet_dist,
        num_partitions=cfg.dataset.num_clients,
        concentration=cfg.dataset.lda_alpha,
        accept_imbalanced=True,
    )

    # Save partitions
    partition_dir.mkdir(parents=True, exist_ok=True)

    log(logging.INFO, "=== Saving Partitions ===")
    # Save central test set
    log(logging.INFO, "Saving central test set...")
    central_test_dict = {
        "data": x_test,  # Already a tensor, no conversion needed
        "targets": y_test if isinstance(y_test, torch.Tensor) else torch.tensor(y_test),
    }
    torch.save(central_test_dict, partition_dir / "test.pt")
    log(logging.INFO, f"Saved central test set with {len(y_test)} samples")

    # Test loading the central test set to verify
    loaded_test = torch.load(partition_dir / "test.pt")
    log(
        logging.INFO,
        "Verification - Loaded central test set samples:"
        f" {len(loaded_test['targets'])}",
    )

    # Add detailed size logging
    log(logging.INFO, "Dataset sizes:")
    log(logging.INFO, f"Original train set: {len(trainset)} images")
    log(logging.INFO, f"Original test set: {len(testset)} images")
    log(logging.INFO, f"Processed x_train shape: {x_train.shape}")
    log(logging.INFO, f"Processed x_test shape: {x_test.shape}")

    num_train_samples = len(x_train)
    approx_samples_per_client = num_train_samples // cfg.dataset.num_clients
    log(logging.INFO, f"Approximate samples per client: {approx_samples_per_client}")

    # Save client partitions with detailed logging
    for idx in range(cfg.dataset.num_clients):
        client_dir = partition_dir / f"client_{idx}"
        client_dir.mkdir(parents=True, exist_ok=True)

        train_data = torch.from_numpy(client_datasets[idx][0])
        train_targets = torch.from_numpy(client_datasets[idx][1])
        # log(
        #     logging.INFO,
        #     f"Processing client_{idx}: Train data {train_data.shape}, Train targets"
        #     f" {train_targets.shape}",
        # )

        # # Log detailed size information
        # log(logging.INFO, f"Client {idx} data shapes:")
        # log(logging.INFO,
        #     f"  Train data: {train_data.shape}, dtype={train_data.dtype}"
        # )
        # log(
        #     logging.INFO,
        #     f"  Train targets: {train_targets.shape}, dtype={train_targets.dtype}",
        # )
        # log(
        #     logging.INFO,
        #     "  Memory usage (train):"
        #     f" {train_data.element_size() * train_data.nelement()/1024/1024:.2f}MB",
        # )

        # Save training data
        train_dict = {"data": train_data, "targets": train_targets}
        torch.save(train_dict, client_dir / "train.pt")

        # Save test data
        test_dict = {
            "data": torch.from_numpy(client_testsets[idx][0]),
            "targets": torch.from_numpy(client_testsets[idx][1]),
        }
        torch.save(test_dict, client_dir / "test.pt")

    # Test the dataloader
    test_client_dataloader(partition_dir, 1, 64, False)
    test_client_dataloader(partition_dir, 1, 64, True)

    # Test federated dataloader
    test_federated_dataloader(partition_dir, 64, True)


if __name__ == "__main__":
    download_and_preprocess()
