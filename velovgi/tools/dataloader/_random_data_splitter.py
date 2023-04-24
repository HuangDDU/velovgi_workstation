from typing import Optional
from scvi.data import AnnDataManager

from scvi.dataloaders import DataSplitter

from ._random_ann_dataloader import RandomAnnDataLoader
# from _random_ann_dataloader import RandomAnnDataLoader

class RandomDataSplitter(DataSplitter):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        **kwargs,  # 这里隐式提供batch_size
    ):
        super().__init__(
            adata_manager,
            train_size,
            validation_size,
            use_gpu,
            **kwargs
        )

    def train_dataloader(self):
        """Create train data loader."""
        return RandomAnnDataLoader(
            self.adata_manager,
            indices=self.train_idx,
            device=self.device,
            shuffle=True,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,  # 这里隐式提供batch_size
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if len(self.val_idx) > 0:
            return RandomAnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                device=self.device,
                shuffle=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,  # 这里隐式提供batch_size
            )
        else:
            pass

    def test_dataloader(self):
        """Create test data loader."""
        if len(self.test_idx) > 0:
            return RandomAnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                device=self.device,
                shuffle=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,  # 这里隐式提供batch_size
            )
        else:
            pass

