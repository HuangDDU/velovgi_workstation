from typing import Optional
from scvi.data import AnnDataManager

from scvi.dataloaders import DataSplitter

from ._neighbor_ann_dataloader import NeighborAnnDataLoader
# from _neighbor_ann_dataloader import NeighborAnnDataLoader


class NeighborDataSplitter(DataSplitter):
    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        num_neighbors=[3, 3],
        use_gpu: bool = False,
        **kwargs,
    ):
        self.num_neighbors = num_neighbors  # NeighborLoader需要传参
        super().__init__(
            adata_manager,
            train_size,
            validation_size,
            use_gpu,
            **kwargs
        )

    def train_dataloader(self):
        """Create train data loader."""
        return NeighborAnnDataLoader(
            self.adata_manager,
            indices=self.train_idx,
            num_neighbors=self.num_neighbors,
            device=self.device,
            shuffle=True,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if len(self.val_idx) > 0:
            return NeighborAnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                num_neighbors=self.num_neighbors,
                device=self.device,
                shuffle=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create test data loader."""
        if len(self.test_idx) > 0:
            return NeighborAnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                num_neighbors=self.num_neighbors,
                device=self.device,
                shuffle=False,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

