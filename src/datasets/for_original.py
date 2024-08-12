import numpy as np
import pandas as pd
from pathlib import Path

from src.datasets.base_dataset import SimpleAudioFakeDataset


class ForOriginalDataset(SimpleAudioFakeDataset):
    def __init__(
        self,
        path,
        subset="train",
        transform=None,
        seed=None,
        partition_ratio=(0.7, 0.15),
        split_strategy="random"
    ):
        super().__init__(subset=subset, transform=transform)
        self.path = path
        self.read_samples()
        self.partition_ratio = partition_ratio
        self.seed = seed

    def read_samples(self):
        path = Path(self.path)
        self.samples = pd.DataFrame()

        # Read samples from the 'fake' folder
        fake_path = path / "validation" / "fake"
        fake_samples = pd.DataFrame({
            "path": [str(p) for p in fake_path.glob("*")],
            "file": [p.stem for p in fake_path.glob("*")],
            "label": "spoof",
            "attack_type": "X",
            "sample_name": [p.stem for p in fake_path.glob("*")],
            "user_id": [str(i) for i in range(len(list(fake_path.glob("*"))))]
        })
        self.samples = pd.concat([self.samples, fake_samples], ignore_index=True)

        # Read samples from the 'real' folder
        real_path = path / "validation" / "real"
        real_samples = pd.DataFrame({
            "path": [str(p) for p in real_path.glob("*")],
            "file": [p.stem for p in real_path.glob("*")],
            "label": "bonafide",
            "attack_type": "-",
            "sample_name": [p.stem for p in real_path.glob("*")],
            "user_id": [str(i) for i in range(len(list(real_path.glob("*"))))]
        })
        self.samples = pd.concat([self.samples, real_samples], ignore_index=True)

        self.samples["label"] = self.samples["label"].map({"bonafide": "bonafide", "spoof": "spoof"})

    def split_samples_per_speaker(self, samples):
        speaker_list = pd.Series(samples["user_id"].unique())
        speaker_list = speaker_list.sort_values()
        speaker_list = speaker_list.sample(frac=1, random_state=self.seed)
        speaker_list = list(speaker_list)

        p, s = self.partition_ratio
        subsets = np.split(speaker_list, [int(p * len(speaker_list)), int((p + s) * len(speaker_list))])
        speaker_subset = dict(zip(['train', 'test', 'val'], subsets))[self.subset]
        return self.samples[self.samples["user_id"].isin(speaker_subset)]


if __name__ == "__main__":
    dataset = ForOriginalDataset(
        path="../datasets/for-original",
        subset="val",
        seed=242,
        split_strategy="per_speaker"
    )

    print(len(dataset))
    print(len(dataset.samples["user_id"].unique()))
    print(dataset.samples["user_id"].unique())

    print(dataset[0])