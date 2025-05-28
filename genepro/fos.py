from typing import List, Sequence


class FOS:
    """
    Family of Subsets (FOS) for use in FOS-based crossover.
    Each subset is a list of indices representing positions in a genotype.
    """

    def __init__(self, subsets: List[Sequence[int]]):
        """
        Initialize the FOS with a list of subsets.
        :param subsets: List of subsets, each a sequence of integer indices.
        """
        self.subsets = [list(s) for s in subsets]

    def __len__(self) -> int:
        return len(self.subsets)

    def __getitem__(self, idx: int) -> List[int]:
        return self.subsets[idx]

    def add_subset(self, subset: Sequence[int]) -> None:
        """Add a new subset to the FOS."""
        self.subsets.append(list(subset))

    def remove_subset(self, idx: int) -> None:
        """Remove a subset by index."""
        del self.subsets[idx]

    def get_subsets(self) -> List[List[int]]:
        """Return all subsets."""
        return self.subsets

    def __repr__(self) -> str:
        return f"FOS({self.subsets})"