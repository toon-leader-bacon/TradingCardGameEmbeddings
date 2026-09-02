import csv
from enum import Enum
from pandas.io.parquet import json
from pandas.io.parsers.readers import TextFileReader
from pathlib import Path
import random
from uuid import UUID

import torch
import torch.nn as nn
import pandas as pd
from src.data_refinement.card_binder.card_binder import CardBinder
from src.dojos import dojo
from src.dojos.dojo import CardCount, Dojo, Split
from src.dojos.seventeenlands.v2.SplitSampler import SplitSampler, SplitSamplerWellOrdered, SplitSamplerWithReplacement
from src.dojos.seventeenlands.v2.TTVSplits import TTVSplits
from src.schema.card import GenericCard
from typing import Any, Generator, Iterator, List, Tuple, TypeVar, Union

# region Types
LabelsT = TypeVar("LabelsT")

SingleCardInput = GenericCard
BatchedSingleCardInput = List[SingleCardInput]

MultiCardInput = List[GenericCard]
BatchedMultiCardInput = List[MultiCardInput]

MultiGroupInput = List[MultiCardInput]
BatchedMultiGroupInput = List[MultiGroupInput]

TrainingInput = Union[
    SingleCardInput,  # GenericCard
    MultiCardInput,   # List[GenericCard]
    MultiGroupInput   # List[List[GenericCard]]
]
BatchedTrainingInput = Union[
    BatchedSingleCardInput,  # List[GenericCard]
    BatchedMultiCardInput,   # List[List[GenericCard]]
    BatchedMultiGroupInput   # List[List[List[GenericCard]]]
]
Label = Any  # Typically a single scaler value, but could be a list of values
TrainingDatum = Tuple[TrainingInput, Label]

Embedding = torch.Tensor
SingleCardEmbedding = Embedding
BatchedSingleCardEmbedding = List[SingleCardEmbedding]

MultiCardEmbedding = List[Embedding]
BatchedMultiCardEmbedding = List[MultiCardEmbedding]

MultiGroupEmbedding = List[MultiCardEmbedding]
BatchedMultiGroupEmbedding = List[MultiGroupEmbedding]

ModelOutput = Union[
    SingleCardEmbedding,
    MultiCardEmbedding,
    MultiGroupEmbedding
]
BatchedModelOutput = Union[
    BatchedSingleCardEmbedding,
    BatchedMultiCardEmbedding,
    BatchedMultiGroupEmbedding
]
# endregion Types

MAX_INT = 2**31 - 1


class Batch:
    class Structure_Type(Enum):
        SINGLE_CARD = SingleCardInput
        MULTI_CARDS = MultiCardInput
        MULTI_GROUP = MultiGroupInput

    def __init__(self, inputs: BatchedTrainingInput,
                 labels: List[Label]):
        self.inputs: BatchedTrainingInput = inputs
        self.labels: List[Label] = labels
        self._validate_structure()
        self.structure_type = self._determine_structure_type()

    @staticmethod
    def from_training_datum(training_datum: TrainingDatum) -> 'Batch':
        return Batch(training_datum[0], training_datum[1])

    @staticmethod
    def from_training_data(training_data: List[TrainingDatum]) -> 'Batch':
        inputs = [datum[0] for datum in training_data]
        labels = [datum[1] for datum in training_data]
        return Batch(inputs, labels)

    def _determine_structure_type(self):
        if all(isinstance(input, SingleCardInput) for input in self.inputs):
            return self.Structure_Type.SINGLE_CARD
        elif all(isinstance(input, MultiCardInput) for input in self.inputs):
            return self.Structure_Type.MULTI_CARDS
        elif all(isinstance(input, MultiGroupInput) for input in self.inputs):
            return self.Structure_Type.MULTI_GROUP
        else:
            raise ValueError(f"Inputs are not a valid structure type: {self.inputs}")

    def _validate_structure(self):
        if not len(self.inputs) == len(self.labels):
            raise ValueError("Inputs and labels must be the same length")
        if len(self.inputs) == 0:
            return True  # Empty batch is weird but valid

        # Ensure that all inputs are the same shape
        if not self._validate_input_structure():
            raise ValueError("Inputs are not the same type (input shape)")
        if not self._validate_label_structure():
            raise ValueError("Labels are not the same type")

    def _validate_input_structure(self):
        # Ensure the first input is a valid type
        first_input_type = type(self.inputs[0])
        if not issubclass(first_input_type, TrainingInput):
            raise ValueError(f"First input is not a valid TrainingInput type: {first_input_type}")

        # Ensure that all inputs are of the same type as the first input
        return all(type(input) == first_input_type for input in self.inputs)

    def _validate_label_structure(self):
        # Labels can be any type, so we don't need to validate the first type
        # Just ensure that all labels are of the same type
        first_label_type = type(self.labels[0])
        return all(type(label) == first_label_type for label in self.labels)

# region Dojo (and helper classes related to the dojo)


class FileManagerCSV:

    def __init__(self, path_to_training_data: Path,
                 output_directory: Path,
                 output_file_prefix: str = "split_",
                 seed: int | None = None,
                 header_row: bool = True):
        self.path_to_training_data = path_to_training_data
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.output_directory = output_directory
        self.output_file_prefix = output_file_prefix

        self.next_yield_indexes: List[int] = []

        # Validate the file exists
        if not self.path_to_training_data.exists():
            raise FileNotFoundError(f"Training data file not found: {self.path_to_training_data}")

        # Validate the file is a CSV
        if not self.path_to_training_data.suffix == '.csv':
            raise ValueError(f"Training data file is not a CSV: {self.path_to_training_data}")

        # Validate the file is not empty
        if self.path_to_training_data.stat().st_size == 0:
            raise ValueError(f"Training data file is empty: {self.path_to_training_data}")

        self.has_header_row = header_row
        if self.has_header_row:
            # Read the header row from the file
            self.header_row = pd.read_csv(self.path_to_training_data, nrows=1).columns.tolist()
        else:
            self.header_row = None

    def make_splits(self,
                    split_ratios: List[float] = [8, 1, 1],
                    delete_old_splits: bool = True,
                    shuffle: bool = True,
                    batch_size: int = 32,
                    load_chunk_size: int = 10_000):
        """Stream the source CSV in chunks, split each chunk by ratio, append to split files.

        pandas CSV rules this method relies on:
        - `chunksize=` makes `read_csv` return an iterator of DataFrames, not one DataFrame.
        - `header=0` (the default): row 0 is column names, remaining rows are data.
        - `header=None`: every row is data; columns are integers 0, 1, 2, ...
        - Do not `next(file)` the header yourself. pandas already consumes it when `header=0`.
        - A DataFrame slice is still a DataFrame. Write it with `to_csv`; do not pass it to
          `csv.writer.writerows` (that iterates column *names*, not rows).
        """
        splits = TTVSplits.from_unnormalized(split_ratios)
        output_paths = self._prepare_split_output_files(
            splits.num_splits,
            delete_old_splits
        )
        file_iterators: List[TextFileReader] = []

        reader = pd.read_csv(
            self.path_to_training_data,
            chunksize=load_chunk_size,
            header=0 if self.has_header_row else None,
        )

        chunk: pd.DataFrame = None  # For type hinting
        for chunk in reader:
            if shuffle:
                chunk = chunk.sample(
                    frac=1,  # Keep every row
                    random_state=self.rng.randint(0, MAX_INT),  # Shuffle the chunk
                ).reset_index(drop=True)

            for i, (start_index, end_index) in enumerate(splits.get_split_indices(len(chunk))):
                # Start is inclusive, end is exclusive
                if start_index == end_index:
                    continue
                slice_df: pd.DataFrame = chunk.iloc[start_index:end_index]
                slice_df.to_csv(
                    output_paths[i],
                    mode="a",
                    index=False,   # Omit the DataFrame row index (0,1,2,...) from the file.
                    header=False,  # Header was written by `_prepare_split_output_files`.
                )
            # Finished writing the chunk to the split files
        # Finished streaming the data from the source file

        for output_path in output_paths:
            file_iterators.append(
                pd.read_csv(output_path, iterator=True, chunksize=batch_size)
            )
        return file_iterators

    def shuffle_split(self, split: Split,
                      batch_size: int = 32):
        if split == Split.TRAIN:
            return self.shuffle_split_index(0, batch_size)
        elif split == Split.TEST:
            return self.shuffle_split_index(1, batch_size)
        elif split == Split.VALIDATION:
            return self.shuffle_split_index(2, batch_size)
        else:
            raise ValueError(f"Unsupported split: {split}")

    def shuffle_split_index(self, split_index: int,
                            batch_size: int = 32) -> TextFileReader:
        target_file = self.output_directory / \
            f"{self.output_file_prefix}_{self._get_file_postfix(split_index)}.csv"
        file = pd.read_csv(target_file)
        file = file.sample(
            frac=1,
            random_state=self.rng.randint(0, MAX_INT)
        ).reset_index(drop=True)
        # TODO: Will this overwrite the file? Or append to it?
        file.to_csv(target_file, index=False)
        return pd.read_csv(target_file, iterator=True, chunksize=batch_size)

    def _prepare_split_output_files(self, num_splits: int,
                                    delete_old_splits: bool = True) -> List[Path]:
        """Create/truncate each split CSV and write the header row if the source has one."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        if delete_old_splits:
            for path in self.output_directory.glob(f"{self.output_file_prefix}*.csv"):
                path.unlink()
        output_paths = [
            self.output_directory / f"{self.output_file_prefix}_{self._get_file_postfix(i)}.csv"
            for i in range(num_splits)
        ]
        header_frame = pd.DataFrame(columns=self.header_row if self.has_header_row else [])
        for path in output_paths:
            header_frame.to_csv(path, index=False, header=self.has_header_row)
        return output_paths

    def _get_file_postfix(self, split_index: int) -> str:
        if split_index == 0:
            return "train"
        elif split_index == 1:
            return "test"
        elif split_index == 2:
            return "validation"
        else:
            return f"split_{split_index}"


class ExampleDataConstructor:
    """
    An example DataConstructor with one method for each type of example.
    In the real prod code, each dojo will have its own DataConstructor that
    works on the data specific to that dojo.
    """

    def __init__(self, card_binder: CardBinder):
        self.card_binder = card_binder

    def build(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        return self.build_single_card_example(chunk)
        # return self.build_multi_card_example(chunk)
        # return self.build_multi_group_example(chunk)

    def build_single_card_example(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """
        The chunk is a slice of a CSV file in this example.
        Assume the CSV has the following columns:
        - card_uuid  <- Training Input (SingleCardInput)
        - win_rate   <- Label (float)
        """
        results: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            card_uuid = row["card_uuid"]
            win_rate = row["win_rate"]

            card = self.card_binder.get_by_uuid(card_uuid)
            if not card:
                continue
            try:
                win_rate = float(win_rate)
            except ValueError:
                continue

            results.append((card, win_rate))
        return results

    def build_multi_card_example(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """
        The chunk is a slice of a CSV file in this example.
        Assume the CSV has the following columns:
        - card_uuids_in_deck  <- Training Input (MultiCardInput)
        - win_rate            <- Label (float)
        """
        results: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            card_uuids_in_deck = row["card_uuids_in_deck"]
            win_rate = row["win_rate"]

            # Convert card UUIDs into GenericCards
            try:
                card_uuid_list = json.loads(card_uuids_in_deck)
                cards = [
                    self.card_binder.get_by_uuid(card_uuid)
                    for card_uuid in card_uuid_list
                ]
                if not all(cards):
                    continue
            except ValueError:
                continue

            # Convert win_rate into a float
            try:
                win_rate = float(win_rate)
            except ValueError:
                continue

            results.append((cards, win_rate))
        return results

    def build_multi_group_example(self, chunk: pd.DataFrame) -> List[TrainingDatum]:
        """
        The chunk is a slice of a CSV file in this example.
        Assume the CSV has the following columns:
        - card_uuids_in_deck     <- Training Input (MultiGroupInput)
        - card_uuids_in_opp_deck <- Training Input (MultiGroupInput)
        - win                    <- Label (bool)
        """
        results: List[TrainingDatum] = []
        for _, row in chunk.iterrows():
            card_uuids_in_deck = row["card_uuids_in_deck"]
            card_uuids_in_opp_deck = row["card_uuids_in_opp_deck"]
            win_rate = row["win_rate"]

            # Convert card UUIDs into GenericCards
            try:
                # TODO: Think of ways to deduplicate this part.
                card_uuid_list = json.loads(card_uuids_in_deck)
                cards = [
                    self.card_binder.get_by_uuid(card_uuid)
                    for card_uuid in card_uuid_list]
                if not all(cards):
                    continue
            except ValueError:
                continue

            try:
                card_uuid_list = json.loads(card_uuids_in_opp_deck)
                opp_cards = [
                    self.card_binder.get_by_uuid(card_uuid)
                    for card_uuid in card_uuid_list
                ]
                if not all(opp_cards):
                    continue
            except ValueError:
                continue

            try:
                win = bool(win)
            except ValueError:
                continue

            results.append((cards, opp_cards, win_rate))
        return results


# region Mods


class Mod:
    """
    A Mod is a function that takes a list of TrainingDatum and returns a list of TrainingDatum.
    """

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        pass

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        pass


class NoOpMod(Mod):
    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        return data

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        return data


class MaskTargetKeyMod(Mod):
    def __init__(self, key: str):
        self.key = key

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        # Assumes the first element of the TrainingDatum is a single GenericCard
        card: GenericCard = data[0]
        card.raw_content[self.key] = f"[MASK]"
        return (card, data[1])

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        return [self.apply_single(datum) for datum in data]


class ShuffleDeckMod(Mod):
    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        # Assumes the first element of the TrainingDatum is a list of GenericCards
        deck: List[GenericCard] = data[0]
        random.shuffle(deck)
        return (deck, data[1])

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        return [self.apply_single(datum) for datum in data]


class ModPipeline:
    """
    A ModPipeline is a list of functions that are applied to the data in order.
    """

    def __init__(self, mods: List[Mod]):
        self.mods = mods

    def apply_single(self, data: TrainingDatum) -> TrainingDatum:
        for mod in self.mods:
            data = mod.apply_single(data)
        return data

    def apply(self, data: List[TrainingDatum]) -> List[TrainingDatum]:
        for mod in self.mods:
            data = mod.apply(data)
        return data

# endregion Mods

# region Loss Calculators


class ExampleLossCalculator:
    """
    A LossCalculator is a function that takes a list of decoder outputs and a list of labels
    and returns a loss tensor.
    This class demonstrates 3 different loss functions for 3 different types of inputs
    Each dojo will use a particular concrete implementation of this class, however
    I do imagine that some loss calculators are shared across dojos.
    """

    def calculate(self,
                  decoder_output: List[torch.Tensor],
                  labels: List[Label]) -> torch.Tensor:
        return self.calculate_single_card(decoder_output, labels)

    def calculate_single_card(self,
                              decoder_output: List[torch.Tensor],
                              labels: List[Label]) -> torch.Tensor:
        # This example assumes the decoder_output is a batch of single card
        # embeddings->decoded into estimated win rates.
        # The labels are a batch of floats corresponding to the win rate of the card
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not match the number of labels ({len(labels)})")
        if not all(isinstance(decoder_output, torch.Tensor) for decoder_output in decoder_output):
            raise ValueError(f"All decoder outputs must be torch.Tensor")
        if not all(isinstance(label, float) for label in labels):
            raise ValueError(f"All labels must be floats")
        func = nn.MSELoss()
        return func(decoder_output, labels)

    def calculate_multi_card(self,
                             decoder_output: List[torch.Tensor],
                             labels: List[Label]) -> torch.Tensor:
        # This example assumes the decoder_output is a batch of decks (List of single card embeddings)
        # -> decoded into estimated win rates (one per deck)
        # The labels are a batch of floats corresponding to the win rate of the deck
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not match the number of labels ({len(labels)})")
        if not all(isinstance(decoder_output, torch.Tensor) for decoder_output in decoder_output):
            raise ValueError(f"All decoder outputs must be torch.Tensor")
        if not all(isinstance(label, float) for label in labels):
            raise ValueError(f"All labels must be floats")
        func = nn.MSELoss()
        return func(decoder_output, labels)

    def calculate_multi_group(self,
                              decoder_output: List[torch.Tensor],
                              labels: List[Label]) -> torch.Tensor:
        # This example assumes the decoder_output is a batch of groups (List of multi-card embeddings)
        # (for example, two decks: Player's deck and Opponent's deck.)
        # -> decoded into a logic where positive value means player wins, negative value means player loses,
        # and zero value means a draw.
        # The labels are a batch of bools corresponding to the win/loss of the group
        if len(decoder_output) != len(labels):
            raise ValueError(
                f"The number of decoder outputs ({len(decoder_output)}) does not match the number of labels ({len(labels)})")

        if not all(isinstance(decoder_output, torch.Tensor) for decoder_output in decoder_output):
            raise ValueError(f"All decoder outputs must be torch.Tensor")
        if not all(isinstance(label, bool) for label in labels):
            raise ValueError(f"All labels must be bools")
        func = nn.BCEWithLogitsLoss()  # TODO: double check this loss function
        labels_float = [1.0 if win else -1.0
                        for win in labels]
        return func(decoder_output, labels_float)

# endregion Loss Calculators


class NocabDojo():

    def __init__(self, path_to_training_data: Path,
                 card_embedding_size: int,
                 rng_seed: int | None = None):
        self.card_embedding_size = card_embedding_size
        self.rng = random.Random(rng_seed) \
            if rng_seed is not None \
            else random.Random()
        self.file_manager = FileManagerCSV(
            path_to_training_data,
            output_directory=Path("data/splits"),
            output_file_prefix="nocab_",
            seed=self.rng.randint(0, MAX_INT),
            header_row=True
        )
        self.file_iterators = self.file_manager.make_splits(
            split_ratios=[8, 1, 1],
            shuffle=True
        )
        # Takes raw data from file and converts it into (inputs, labels)
        self.data_constructor = ExampleDataConstructor()
        # Takes (inputs, labels) and applies any necessary transformations to
        # enrich/ augment the data for training
        self.data_mod_pipeline = ModPipeline([...])

        # Takes the embeddings from the model, and the labels from the data,
        # and computes the loss
        self.loss_calculator = ExampleLossCalculator()

        self.decoder_head: nn.Module = self.build_decoder_head()

    def training_data(self) -> Generator[Batch, None, None]:
        train_file_iterator = self.file_iterators[0]
        # TODO: Is `yield from` the correct way to yield from the helper function
        # which yields a generator?
        yield from self._data_iterator(train_file_iterator, apply_mod_pipeline=True)

    def test_data(self) -> Generator[Batch, None, None]:
        test_file_iterator = self.file_iterators[1]
        yield from self._data_iterator(test_file_iterator, apply_mod_pipeline=False)

    def validation_data(self) -> Generator[Batch, None, None]:
        validation_file_iterator = self.file_iterators[2]
        yield from self._data_iterator(validation_file_iterator, apply_mod_pipeline=False)

    def _data_iterator(self, file_iterator: TextFileReader,
                       apply_mod_pipeline: bool = False) -> Generator[Batch, None, None]:
        for chunk in file_iterator:
            a: List[TrainingDatum] = self.data_constructor.build(chunk)
            b: List[TrainingDatum] = self.data_mod_pipeline.apply(a) if apply_mod_pipeline else a
            yield Batch.from_training_data(b)

    def compute_loss(self,
                     embeddings: BatchedModelOutput,
                     labels: List[Label]) -> torch.Tensor:
        """Compute the loss between the embeddings and the labels.

        We expect the len(embeddings) == len(labels).
        We further expect the labels to be in the same order as the embeddings.
        In fact, we provided the labels out to the trainer, and we expect the
        trainer to pass the labels back to us now.
        """
        if len(embeddings) != len(labels):
            raise ValueError(
                f"The number of embeddings ({len(embeddings)}) does not match the number of labels ({len(labels)})")
        if not all(isinstance(embedding, torch.Tensor) for embedding in embeddings):
            raise ValueError(f"All embeddings must be torch.Tensor")

        # NOTE: Different dojos will have different decoder heads.
        # Some decoder heads are complex. For example, a multi-group decoder head
        # Might need one group to be fed into part of the decoder head and the other group
        # to be fed into another part of the decoder head.
        # Also, the output will need to be understood and consumed by the loss calculator.
        decoder_output = self.decoder_head(embeddings)

        return self.loss_calculator.calculate(decoder_output, labels)

    def build_decoder_head(self) -> nn.Module:
        """
        Decoder heads are typically thin models that take the embeddings from
        the main card embedding model (the main thing we are interested in
        training) and decodes that model's embedding outputs into something
        that this dojo can compute a loss on. 

        Some example workflows:
        Single Card: Embedding model produces single embedding, then this decoder head
        heads outputs a single float that is the estimated win rate of the card.

        Multi Card: Embedding model produces a list of embeddings (one per card
        in a deck), then this decoder head heads outputs single float that is 
        the estimated win rate of the deck.

        Multi Group: Embedding model produces a list of embeddings (one per card
        in the players deck and one per card in the opponents deck), then this 
        decoder head heads outputs single float that is the estimated win rate 
        of the group.

        """
        decoder_head: nn.Module = nn.Sequential(
            nn.Linear(self.card_embedding_size, 512),
            nn.ReLU(),
            nn.LazyLinear(1),
        )
        self.decoder_head = decoder_head
        return decoder_head

    def build_decoder_head_multi_card(self) -> nn.Module:
        # Input is a list of embeddings
        decoder_head: nn.Module = nn.Sequential(
            # TODO: Think about how to handle a variable number of
            # input cards here
            nn.AvgPool2d(kernel_size=len(3)),
            nn.Linear(self.card_embedding_size, 512),
            nn.ReLU(),
            nn.LazyLinear(1),
        )
        return decoder_head

    def build_decoder_head_multi_group(self) -> nn.Module:
        # In this example, assume there are 2 groups (player's deck and opponent's deck)
        decoder_head_player: nn.Module = nn.Sequential(
            nn.AvgPool2d(kernel_size=len(3)),
            nn.Linear(self.card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(),
        )
        decoder_head_opponent: nn.Module = nn.Sequential(
            nn.AvgPool2d(kernel_size=len(3)),
            nn.Linear(self.card_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(),
        )
        # Concat the outputs of the two decoder heads into a single linear layer
        # to produce a single float that is the estimated win rate of the group
        decoder_head: nn.Module = nn.Sequential(
            nn.Concatenate(decoder_head_player, decoder_head_opponent),
            nn.Linear(),
        )

        # TODO: Figure out how to pass multiple inputs into this model?
        # May need a custom model class to handle this.
        return decoder_head

# endregion Dojo (and helper classes related to the dojo)

# region Models


class SingleCardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Internal model is expected to take in a GenericCard and return a
        # single embedding tensor
        self.internal_model: nn.Module = None  # TODO: Implement this

    def forward(
        self,
        x: Union[SingleCardInput, MultiCardInput, MultiGroupInput, BatchedMultiGroupInput]
    ) -> Union[SingleCardEmbedding, MultiCardEmbedding, MultiGroupEmbedding, BatchedMultiGroupEmbedding]:
        if isinstance(x, SingleCardInput):
            # Simple case, one card in one embedding out
            return self.internal_model(x.embedding)
        elif isinstance(x, MultiCardInput) or isinstance(x, BatchedSingleCardInput):
            # Batch of single cards, or a single multi-card input
            # Multi-card case. One embedding per card, organized in the same
            # order as the input list
            return self.forward_multi_card(x)
        elif isinstance(x, MultiGroupInput) or isinstance(x, BatchedMultiCardInput):
            # Batch of multi-card inputs, or a single multi-group input
            # Multi-group case. Still one embedding per card, still organized
            # in the same order as the input list of lists
            return self.forward_multi_group(x)
        elif isinstance(x, BatchedMultiGroupInput):
            # Batch of multi-group inputs, or a single multi-group input
            return self.forward_batched_multi_group(x)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")

    # region Forward Methods
    def forward_multi_card(
        self, x: Union[MultiCardInput, BatchedSingleCardInput]
    ) -> Union[MultiCardEmbedding, BatchedSingleCardEmbedding]:
        results: MultiCardEmbedding = []
        for card in x:
            results.append(self.internal_model(card.embedding))
        return results

    def forward_multi_group(
        self, x: Union[MultiGroupInput, BatchedMultiCardInput]
    ) -> Union[MultiGroupEmbedding, BatchedMultiCardEmbedding]:
        results: MultiGroupEmbedding = []
        for group in x:
            results.append(self.forward_multi_card(group))
        return results

    def forward_batched_multi_group(
        self, x: BatchedMultiGroupInput
    ) -> BatchedMultiGroupEmbedding:
        results: BatchedMultiGroupEmbedding = []
        for multi_group in x:
            results.append(self.forward_multi_group(multi_group))
        return results
    # endregion Forward Methods


class MultiCardModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Internal model is exected to take in a list of GenericCards and return
        # a list of embeddings one per input card in the list
        # (similar to BERT's forward pass)
        internal_model: nn.Module = None  # TODO: Implement this

    def forward(
        self,
        x: Union[SingleCardInput, MultiCardInput, MultiGroupInput, BatchedMultiGroupInput]
    ) -> Union[SingleCardEmbedding, MultiCardEmbedding, MultiGroupEmbedding, BatchedMultiGroupEmbedding]:
        if isinstance(x, GenericCard):
            return self.forward_single_card(x)
        elif isinstance(x, MultiCardInput) or isinstance(x, BatchedSingleCardInput):
            return self.forward_multi_card(x)
        elif isinstance(x, MultiGroupInput) or isinstance(x, BatchedMultiCardInput):
            return self.forward_multi_group(x)
        elif isinstance(x, BatchedMultiGroupInput):
            return self.forward_batched_multi_group(x)
        else:
            raise ValueError(f"Unsupported input type: {type(x)} for MultiCardModel ")

    # region Forward Methods

    def forward_single_card(self, x: SingleCardInput) -> SingleCardEmbedding:
        # Single card in, single embedding out
        # For this multi-card model, it only accepts a list of cards and outputs
        # a list of embeddings (one per card). So to support the single card case,
        # we wrap the single card in a list and pass it to the internal model,
        # then return the first (and only) embedding in the list
        model_out: List[torch.Tensor] = [self.internal_model([x])]
        return model_out[0]

    def forward_multi_card(self, x: Union[MultiCardInput, BatchedSingleCardInput]) -> Union[MultiCardEmbedding, BatchedSingleCardEmbedding]:
        return self.internal_model(x)

    def forward_multi_group(self, x: Union[MultiGroupInput, BatchedMultiCardInput]) -> Union[MultiGroupEmbedding, BatchedMultiCardEmbedding]:
        results: MultiGroupEmbedding = []
        for group in x:
            results.append(self.forward_multi_card(group))
        return results

    def forward_batched_multi_group(self, x: BatchedMultiGroupInput) -> BatchedMultiGroupEmbedding:
        results: BatchedMultiGroupEmbedding = []
        for multi_group in x:
            results.append(self.forward_multi_group(multi_group))
        return results

    # endregion Forward Methods

# endregion Models


class DemoTrainingLoop:
    def __init__(self):
        pass

    def example_training_usage(self):
        model: SingleCardModel = SingleCardModel()  # example
        dojo = NocabDojo()  # example
        dojo.prepare_splits(shuffle=True, rng_seed=42)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # example

        # Train the model
        best_test_loss = float('inf')
        for epoch in range(10):
            dojo.reset_split(split=Split.TRAIN, shuffle=True)
            for batch in dojo.next_batch(Split.TRAIN, batch_size=128):
                # Pass the inputs to the model to generate embeddings
                # Provide those embeddings to the dojo to generate the loss
                embeddings = model.forward(batch.inputs)
                loss = dojo.compute_loss(embeddings, batch.labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            # Test the model
            total_test_loss = 0.0
            total_test_samples = 0
            for (inputs, labels) in dojo.next_batch(Split.TEST, batch_size=128):
                # TODO: Think about how to signal/ produce the full batch without
                # looping over the split/ duplicating elements in the split.
                with torch.no_grad():
                    embeddings = model.forward(inputs)
                    loss = dojo.compute_loss(embeddings, labels)
                total_test_loss += loss.item()
                total_test_samples += len(inputs)
            print(f"Average Test loss: {total_test_loss / total_test_samples}")
            if total_test_loss < best_test_loss:
                best_test_loss = total_test_loss
                # Save the model
        # End of all epochs

        # validate the model
        for (inputs, labels) in dojo.next_batch(Split.VALIDATION, batch_size=128):
            with torch.no_grad():
                embeddings = model.forward(inputs)
                loss = dojo.compute_loss(embeddings, labels)
            total_validation_loss += loss.item()
            total_validation_samples += len(inputs)
        print(f"Average Validation loss: {total_validation_loss / total_validation_samples}")
