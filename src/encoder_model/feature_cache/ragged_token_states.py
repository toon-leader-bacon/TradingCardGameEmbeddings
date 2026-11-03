"""RaggedTokenStates: the token states of several texts, packed without
padding.

The feature type of a token-level feature cache. One text's tokens are
consecutive rows of `values`; `lengths` says how many rows each text
has. Padding is dropped on the way in (from_encoding) and rebuilt on the
way out (to_encoding), so a stored row holds only real tokens and costs
storage in proportion to its text's length.
"""

from dataclasses import dataclass
from typing import Sequence

import torch

from src.encoder_model.text_encoder import TokenEncoding


@dataclass(frozen=True)
class RaggedTokenStates:
    """Several texts' token states, packed.

    values: (total_tokens, width), text i's tokens at rows
        offsets[i]:offsets[i + 1], in token order.
    lengths: (texts,) int64 on the CPU, each >= 1 and summing to
        total_tokens. On the CPU because slicing and slab bookkeeping read
        it on the host; reading it never syncs the device.
    """

    values: torch.Tensor
    lengths: torch.Tensor

    def __post_init__(self) -> None:
        """Inputs: none (checks self). Output: none. Side effects: none.
        Exceptions: ValueError if values is not 2-D, lengths is not a 1-D
            int64 CPU tensor, a length is < 1, or the lengths do not sum
            to values' row count."""
        if self.values.dim() != 2:
            raise ValueError(f"values must be 2-D, got {tuple(self.values.shape)}")
        lengths = self.lengths
        if (
            lengths.dim() != 1
            or lengths.dtype != torch.int64
            or lengths.device.type != "cpu"
        ):
            raise ValueError("lengths must be a 1-D int64 CPU tensor")
        if lengths.numel() and int(lengths.min()) < 1:
            raise ValueError("every text needs at least one token")
        if int(lengths.sum()) != self.values.shape[0]:
            raise ValueError(
                f"lengths sum to {int(lengths.sum())}, values has "
                f"{self.values.shape[0]} rows"
            )

    @property
    def text_count(self) -> int:
        """How many texts. Inputs: none. Output: int. Side effects: none.
        Exceptions: none."""
        return int(self.lengths.shape[0])

    @property
    def width(self) -> int:
        """Each token's width. Inputs: none. Output: int. Side effects:
        none. Exceptions: none."""
        return int(self.values.shape[1])

    @classmethod
    def from_encoding(cls, encoding: TokenEncoding) -> "RaggedTokenStates":
        """Pack a padded encoding: keep each text's real tokens (mask 1),
        in order. Right padding is assumed, as the tokenizer produces it.

        Inputs: encoding (hidden_states (N, L, W), attention_mask (N, L)).
        Output: RaggedTokenStates with N texts, values on encoding's
            device and dtype.
        Side effects: one device -> host copy of the mask's row sums.
        Exceptions: ValueError if a text has no real token.

        Example:
            >>> RaggedTokenStates.from_encoding(encoder.encode_through(texts, 20)).text_count
            2
        """
        real = encoding.attention_mask.bool()
        # Count each text's real tokens on the host (one small copy)
        lengths = real.sum(dim=1).to("cpu", torch.int64)
        # Row-major boolean indexing keeps each text's tokens in order
        values = encoding.hidden_states[real]
        return cls(values, lengths)

    def to_encoding(self) -> TokenEncoding:
        """Pad back to (N, max_length, W), with a mask marking each text's
        real tokens; padding rows are zero.

        Inputs: none. Output: TokenEncoding on values' device and dtype;
            the mask is int64, like the tokenizer's.
        Side effects: none. Exceptions: none.

        Example:
            >>> states.to_encoding().hidden_states.shape
            torch.Size([2, 37, 768])
        """
        device = self.values.device
        longest = int(self.lengths.max()) if self.text_count else 0
        # Build the (N, longest) mask from lengths on the host, then move it
        ramp = torch.arange(longest).unsqueeze(0)
        real = (ramp < self.lengths.unsqueeze(1)).to(device)
        hidden_states = self.values.new_zeros(self.text_count, longest, self.width)
        # One masked assignment: row-major order matches the packing
        hidden_states[real] = self.values
        return TokenEncoding(hidden_states=hidden_states, attention_mask=real.long())

    def select(self, positions: Sequence[int]) -> "RaggedTokenStates":
        """The given texts, in the given order (repeats allowed).

        Inputs: positions (each in [0, text_count)).
        Output: RaggedTokenStates on values' device.
        Side effects: none. Exceptions: IndexError for a position out of
            range.

        Example:
            >>> states.select([1, 0]).lengths.tolist()
            [37, 12]
        """
        rows = token_row_ranges(self.lengths, positions)
        index = torch.as_tensor(positions, dtype=torch.int64)
        values = self.values.index_select(0, rows.to(self.values.device))
        return RaggedTokenStates(values, self.lengths.index_select(0, index))

    def to(self, device: torch.device, dtype: torch.dtype) -> "RaggedTokenStates":
        """values moved and cast; lengths stay on the CPU.

        Inputs: device, dtype. Output: RaggedTokenStates (self when nothing
            changes). Side effects: none. Exceptions: none.

        Example:
            >>> states.to(torch.device("cpu"), torch.float16).values.dtype
            torch.float16
        """
        values = self.values.to(device, dtype)
        if values is self.values:
            return self
        return RaggedTokenStates(values, self.lengths)

    @staticmethod
    def concatenate(parts: Sequence["RaggedTokenStates"]) -> "RaggedTokenStates":
        """parts' texts, one after another.

        Inputs: parts (non-empty; same width, device and dtype).
        Output: RaggedTokenStates. Side effects: none.
        Exceptions: ValueError for no parts.

        Example:
            >>> RaggedTokenStates.concatenate([a, b]).text_count == a.text_count + b.text_count
            True
        """
        if not parts:
            raise ValueError("concatenate needs at least one part")
        if len(parts) == 1:
            return parts[0]
        values = torch.cat([part.values for part in parts])
        lengths = torch.cat([part.lengths for part in parts])
        return RaggedTokenStates(values, lengths)


def token_row_ranges(lengths: torch.Tensor, positions: Sequence[int]) -> torch.Tensor:
    """The rows of `values` that the given texts occupy, concatenated in
    positions' order: an int64 CPU index for one index_select.

    Consumer: RaggedTokenStates.select (public for its tests).
    Inputs: lengths ((texts,) int64 CPU), positions (indices into it).
    Output: (sum of the selected lengths,) int64 CPU tensor.
    Side effects: none. Exceptions: IndexError for a position out of
        range.

    Example:
        >>> token_row_ranges(torch.tensor([2, 3]), [1, 0]).tolist()
        [2, 3, 4, 0, 1]
    """
    index = torch.as_tensor(positions, dtype=torch.int64)
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= len(lengths)):
        raise IndexError(f"a position is outside [0, {len(lengths)})")
    starts = torch.cumsum(lengths, 0) - lengths
    picked_lengths = lengths.index_select(0, index)
    picked_starts = starts.index_select(0, index)
    # Each text's rows: its start repeated over its length, plus a ramp
    # that restarts at 0 for every text
    total = int(picked_lengths.sum())
    repeated_starts = torch.repeat_interleave(picked_starts, picked_lengths)
    block_starts = torch.cumsum(picked_lengths, 0) - picked_lengths
    ramp = torch.arange(total) - torch.repeat_interleave(block_starts, picked_lengths)
    return repeated_starts + ramp
