"""Dojo: regress a card's embedding against a numeric 17lands draft metric.

See plans/metric_regression_dojo.md and src/dojos/dojo.py for the Dojo
contract this implements. Generic over which numeric metric it reads —
metric_name/metrics_path are constructor parameters, not hardcoded —
so this one class is instantiated once per metric (e.g.
average_pick_number, pick_sideboard_rate:
src/data_refinement/seventeenlands/draft_data_metrics/metrics/) rather
than duplicated per metric; the two target metrics differ only by
which metric_name to read, not by any different logic this dojo would
need to special-case.

Deliberately does NOT share a base class with GameClassificationDojo
for their near-identical split/cursor mechanics — this project has a
standing "rule of three, not preemptive" convention for exactly this
kind of two-instance similarity (see e.g. average_pick_number.py's own
docstring); this dojo follows that same precedent rather than opening
a new one.
"""

import random
from pathlib import Path
from typing import cast
from uuid import UUID

import pandas as pd
import torch

from src.data_refinement.card_binder.card_lookup import CardLookup
from src.dojos.dojo import CardCount, CardSet, DecoderHead, Dojo, DojoBatch, Split
from src.dojos.losses.loss import Loss
from src.encoder_model.embedding import Embedding


class MetricRegressionDojo(Dojo[float]):
    """Predicts a single card's numeric metric value from its own embedding.

    CardCount(1, 1) — single-card, same arity as GameClassificationDojo.
    Labels are plain floats, read once (in prepare_splits) from a
    MetricResult parquet file (src/data_refinement/seventeenlands/
    metric_result.py, written by write_metric_results) and filtered to
    metric_name — a metrics file may hold more than one metric's rows,
    see this class's own prepare_splits docstring.
    """

    def __init__(
        self,
        metrics_path: Path,
        metric_name: str,
        train_ratio: float,
        validate_ratio: float,
        loss: Loss[torch.Tensor, float],
        rng_seed: int,
    ) -> None:
        """Construct a dojo that regresses against one named metric's values.

        Inputs:
            metrics_path: path to a MetricResult parquet file (e.g.
                data/final/metrics/17lands/draft/MSH.PremierDraft.parquet
                — the same output_path a DraftMetricScanner run was
                configured with).
            metric_name: which metric's rows to read from metrics_path
                (e.g. "average_pick_number") — rows with any other
                metric_name are ignored.
            train_ratio: fraction of eligible cards assigned to the
                train pool.
            validate_ratio: fraction of eligible cards assigned to the
                validate pool. The remainder goes to test.
            loss: the injected loss computation this dojo delegates to
                from compute_loss (e.g. MSELoss).
            rng_seed: seed for this dojo's own shuffling/sampling, for
                reproducible splits and batches.
        Output: none (constructor).
        Side effects: none — metrics_path is not read until
            prepare_splits is called.
        Exceptions: raises ValueError if train_ratio + validate_ratio
            is not strictly less than 1, or either ratio is not in
            (0, 1).

        Example:
            >>> dojo = MetricRegressionDojo(
            ...     metrics_path=Path("data/final/metrics/17lands/draft/MSH.PremierDraft.parquet"),
            ...     metric_name="average_pick_number",
            ...     train_ratio=0.8,
            ...     validate_ratio=0.1,
            ...     loss=MSELoss(),
            ...     rng_seed=0,
            ... )
        """
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if not (0.0 < validate_ratio < 1.0):
            raise ValueError(f"validate_ratio must be in (0, 1), got {validate_ratio}")
        if train_ratio + validate_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + validate_ratio must be < 1, "
                f"got {train_ratio} + {validate_ratio}"
            )
        self._metrics_path = metrics_path
        self._metric_name = metric_name
        self._train_ratio = train_ratio
        self._validate_ratio = validate_ratio
        self._loss = loss
        self._rng = random.Random(rng_seed)
        self._corpus: CardLookup | None = None
        self._decoder_head: DecoderHead | None = None
        self._labels: dict[UUID, float] = {}
        self._train_uuids: list[UUID] = []
        self._pass_uuids: dict[Split, list[UUID]] = {}
        self._pass_index: dict[Split, int] = {}

    def card_count(self) -> CardCount:
        """Output: CardCount(1, 1) — this dojo is single-card-only."""
        return CardCount(minimum=1, maximum=1)

    def prepare_splits(self, corpus: CardLookup, held_out_cards: set[UUID]) -> None:
        """Load this dojo's metric values and split their cards by configured ratios.

        Reads self._metrics_path via pandas.read_parquet, filters rows
        to metric_name == self._metric_name, and builds
        self._labels: dict[UUID, float] from the filtered rows
        (UUID(row["nocab_uuid"]) -> row["value"]). This dict's keys are
        this dojo's ENTIRE example population — not corpus.all_uuids()
        — since a metrics file is typically a strict subset of a full
        card binder (only cards actually drafted in the scanned
        expansion/format have a value). For each key, looks up the
        GenericCard via corpus.get_by_uuid(); a missing card is a
        metrics-file/corpus version mismatch, not an expected case —
        see this method's Exceptions.

        held_out_cards is honored exactly as GameClassificationDojo
        honors it: any uuid in held_out_cards is kept out of the train
        pool. Same shuffle/train-ratio/validate-ratio partition logic
        as GameClassificationDojo.prepare_splits, over this dojo's own
        label-derived key set instead of corpus.all_uuids(). Caches
        corpus, so later next_batch calls can look up cards by uuid
        without needing corpus passed in again.

        Inputs:
            corpus: card lookup to resolve this dojo's metric-derived
                uuids into GenericCards, and to cache for next_batch's
                own later lookups.
            held_out_cards: nocab_uuids to keep out of the train pool.
        Output: none — populates this dojo's own internal label map and
            train/test/validate uuid pools.
        Side effects: reads self._metrics_path; mutates this dojo's own
            internal state.
        Exceptions: raises ValueError if any uuid present in
            self._metrics_path's filtered rows has no card in corpus.
        """
        self._corpus = corpus
        rows = pd.read_parquet(self._metrics_path)
        rows = rows[rows["metric_name"] == self._metric_name]
        # pandas' itertuples() types each field as a broad Any-ish union
        # regardless of the DataFrame's actual dtype; cast rather than
        # str()/float() (which mypy's stubs don't accept for that union)
        # since the real runtime types are already str/float64 (verified
        # against write_metric_results' explicit dtype cast).
        self._labels = {
            UUID(cast(str, row.nocab_uuid)): cast(float, row.value) for row in rows.itertuples()
        }

        all_uuids = list(self._labels.keys())
        for nocab_uuid in all_uuids:
            if corpus.get_by_uuid(nocab_uuid) is None:
                raise ValueError(f"no card found for nocab_uuid {nocab_uuid}")

        self._rng.shuffle(all_uuids)

        eligible_for_train = [u for u in all_uuids if u not in held_out_cards]
        train_count = min(round(len(all_uuids) * self._train_ratio), len(eligible_for_train))
        train_uuids = eligible_for_train[:train_count]

        train_uuid_set = set(train_uuids)
        remaining = [u for u in all_uuids if u not in train_uuid_set]
        self._rng.shuffle(remaining)

        validate_count = round(len(all_uuids) * self._validate_ratio)
        validate_uuids = remaining[:validate_count]
        test_uuids = remaining[validate_count:]

        self._train_uuids = train_uuids
        self._pass_uuids = {Split.TEST: test_uuids, Split.VALIDATE: validate_uuids}
        self._pass_index = {Split.TEST: 0, Split.VALIDATE: 0}

    def split_size(self, split: Split) -> int:
        """Report how many cards are in one of this dojo's pools.

        Inputs:
            split: which pool to size.
        Output: that pool's card count, as of the last prepare_splits
            call.
        Side effects: none.
        Exceptions: raises RuntimeError if called before
            prepare_splits.
        """
        if self._corpus is None:
            raise RuntimeError("prepare_splits must be called before split_size")
        if split == Split.TRAIN:
            return len(self._train_uuids)
        return len(self._pass_uuids[split])

    def next_batch(self, split: Split, batch_size: int) -> DojoBatch[float]:
        """Draw batch_size cards from split, each its own length-1 CardSet.

        See Split's own docstring (src/dojos/dojo.py) for the TRAIN
        vs. TEST/VALIDATE sampling behavior this must honor. Dispatches
        to the same _sample_train_uuids/_next_pass_uuids/_build_batch
        decomposition GameClassificationDojo uses, generalized over
        this dojo's own self._labels map instead of a card field.

        Inputs:
            split: which pool to draw from.
            batch_size: how many examples to draw.
        Output: a DojoBatch of length-1 CardSets, each labeled with
            that card's metric value.
        Side effects: advances this dojo's internal cursor for split.
        Exceptions: raises RuntimeError if called before
            prepare_splits.

        Example:
            >>> dojo = MetricRegressionDojo(
            ...     metrics_path, "average_pick_number", 0.8, 0.1, MSELoss(), rng_seed=0
            ... )
            >>> dojo.prepare_splits(corpus, held_out_cards=set())
            >>> batch = dojo.next_batch(Split.TRAIN, batch_size=4)
        """
        if self._corpus is None:
            raise RuntimeError("prepare_splits must be called before next_batch")
        if split == Split.TRAIN:
            uuids = self._sample_train_uuids(batch_size)
        else:
            uuids = self._next_pass_uuids(split, batch_size)
        return self._build_batch(uuids)

    def build_decoder_head(self, embedding_dim: int) -> DecoderHead:
        """Construct a thin linear regressor: one scalar output per card.

        Caches the returned head on this dojo, so compute_loss can run
        it later.

        Inputs:
            embedding_dim: the encoder's output embedding width.
        Output: a fresh torch.nn.Linear(embedding_dim, 1).
        Side effects: caches the constructed head on this dojo.
        Exceptions: none.
        """
        self._decoder_head = torch.nn.Linear(embedding_dim, 1)
        return self._decoder_head

    def compute_loss(self, embeddings: list[list[Embedding]], labels: list[float]) -> torch.Tensor:
        """Run this dojo's decoder head, then its injected loss, over a batch.

        Same shape validation as GameClassificationDojo.compute_loss.
        Stacks the single embedding per example, runs the cached
        decoder head (output shape (batch_size, 1)), then squeezes it
        to (batch_size,) via .squeeze(dim=1) — NOT a bare .squeeze(),
        which would collapse a short final TEST/VALIDATE batch of size
        1 (Split's docstring permits this) from (1, 1) down to a 0-d
        tensor instead of (1,) — before delegating to self._loss.

        Inputs:
            embeddings: one length-1 list[Embedding] per example.
            labels: this batch's float labels, as returned by
                next_batch, same order.
        Output: a scalar loss tensor for this batch (self._loss's
            output, given this dojo's decoder head's squeezed
            predictions and labels).
        Side effects: none — does not call .backward() or step any
            optimizer.
        Exceptions: raises ValueError if any embeddings[i] does not
            hold exactly one Embedding, or len(embeddings) !=
            len(labels), or build_decoder_head hasn't been called yet.
        """
        if self._decoder_head is None:
            raise ValueError("build_decoder_head must be called before compute_loss")
        if len(embeddings) != len(labels):
            raise ValueError(
                f"embeddings and labels must be the same length, "
                f"got {len(embeddings)} and {len(labels)}"
            )
        for card_embeddings in embeddings:
            if len(card_embeddings) != 1:
                raise ValueError(
                    f"MetricRegressionDojo is CardCount(1, 1); expected "
                    f"exactly one embedding per example, got {len(card_embeddings)}"
                )

        stacked = torch.stack([card_embeddings[0] for card_embeddings in embeddings])
        predictions = self._decoder_head(stacked).squeeze(dim=1)
        return self._loss.compute(predictions, labels)

    def _sample_train_uuids(self, batch_size: int) -> list[UUID]:
        """Sample batch_size uuids from the train pool, with replacement.

        Private helper — single consumer is next_batch() for
        Split.TRAIN. Same mechanics as
        GameClassificationDojo._sample_train_uuids.

        Inputs:
            batch_size: how many uuids to sample.
        Output: batch_size uuids drawn from this dojo's train pool.
        Side effects: consumes from this dojo's own rng state.
        Exceptions: raises IndexError if the train pool is empty.
        """
        return self._rng.choices(self._train_uuids, k=batch_size)

    def _next_pass_uuids(self, split: Split, batch_size: int) -> list[UUID]:
        """Advance split's cursor and return its next (possibly short) chunk.

        Private helper — single consumer is next_batch() for
        Split.TEST/Split.VALIDATE. Same mechanics as
        GameClassificationDojo._next_pass_uuids, including wrapping to
        a freshly-shuffled order once a full pass over split completes.

        Inputs:
            split: TEST or VALIDATE.
            batch_size: how many uuids to return, except possibly
                fewer on the final chunk of a pass.
        Output: the next chunk of uuids for split, length batch_size
            except a pass's final (possibly short) chunk.
        Side effects: advances this dojo's internal cursor for split.
        Exceptions: none.
        """
        pool = self._pass_uuids[split]
        if not pool:
            return []

        index = self._pass_index[split]
        if index == 0:
            self._rng.shuffle(pool)

        end = index + batch_size
        chunk = pool[index:end]
        next_index = index + len(chunk)
        self._pass_index[split] = 0 if next_index >= len(pool) else next_index
        return chunk

    def _build_batch(self, uuids: list[UUID]) -> DojoBatch[float]:
        """Turn a list of uuids into a DojoBatch via this dojo's cached corpus.

        Private helper — single consumer is next_batch(), shared by
        all three Split cases once each has produced its uuid list.
        Unlike GameClassificationDojo._build_batch (which reads a
        card's own source_game field), this looks each uuid's label up
        in self._labels (already resolved in prepare_splits).

        Inputs:
            uuids: uuids to look up via this dojo's cached CardLookup
                and self._labels.
        Output: a DojoBatch of length-1 CardSets, labeled by each
            card's metric value.
        Side effects: none.
        Exceptions: raises ValueError if any uuid has no card in this
            dojo's cached CardLookup (should not happen for a uuid
            this dojo itself produced from self._labels' keys, which
            were already validated in prepare_splits — defensive, not
            an expected path).
        """
        assert self._corpus is not None  # guaranteed by next_batch's own guard
        card_sets = []
        labels = []
        for nocab_uuid in uuids:
            card = self._corpus.get_by_uuid(nocab_uuid)
            if card is None:
                raise ValueError(f"no card found for nocab_uuid {nocab_uuid}")
            card_sets.append(CardSet(cards=[card]))
            labels.append(self._labels[nocab_uuid])
        return DojoBatch(card_sets=card_sets, labels=labels)
