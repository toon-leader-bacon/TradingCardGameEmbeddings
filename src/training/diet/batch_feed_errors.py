"""The errors a BatchFeed raises about its own lifecycle.

Its own module because batch_feed.py imports prefetching_batch_streams.py,
and both raise these. A BatchFeedError is never a dojo's fault: the
trainer ends the run instead of counting it against the dojo it asked
for. A dojo's own error, raised while building its batch, keeps its own
type.
"""


class BatchFeedError(RuntimeError):
    """A feed was used out of order or can no longer produce batches."""


class ScheduleMismatchError(BatchFeedError):
    """The trainer and the feed disagree about the round: next_batch for a
    dojo that is not the next scheduled one, next_batch past the
    schedule's end or with no round open, or begin_round before the
    previous round ended."""


class PrefetchStoppedError(BatchFeedError):
    """next_batch found nothing built and a worker that will not build
    more (never started, closed, or dead), or begin_round found the worker
    dead."""
