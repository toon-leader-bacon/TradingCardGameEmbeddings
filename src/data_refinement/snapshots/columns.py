"""The snapshots' column names and types, shared by the builders, the
readers and the per-run caches numbered by a snapshot's rows."""

import pyarrow as pa

# Bytes in a nocab_uuid stored as fixed_size_binary
UUID_WIDTH = 16

# Both snapshots: each row's nocab_uuid, fixed_size_binary(UUID_WIDTH)
NOCAB_UUID_COLUMN = "nocab_uuid"

# Card snapshot: true on the game's Unknown sentinel row only
IS_UNKNOWN_COLUMN = "is_unknown"

# Deck snapshot: one (card_index, count) entry per distinct card of a deck
CARDS_COLUMN = "cards"
CARD_INDEX_FIELD = "card_index"
COUNT_FIELD = "count"
CARDS_TYPE = pa.list_(
    pa.struct([(CARD_INDEX_FIELD, pa.uint16()), (COUNT_FIELD, pa.uint8())])
)
