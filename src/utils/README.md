# utils

General-purpose helpers ported from the author's other libraries. Treat
this directory like a vendored external library: nothing here imports
from the rest of `src/` or knows about cards, dojos or training.

- [drop_table.py](drop_table.py): `DropTable[T]`, a weighted random choice
  over outcomes, ported from nocabLib's `NocabRNG/DropTable.cs`. Tables
  are immutable, their weights are validated (finite, >= 0, positive
  total), and the random source is passed to `pull(rng)` so one table can
  be shared. An entry's outcome may be another `DropTable`, which `pull`
  rolls on in turn. `filtered(keep)` returns a new table holding only the
  outcomes `keep` accepts (emptied sub-tables drop out; `None` if nothing
  pullable survives), so relative odds hold over what remains. Used by
  `src/dojos/mods/card_field_mods.py`'s `WeightedFieldMaskMod`.
