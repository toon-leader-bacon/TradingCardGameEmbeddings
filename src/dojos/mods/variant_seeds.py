"""Fixed seeds for bounded mod variants.

A card-field mod built with `variants=N` (N > 0) gives each card at most
N modified texts. Per card, the mod's own run-seeded random state picks
an index v in [0, N); the card is then modified with a fresh
random.Random seeded from VARIANT_SEEDS[v], the card's nocab_uuid and the
mod class. That random state is new for every card and never carried
over, so a card with a given v always gets the same text, in every dojo
and every run, and the feature cache can keep it.

The card's uuid is mixed in so variant v is not the same draw on every
card (the same table row, the same key position across the corpus). The
mod class is mixed in so two chained mods with the same v do not make
the same draws.

Never edit or reorder VARIANT_SEEDS: every cached modded text depends on
it. Generated once with secrets.randbits(64), 2026-10-24.
"""

import random
import zlib
from uuid import UUID

# Most variants a mod may ask for: one seed each
VARIANT_SEEDS: tuple[int, ...] = (
    0xEAFCC06135115B86,
    0x54DE09D58C528AD0,
    0xBDFB5FADEFF9AB64,
    0x85C34CD01799819A,
    0xA9C49B47C0C84991,
    0xDC81E5F4A87AF366,
    0xDF9CF5C363C088DD,
    0xFF79CFA1D9C1B034,
    0x21BA1CD8DDC02A2A,
    0xA78B0F2B4334BA60,
    0x4DF57B5B69E6F7CE,
    0xD9B61BDF9BC9E126,
    0xCD8B6B8BF0E7F8DF,
    0xFA952A65537C3822,
    0x5C45273EDB13E2DD,
    0x4635A01DA3ADEFF3,
    0x5935C590749C7580,
    0xE00397E2C396072E,
    0xBC19FF2EBF2ACFD6,
    0xDFA86B0A4464CD14,
    0xD594B9313776243C,
    0x07A295A915B14D74,
    0x2E80CDC6893BC17B,
    0xCB492E6DEA200C59,
    0x8BE9D305FC7AA7EA,
    0x0D01EB4C1849658D,
    0x57AC281AF1210F03,
    0x00C600A3E0D207AA,
    0xB6A128E7DEB1BDC2,
    0x12CE86270179218F,
    0xD86124FA81790ECA,
    0x00D5475160AEADDB,
    0xF7743128C1EC5271,
    0x59DA6A3908342535,
    0x29A56F589955577D,
    0x8A6BF7D991614D3A,
    0x2FC3F036536AEBB3,
    0x4946BF2CCC9AF1A1,
    0x0FD5FC6C8A102C2C,
    0x2F113E10DF1FCBD7,
    0x57FF64ECF5CDB865,
    0xD2BABC253C318DB7,
    0xF5B4D1F963F2B3B2,
    0xB920C655423A8CCC,
    0xEDBF3A3204690DD9,
    0x5E1AC31427CFAF69,
    0xBE02937500AEEEFE,
    0x36C8680696727664,
    0x48A76C1E212EDB0B,
    0x63B470A70F6587CB,
    0x186F1F81D0090D25,
    0x4623875BD3FC7DDA,
    0xD92DFA94A3E9B9B2,
    0xDAB06421C249DBF7,
    0xD62732BDE2EC9F2F,
    0x780DC76147062341,
    0xC4EA66D78B6CA5BA,
    0x12B083078D72C0DD,
    0x78C867FC1ECAF4D1,
    0xCB48FCFFE326597A,
    0x2B3E6E4B7731FEF3,
    0x41BBCC9CA16B5536,
    0xCDD23E532BF7EB5B,
    0x28063A3E0B0179E3,
    0xE5C600FE27C90230,
    0x39E739A30DE177B4,
    0x32106D61440ED29F,
    0xBBC425C77CE50A5C,
    0xE7BACBF35FB64F5D,
    0xB64DC7E9E4097759,
    0x5BD96E10201599AC,
    0x18A42879697E3BA1,
    0x6DFF67DFDAEFCA81,
    0xB3837B9070CB6759,
    0x49351AA20AE94F40,
    0x45E9B2A45E5141DD,
    0xFEB08882BF67FAAA,
    0x35EE46EDC5DB3376,
    0x6210C92A41250ACD,
    0xAD4CF92E3B271F0D,
    0x3877946E729C48D9,
    0x63AE1FD303B2798F,
    0xB1AFDFC51456D4BF,
    0xCC4AAEE0549B805E,
    0x5EB42DB7DA840719,
    0xF7C0E6CFF7C6A89B,
    0xB9FF2322CF00EC9D,
    0x7024306C6D61BF07,
    0xDFE54CA5BC2496BB,
    0x319A90316878177D,
    0x840639F82ED59500,
    0xEBDCD9F2DE87A8DF,
    0xA0B17E6626715420,
    0xF21F6A5F9D683E8D,
    0xE5428AC84CEB3272,
    0x5AD49188D844A75D,
    0xA617BB4387748B50,
    0x538309D6AEF1F5C8,
    0x31CACA09E0290171,
    0xCF845A2AAE619C33,
)

MAX_VARIANTS = len(VARIANT_SEEDS)


def mod_salt(mod_class: type) -> int:
    """A fixed 32-bit number for a mod class, from its name (zlib.crc32,
    stable across processes, unlike hash()).

    Inputs: mod_class. Output: int in [0, 2**32).
    Side effects: none. Exceptions: none.

    Example:
        >>> mod_salt(RandomKeyMaskMod) == mod_salt(RandomKeyMaskMod)
        True
    """
    return zlib.crc32(mod_class.__name__.encode("utf-8"))


def card_variant_rng(variant: int, card_uuid: UUID, salt: int) -> random.Random:
    """A fresh random state for one variant of one card under one mod: the
    same inputs always give the same sequence of draws.

    Inputs: variant (index into VARIANT_SEEDS, in [0, MAX_VARIANTS)),
        card_uuid (the card's nocab_uuid), salt (mod_salt of the mod).
    Output: random.Random.
    Side effects: none.
    Exceptions: IndexError if variant is outside [0, MAX_VARIANTS).

    Example:
        >>> card_variant_rng(0, card.nocab_uuid, 7).random() == card_variant_rng(
        ...     0, card.nocab_uuid, 7).random()
        True
    """
    if not 0 <= variant < MAX_VARIANTS:
        raise IndexError(f"variant {variant} is outside [0, {MAX_VARIANTS})")
    # Disjoint bit ranges: uuid (128 bits), salt (32), seed (64)
    seed = (VARIANT_SEEDS[variant] << 160) | (salt << 128) | card_uuid.int
    return random.Random(seed)
