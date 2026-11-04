"""CardTextKey: what the feature cache files a card's features under.

The key is a digest of the serialized text the text encoder reads, never
the card's nocab_uuid: a mod keeps the uuid and changes the text, so only
the text says which features belong to it.
"""

import hashlib

# 16 bytes: collisions are negligible at any card count this project sees
# (well under 2**40 distinct texts), at a fraction of the text's own size
_DIGEST_BYTES = 16


class CardTextKey(bytes):
    """A 16-byte BLAKE2b digest of one serialized card text; the key is
    the digest bytes themselves. Lives on the CPU.

    A bytes subclass rather than a dataclass holding bytes: every store
    lookup hashes and compares keys, and bytes does both in C (a frozen
    dataclass's generated __hash__ and __eq__ run in Python, about 4x
    slower per dict lookup, measured 2026-10-21). It hashes and compares
    equal to the plain bytes of the same digest, so SQLite's BLOBs match
    it directly.
    """

    __slots__ = ()

    def __new__(cls, digest: bytes) -> "CardTextKey":
        """Inputs: digest (bytes, _DIGEST_BYTES long). Output: CardTextKey.
        Side effects: none.
        Exceptions: ValueError if digest is not _DIGEST_BYTES long;
            TypeError if digest is not bytes-like.

        Example:
            >>> CardTextKey(bytes(16)) == bytes(16)
            True
        """
        if len(digest) != _DIGEST_BYTES:
            raise ValueError(f"digest must be {_DIGEST_BYTES} bytes, got {len(digest)}")
        return super().__new__(cls, digest)

    @staticmethod
    def of_text(text: str) -> "CardTextKey":
        """The key of one serialized card text.

        Inputs: text (str), as serialize_card_to_json returns it.
        Output: CardTextKey; equal texts always give equal keys.
        Side effects: none. Exceptions: none.

        Example:
            >>> CardTextKey.of_text('{"name":"Bolt"}') == CardTextKey.of_text('{"name":"Bolt"}')
            True
        """
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=_DIGEST_BYTES)
        # The digest size is fixed above, so skip __new__'s length check
        return bytes.__new__(CardTextKey, digest.digest())
