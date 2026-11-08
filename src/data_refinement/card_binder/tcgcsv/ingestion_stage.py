"""The tcgcsv games' card stages: the tcgjson pipeline
(card_binder/tcgjson/ingestion_stage.py) over tcgcsv catalogs, one
subclass per game naming its spec (tcgcsv/game_spec.py). The spec's
catalog_format (TCGCSV_FORMAT) makes the base read <slug>.json.gz with
read_tcgcsv_catalog and register product ids as DataSource.TCGCSV
aliases; everything else (identity, voting, the binder mirroring the
catalog) is the base's.
"""

from src.data_refinement.card_binder.tcgcsv.game_spec import (
    BATTLE_SPIRITS_SAGA_SPEC,
    DRAGON_BALL_FUSION_WORLD_SPEC,
    DRAGON_BALL_MASTERS_SPEC,
    GUNDAM_SPEC,
)
from src.data_refinement.card_binder.tcgjson.ingestion_stage import (
    TcgjsonCardIngestionStage,
)


class GundamTcgcsvCardIngestionStage(TcgjsonCardIngestionStage):
    """The Gundam Card Game's cards from its tcgcsv catalog (GUNDAM_SPEC)."""

    SPEC = GUNDAM_SPEC


class DragonBallFusionWorldTcgcsvCardIngestionStage(TcgjsonCardIngestionStage):
    """Dragon Ball Super Fusion World's cards from its tcgcsv catalog
    (DRAGON_BALL_FUSION_WORLD_SPEC)."""

    SPEC = DRAGON_BALL_FUSION_WORLD_SPEC


class DragonBallMastersTcgcsvCardIngestionStage(TcgjsonCardIngestionStage):
    """Dragon Ball Super Masters' cards from its tcgcsv catalog
    (DRAGON_BALL_MASTERS_SPEC)."""

    SPEC = DRAGON_BALL_MASTERS_SPEC


class BattleSpiritsSagaTcgcsvCardIngestionStage(TcgjsonCardIngestionStage):
    """Battle Spirits Saga's cards from its tcgcsv catalog
    (BATTLE_SPIRITS_SAGA_SPEC)."""

    SPEC = BATTLE_SPIRITS_SAGA_SPEC
