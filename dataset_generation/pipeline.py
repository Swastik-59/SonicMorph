import logging
from .config import config
from .database import init_db, get_db_path
from .logging_setup import setup_logging

logger = logging.getLogger(__name__)

STAGES = [
    "s01_artist_discovery",
    "s02_audio_collection",
    "s03_audio_validation",
    "s04_duplicate_detection",
    "s05_audio_normalization",
    "s06_stem_separation",
    "s07_metadata_extraction",
    "s08_feature_extraction",
    "s09_dataset_packaging",
    "s10_manifest_generation",
    "s11_quality_reporting",
]


class Pipeline:
    def __init__(self, log_level: str = "INFO"):
        setup_logging(log_level)
        self.config = config
        self.db_path = init_db()

    def run_stage(self, stage_name: str) -> bool:
        logger.info("Running stage: %s", stage_name)
        try:
            # Jobs are treated as execution logs only; stage state is driven by song status columns.
            from .jobs import create_job, start_job, complete_job

            job_id = create_job(stage_name)
            start_job(job_id)

            mod = __import__(f"sonicmorph.stages.{stage_name}", fromlist=["run"])
            ok = mod.run(self.config, db_conn=None)
            complete_job(job_id, success=ok)
            return ok
        except Exception as exc:  # noqa: BLE001
            logger.exception("Stage %s failed: %s", stage_name, exc)
            return False

    def run_all(self):
        results = {}
        for s in STAGES:
            ok = self.run_stage(s)
            results[s] = ok
            if not ok:
                logger.warning("Stage %s reported failure, stopping pipeline", s)
                break
        return results


if __name__ == "__main__":
    p = Pipeline()
    res = p.run_all()
    print(res)
