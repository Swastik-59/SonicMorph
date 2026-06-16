from .base_collector import BaseCollector

from pathlib import Path
from typing import Iterable, Dict, Any

import json
import logging
import subprocess

logger = logging.getLogger(__name__)


class YouTubeCollector(BaseCollector):

    def discover(self, artist: str) -> Iterable[Dict[str, Any]]:

        logger.info(
            "Searching YouTube for artist: %s",
            artist,
        )

        search_count = 50

        cmd = [
            "yt-dlp",
            "--dump-json",
            f"ytsearch{search_count}:{artist} official audio",
        ]

        try:

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

        except subprocess.CalledProcessError as exc:

            logger.warning(
                "YouTube discovery failed for %s:\n%s",
                artist,
                exc.stderr,
            )

            return

        seen_ids = set()

        for line in result.stdout.splitlines():

            try:

                item = json.loads(line)

            except Exception:
                continue

            video_id = item.get("id")

            if not video_id:
                continue

            if video_id in seen_ids:
                continue

            seen_ids.add(video_id)

            title = (
                item.get("title", "")
                .lower()
            )

            duration = item.get("duration")

            #
            # basic filtering
            #

            bad_keywords = [
                "reaction",
                "karaoke",
                "instrumental",
                "nightcore",
                "8d audio",
                "slowed",
                "reverb",
                "sped up",
                "cover by",
                "tribute",
            ]

            if any(
                k in title
                for k in bad_keywords
            ):
                continue

            if duration is not None:

                if duration < 60:
                    continue

                if duration > 900:
                    continue

            yield {
                "type": "youtube_video",
                "artist": artist,
                "video_id": video_id,
                "title": item.get("title"),
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "duration": duration,
            }

    def download(
        self,
        candidate: Dict[str, Any],
        out_dir: Path,
    ) -> Path:

        if candidate.get("type") != "youtube_video":

            raise NotImplementedError(
                "Only youtube_video candidates supported"
            )

        out_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        safe_name = (
            candidate.get(
                "artist",
                "unknown",
            )
            .replace("/", "_")
            .replace("\\", "_")
        )

        out_template = str(
            out_dir
            / f"{safe_name}-%(title)s-%(id)s.%(ext)s"
        )

        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--extract-audio",
            "--audio-format",
            "mp3",
            "--output",
            out_template,
            candidate["url"],
        ]

        logger.info(
            "Downloading: %s",
            candidate.get(
                "title",
                candidate["url"],
            ),
        )

        try:

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

        except subprocess.CalledProcessError as exc:

            logger.warning(
                "yt-dlp failed:\n%s",
                exc.stderr,
            )

            raise

        files = sorted(
            out_dir.glob(
                f"{safe_name}-*"
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        if not files:

            raise FileNotFoundError(
                "yt-dlp completed but produced no file"
            )

        logger.info(
            "Downloaded file: %s",
            files[0],
        )

        return files[0]
