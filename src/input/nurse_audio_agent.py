import re
from typing import Optional

from faster_whisper import WhisperModel


class NurseAudioAgent:
    """
    Converts nurse audio into structured physical parameters
    required by the Sepsis assessment pipeline.

    Speech recognition:
        faster-whisper

    Important:
        Extracted values should be confirmed by the nurse
        before being stored or sent to the clinical models.
    """

    def __init__(self):
        # CPU-friendly model for development/demo.
        # Download occurs automatically on first use.
        self.model = WhisperModel(
            "small",
            device="cpu",
            compute_type="int8",
        )

    def transcribe(self, audio_path: str) -> str:
        segments, _ = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=5,
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return text.strip()

    @staticmethod
    def _find_number(
        text: str,
        patterns: list[str],
    ) -> Optional[float]:

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return float(match.group(1))

        return None

    def extract_parameters(
        self,
        transcript: str,
    ) -> dict:

        text = transcript.lower()

        parameters = {}
        metadata = {}

        # -------------------------
        # HEART RATE
        # -------------------------

        hr = self._find_number(
            text,
            [
                r"heart\s*rate\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"\bhr\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"pulse\s*(?:rate)?\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ],
        )

        if hr is not None:
            parameters["HR"] = hr

        # -------------------------
        # OXYGEN SATURATION
        # -------------------------

        o2sat = self._find_number(
            text,
            [
                r"oxygen\s*saturation\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"spo2\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"o2\s*sat(?:uration)?\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ],
        )

        if o2sat is not None:
            parameters["O2Sat"] = o2sat

        # -------------------------
        # TEMPERATURE
        # -------------------------

        temp = self._find_number(
            text,
            [
                r"temperature\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"\btemp\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ],
        )

        if temp is not None:
            parameters["Temp"] = temp

        # -------------------------
        # BLOOD PRESSURE
        # -------------------------

        bp_patterns = [
            r"blood\s*pressure\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)\s*(?:over|/)\s*(\d+(?:\.\d+)?)",
            r"\bbp\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)\s*(?:over|/)\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in bp_patterns:
            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                sbp = float(match.group(1))
                dbp = float(match.group(2))

                parameters["SBP"] = sbp

                # DBP isn't currently an input to our
                # Sepsis LSTM, but retain it as useful
                # clinical information.
                parameters["DBP"] = dbp

                # Approximate MAP derived from SBP/DBP.
                map_value = (
                    sbp + (2 * dbp)
                ) / 3

                parameters["MAP"] = round(
                    map_value,
                    2,
                )

                metadata["MAP"] = {
                    "source": "derived",
                    "formula":
                        "(SBP + 2*DBP) / 3",
                }

                break

        # -------------------------
        # RESPIRATORY RATE
        # -------------------------

        resp = self._find_number(
            text,
            [
                r"respiratory\s*rate\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"respiration\s*(?:rate)?\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"\bresp\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ],
        )

        if resp is not None:
            parameters["Resp"] = resp

        # -------------------------
        # LACTATE
        # -------------------------

        lactate = self._find_number(
            text,
            [
                r"lactate\s*(?:result|level)?\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
                r"lactic\s*acid\s*(?:is|of|:)?\s*(\d+(?:\.\d+)?)",
            ],
        )

        if lactate is not None:
            parameters["Lactate"] = lactate

        return {
            "parameters": parameters,
            "metadata": metadata,
        }

    def analyze(
        self,
        audio_path: str,
    ) -> dict:

        transcript = self.transcribe(
            audio_path
        )

        extraction = self.extract_parameters(
            transcript
        )

        return {
            "status": "success",
            "source": "nurse_audio",
            "transcript": transcript,
            "extracted_parameters":
                extraction["parameters"],
            "metadata":
                extraction["metadata"],
            "requires_confirmation": True,
        }