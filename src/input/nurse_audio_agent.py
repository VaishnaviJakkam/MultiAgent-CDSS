import re
from threading import Lock
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
        self.model: WhisperModel | None = None
        self._transcription_lock = Lock()

    def transcribe(self, audio_path: str) -> str:
        with self._transcription_lock:
            if self.model is None:
                self.model = WhisperModel(
                    "small",
                    device="cpu",
                    compute_type="int8",
                )
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

    @staticmethod
    def _normalize_spoken_numbers(text: str) -> str:
        """Normalize common dictated vital-number phrases without inventing values."""
        replacements = {
            "one hundred and twenty": "120",
            "one hundred twenty": "120",
            "one twenty": "120",
            "one hundred and ten": "110",
            "one hundred ten": "110",
            "one ten": "110",
            "one hundred": "100",
            "ninety": "90",
            "eighty": "80",
            "seventy": "70",
            "sixty": "60",
            "fifty": "50",
            "forty": "40",
            "thirty": "30",
            "twenty": "20",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "point": ".",
        }
        normalized = text.lower()
        for phrase, number in sorted(replacements.items(), key=lambda item: -len(item[0])):
            normalized = re.sub(rf"\b{re.escape(phrase)}\b", number, normalized)
        return normalized

    @staticmethod
    def _extract_wbc(text: str) -> Optional[float]:
        """Extract one WBC result from numeric or naturally spoken numeric text."""
        spoken_decimal = re.search(r"(?<![\w.])(\d+)\s*\.\s*(\d+)(?![\w.])", text)
        if spoken_decimal:
            return float(f"{spoken_decimal.group(1)}.{spoken_decimal.group(2)}")

        numeric = re.search(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text)
        if numeric:
            return float(numeric.group(0))

        return None

    @staticmethod
    def _extract_lactate(text: str) -> Optional[float]:
        """Extract one lactate result from numeric or naturally spoken numeric text."""
        lactate_context = re.search(
            r"(?:lactate|lactic\s+acid)(?:\s+(?:result|level|value))?\s*(?:is|of|:)?\s*"
            r"(\d+(?:\s*\.\s*\d+)?)",
            text,
            flags=re.IGNORECASE,
        )
        if lactate_context:
            return float(re.sub(r"\s+", "", lactate_context.group(1)))

        spoken_decimal = re.search(r"(?<![\w.])(\d+)\s*\.\s*(\d+)(?![\w.])", text)
        if spoken_decimal:
            return float(f"{spoken_decimal.group(1)}.{spoken_decimal.group(2)}")

        numeric = re.search(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", text)
        if numeric:
            return float(numeric.group(0))

        return None

    def extract_parameters(
        self,
        transcript: str,
    ) -> dict:

        text = self._normalize_spoken_numbers(transcript)

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
        # WHITE BLOOD CELL COUNT
        # -------------------------

        wbc = (
            self._extract_wbc(text)
            if re.search(r"(?:\bwbc\b|white\s+blood\s+cell)", text, flags=re.IGNORECASE)
            else None
        )
        if wbc is not None:
            parameters["WBC"] = wbc

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

        lactate = (
            self._extract_lactate(text)
            if re.search(r"\b(?:lactate|lactic\s+acid)\b", text, flags=re.IGNORECASE)
            else None
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

        no_value = bool(
            re.search(
                r"\b(?:none|no\s+value|not\s+available|unavailable|unknown)\b",
                transcript,
                flags=re.IGNORECASE,
            )
        )

        return {
            "status": "success",
            "source": "nurse_audio",
            "transcript": transcript,
            "extracted_parameters":
                extraction["parameters"],
            "metadata":
                extraction["metadata"],
            "no_value": no_value,
            "requires_confirmation": True,
        }