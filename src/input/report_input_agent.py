import re
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image
from rapidfuzz import fuzz


# Windows Tesseract installation path
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


class ReportInputAgent:

    # Canonical name -> possible report names
    LAB_ALIASES = {
        "WBC": [
            "wbc",
            "wbc count",
            "total wbc count",
            "total leucocyte count",
            "total leukocyte count",
            "tlc",
        ],

        "Hemoglobin": [
            "haemoglobin",
            "hemoglobin",
            "hb",
        ],

        "RBC": [
            "rbc",
            "rbc count",
            "total rbc count",
        ],

        "Hematocrit": [
            "hematocrit",
            "haematocrit",
            "pcv",
            "packed cell volume",
        ],

        "Platelets": [
            "platelet count",
            "platelets",
        ],

        "Neutrophils": [
            "neutrophils",
            "segmented neutrophils",
        ],

        "Lymphocytes": [
            "lymphocytes",
        ],

        "Monocytes": [
            "monocytes",
        ],

        "Eosinophils": [
            "eosinophils",
        ],

        "Basophils": [
            "basophils",
        ],

        "MCV": [
            "mcv",
            "mean corpuscular volume",
        ],

        "MCH": [
            "mch",
            "mean corpuscular hemoglobin",
        ],

        "MCHC": [
            "mchc",
            "mean corpuscular hemoglobin concentration",
        ],

        "RDW": [
            "rdw",
            "red cell distribution width",
        ],

        "Lactate": [
            "lactate",
            "blood lactate",
            "serum lactate",
            "lactic acid",
        ],

        "Creatinine": [
            "creatinine",
            "serum creatinine",
        ],

        "BUN": [
            "bun",
            "blood urea nitrogen",
        ],

        "Urea": [
            "urea",
            "blood urea",
            "serum urea",
        ],

        "Sodium": [
            "sodium",
            "serum sodium",
        ],

        "Potassium": [
            "potassium",
            "serum potassium",
        ],

        "Bilirubin": [
            "bilirubin",
            "total bilirubin",
        ],

        "CRP": [
            "crp",
            "c reactive protein",
            "c-reactive protein",
        ],

        "ESR": [
            "esr",
            "erythrocyte sedimentation rate",
        ],
    }

    SEPSIS_REQUIRED_FEATURES = [
        "HR",
        "O2Sat",
        "Temp",
        "SBP",
        "MAP",
        "Resp",
        "WBC",
        "Lactate",
    ]

    def extract_text(self, image_path: str) -> str:
        path = Path(image_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Report not found: {image_path}"
            )

        image = Image.open(path)

        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6",
        )

        return text

    @staticmethod
    def _clean_line(line: str) -> str:
        line = line.replace("|", " ")
        line = re.sub(r"\s+", " ", line)

        return line.strip()

    @staticmethod
    def _clean_label(text: str) -> str:
        text = text.lower()

        text = re.sub(
            r"[^a-z\s\-]",
            " ",
            text,
        )

        text = re.sub(
            r"\s+",
            " ",
            text,
        )

        return text.strip()

    def _canonical_parameter(
        self,
        possible_label: str,
    ) -> Optional[str]:

        label = self._clean_label(
            possible_label
        )

        if not label:
            return None

        best_parameter = None
        best_score = 0

        for canonical, aliases in self.LAB_ALIASES.items():

            for alias in aliases:

                alias_clean = self._clean_label(
                    alias
                )

                # Exact label match
                if label == alias_clean:
                    return canonical

                # Whole-string fuzzy matching
                score = fuzz.ratio(
                    label,
                    alias_clean,
                )

                if score > best_score:
                    best_score = score
                    best_parameter = canonical

        # Handles common OCR mistakes such as:
        # WBC -> WAC
        # MCV -> MEV
        # PCV -> POV
        # RDW -> ROW
        if best_score >= 75:
            return best_parameter

        return None

    @staticmethod
    def _extract_numbers(
        line: str,
    ) -> list[float]:

        matches = re.findall(
            r"(?<![\w.])-?\d+(?:\.\d+)?",
            line,
        )

        values = []

        for value in matches:
            try:
                values.append(
                    float(value)
                )
            except ValueError:
                continue

        return values

    @staticmethod
    def _extract_unit(
        text: str,
    ) -> Optional[str]:

        unit_patterns = [
            r"x\s*10\^?3\s*/\s*[uµ]l",
            r"10\^?3\s*/\s*[uµ]l",
            r"mmol\s*/\s*l",
            r"mg\s*/\s*dl",
            r"g\s*/\s*dl",
            r"thou\s*/\s*mm3",
            r"thou\s*/\s*mm³",
            r"/\s*cu\.?\s*mm",
            r"/\s*cumm",
            r"cells\s*/\s*[uµ]l",
            r"mill\s*/\s*mm3",
            r"mm\s*/\s*hr",
            r"\%",
            r"\bfl\b",
            r"\bpg\b",
        ]

        for pattern in unit_patterns:

            match = re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            )

            if match:
                return match.group(0)

        return None

    @staticmethod
    def _normalize_value(
        parameter: str,
        value: float,
        unit: Optional[str],
    ) -> float:

        if parameter == "WBC":

            unit_text = (
                unit or ""
            ).lower()

            # Example:
            # 5500 /cu.mm -> 5.5
            # 9000 /cumm -> 9.0
            if (
                "cu" in unit_text
                or "cumm" in unit_text
                or "cells" in unit_text
            ):

                if value > 100:
                    return round(
                        value / 1000.0,
                        3,
                    )

            # Example:
            # 4.20 thou/mm3
            # already represented as 4.20
            if "thou" in unit_text:
                return round(
                    value,
                    3,
                )

            # OCR may lose the unit completely.
            # If value clearly resembles an absolute WBC count,
            # normalize carefully.
            if (
                unit is None
                and 1000 <= value <= 100000
            ):
                return round(
                    value / 1000.0,
                    3,
                )

        return value

    def extract_lab_results(
        self,
        text: str,
    ) -> dict:

        results = {}

        lines = []

        for line in text.splitlines():

            cleaned = self._clean_line(
                line
            )

            if cleaned:
                lines.append(
                    cleaned
                )

        for line in lines:

            # First number in the line is assumed
            # to be the reported result.
            number_match = re.search(
                r"-?\d+(?:\.\d+)?",
                line,
            )

            if not number_match:
                continue

            label_part = line[
                :number_match.start()
            ].strip()

            parameter = (
                self._canonical_parameter(
                    label_part
                )
            )

            if parameter is None:
                continue

            values = self._extract_numbers(
                line
            )

            if not values:
                continue

            raw_value = values[0]

            # Only inspect text immediately after
            # the result for a unit.
            after_result = line[
                number_match.end():
            ]

            next_number = re.search(
                r"-?\d+(?:\.\d+)?",
                after_result,
            )

            if next_number:
                result_context = (
                    after_result[
                        :next_number.start()
                    ]
                )
            else:
                result_context = (
                    after_result
                )

            unit = self._extract_unit(
                result_context
            )

            normalized_value = (
                self._normalize_value(
                    parameter,
                    raw_value,
                    unit,
                )
            )

            results[parameter] = {
                "value":
                    normalized_value,

                "raw_value":
                    raw_value,

                "unit":
                    unit,

                "raw_line":
                    line,
            }

        return results

    def get_sepsis_parameters(
        self,
        lab_results: dict,
    ) -> dict:

        sepsis_parameters = {}

        for feature in [
            "WBC",
            "Lactate",
        ]:

            if feature in lab_results:

                sepsis_parameters[
                    feature
                ] = lab_results[
                    feature
                ]["value"]

        return sepsis_parameters

    def analyze(
        self,
        image_path: str,
    ) -> dict:

        raw_text = self.extract_text(
            image_path
        )

        lab_results = (
            self.extract_lab_results(
                raw_text
            )
        )

        sepsis_parameters = (
            self.get_sepsis_parameters(
                lab_results
            )
        )

        missing = [
            feature
            for feature
            in self.SEPSIS_REQUIRED_FEATURES
            if feature
            not in sepsis_parameters
        ]

        status = (
            "success"
            if not missing
            else "partial"
        )

        return {
            "status":
                status,

            "source":
                "lab_report",

            "lab_results":
                lab_results,

            "sepsis_parameters":
                sepsis_parameters,

            "missing_sepsis_parameters":
                missing,

            "raw_text":
                raw_text,
        }