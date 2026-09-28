from __future__ import annotations

import re
from typing import Optional

from rapidfuzz import fuzz


class ReportInputAgent:
    """OCR and complete structured extraction for uploaded laboratory reports."""

    LAB_ALIASES = {
        "BaseExcess": ["baseexcess", "base excess", "base excess be"],
        "HCO3": ["hco3", "bicarbonate", "bicarbonate hco3"],
        "FiO2": ["fio2", "fi o2", "fraction of inspired oxygen"],
        "pH": ["ph", "p h"],
        "PaCO2": ["paco2", "pa co2", "partial pressure of carbon dioxide"],
        "SaO2": ["sao2", "sa o2", "arterial oxygen saturation"],
        "AST": ["ast", "aspartate aminotransferase", "sgot"],
        "Alkalinephos": ["alkalinephos", "alkaline phos", "alkaline phosphatase", "alp"],
        "Calcium": ["calcium", "serum calcium"],
        "Chloride": ["chloride", "serum chloride"],
        "Creatinine": ["creatinine", "serum creatinine"],
        "Bilirubin_direct": ["bilirubin direct", "direct bilirubin"],
        "Glucose": ["glucose", "blood glucose", "blood sugar"],
        "Lactate": ["lactate", "blood lactate", "serum lactate", "lactic acid"],
        "Magnesium": ["magnesium", "serum magnesium"],
        "Phosphate": ["phosphate", "phosphorus", "serum phosphate"],
        "Potassium": ["potassium", "serum potassium"],
        "Sodium": ["sodium", "serum sodium"],
        "Bilirubin_total": ["bilirubin total", "total bilirubin"],
        "TroponinI": ["troponini", "troponin i", "troponin"],
        "Hct": ["hct", "hematocrit", "haematocrit", "pcv", "packed cell volume"],
        "Hgb": ["hgb", "hb", "hemoglobin", "haemoglobin"],
        "PTT": ["ptt", "partial thromboplastin time"],
        "WBC": ["wbc", "wbc count", "total wbc count", "total leucocyte count", "total leukocyte count", "tlc"],
        "Fibrinogen": ["fibrinogen"],
        "Temp": ["temp", "temperature", "body temperature"],
        "SBP": ["sbp", "systolic blood pressure", "blood pressure systolic"],
        "BP": ["bp", "blood pressure"],
        "MAP": ["map", "mean arterial pressure"],
        "DBP": ["dbp", "diastolic blood pressure", "blood pressure diastolic"],
        "Resp": ["resp", "resp rate", "respiratory rate", "respiration rate"],
        "ETCO2": ["etco2", "et co2", "end tidal carbon dioxide"],
        "HR": ["hr", "heart rate", "heartrate", "pulse", "pulse rate"],
        "O2Sat": ["o2sat", "o2 sat", "spo2", "oxygen saturation", "oxygen sat"],
        "Platelets": ["platelet count", "platelets"],
        "RBC": ["rbc", "rbc count", "total rbc count"],
        "MCV": ["mcv", "mean corpuscular volume"],
        "MCH": ["mch", "mean corpuscular hemoglobin"],
        "MCHC": [" mchc", "mean corpuscular hemoglobin concentration"],
        "RDW": ["rdw", "red cell distribution width"],
        "Urea": ["urea", "blood urea", "serum urea"],
        "CRP": ["crp", "c reactive protein", "c-reactive protein"],
        "ESR": ["esr", "erythrocyte sedimentation rate"],
    }

    SEPSIS_REQUIRED_FEATURES = ["HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Lactate"]

    @staticmethod
    def _clean_line(line: str) -> str:
        return re.sub(r"\s+", " ", line.replace("|", " ")).strip()

    @staticmethod
    def _clean_label(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s\-]", " ", text.lower())).strip()

    def _canonical_parameter(self, possible_label: str) -> Optional[str]:
        label = self._clean_label(possible_label)
        if not label:
            return None
        best_parameter: str | None = None
        best_score = 0
        for canonical, aliases in self.LAB_ALIASES.items():
            for alias in aliases:
                alias_clean = self._clean_label(alias)
                if label == alias_clean:
                    return canonical
                score = fuzz.ratio(label, alias_clean)
                if score > best_score:
                    best_parameter, best_score = canonical, score
        return best_parameter if best_score >= 75 else None

    @staticmethod
    def _extract_numbers(text: str) -> list[float]:
        pattern = r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
        return [float(value) for value in re.findall(pattern, text)]

    @staticmethod
    def _extract_value(text: str) -> float | None:
        values = ReportInputAgent._extract_numbers(text)
        return values[0] if values else None

    @staticmethod
    def _extract_unit(text: str) -> Optional[str]:
        return text.strip() or None

    @staticmethod
    def _normalize_value(parameter: str, value: float, unit: Optional[str]) -> float:
        if parameter == "WBC" and value > 100 and unit and any(token in unit.lower() for token in ("cu", "cumm", "cells")):
            return round(value / 1000.0, 3)
        if parameter == "WBC" and value > 1000 and unit is None:
            return round(value / 1000.0, 3)
        return value

    def _entry(self, parameter: str, source_name: str, value: float | None, unit: Optional[str], line: str) -> dict:
        return {
            "canonical_name": parameter,
            "source_name": source_name,
            "value": self._normalize_value(parameter, value, unit) if value is not None else None,
            "raw_value": value,
            "unit": unit,
            "raw_line": line,
        }

    @staticmethod
    def _fallback_parameter_name(source_name: str) -> str | None:
        cleaned = re.sub(r"[^A-Za-z0-9]+", "_", source_name).strip("_")
        return cleaned or None

    def extract_lab_results(self, text: str) -> dict:
        results: dict[str, dict] = {}
        value_pattern = r"(?<![\w.])[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|\bnull\b"

        for raw_line in text.splitlines():
            line = self._clean_line(raw_line)
            if not line:
                continue
            cells = [cell.strip() for cell in raw_line.split("|")]
            table_row = len(cells) >= 2
            if table_row:
                label = cells[0]
                value_text = cells[1]
                unit = cells[2] if len(cells) >= 3 and cells[2] else None
            else:
                match = re.search(value_pattern, line, flags=re.IGNORECASE)
                if not match:
                    continue
                label = line[:match.start()].strip()
                value_text = match.group(0)
                unit = self._extract_unit(line[match.end():])

            parameter = self._canonical_parameter(label)
            if parameter is None:
                parameter = self._fallback_parameter_name(label)
            if parameter is None:
                continue
            is_null = value_text.lower() == "null"
            value = None if is_null else self._extract_value(value_text)
            if value is None and not is_null:
                continue

            if parameter == "BP" and not table_row and re.search(r"/|\bover\b", line, flags=re.IGNORECASE):
                values = self._extract_numbers(line)
                if len(values) >= 2:
                    results["SBP"] = self._entry("SBP", label, values[0], unit, line)
                    results["DBP"] = self._entry("DBP", label, values[1], unit, line)
                    results["MAP"] = self._entry("MAP", label, round((values[0] + 2 * values[1]) / 3, 2), unit, line)
                    continue

            results[parameter] = self._entry(parameter, label, value, unit, line)

        return results

    def get_sepsis_parameters(self, lab_results: dict) -> dict:
        return {
            feature: lab_results[feature]["value"]
            for feature in self.SEPSIS_REQUIRED_FEATURES
            if feature in lab_results
        }

    def analyze(self, raw_text: str) -> dict:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("Report extraction received no raw text")
        lab_results = self.extract_lab_results(raw_text)
        sepsis_parameters = self.get_sepsis_parameters(lab_results)
        missing = [feature for feature in self.SEPSIS_REQUIRED_FEATURES if feature not in sepsis_parameters]
        return {
            "status": "success" if not missing else "partial",
            "source": "lab_report",
            "lab_results": lab_results,
            "extracted_data": lab_results,
            "sepsis_parameters": sepsis_parameters,
            "missing_sepsis_parameters": missing,
            "raw_text": raw_text,
        }
