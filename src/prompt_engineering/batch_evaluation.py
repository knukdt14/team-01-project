"""무작위 RAG 질문 평가의 데이터 로딩, 집계, 보고서 생성을 담당한다."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import posixpath
import random
import statistics
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree

from prompt_templates import PROMPT_LABELS, PromptVariant


DATASET_SHEET_NAME = "질문_모범답안"
DEFAULT_SAMPLE_SIZE = 25
DEFAULT_RANDOM_SEED = 42

MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
OFFICE_REL_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
PACKAGE_REL_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
XML_NAMESPACES = {"x": MAIN_NS}

VEHICLE_TO_CAR = {
    "아반떼": "avante",
    "아반떼 하이브리드": "avante_hev",
    "아이오닉 6": "ioniq6",
    "아이오닉6": "ioniq6",
    "넥쏘": "nexo",
    "투싼": "tucson",
}

METRIC_LABELS = {
    "bertscore_precision": "BERTScore Precision",
    "bertscore_recall": "BERTScore Recall",
    "bertscore_f1": "BERTScore F1",
    "faithfulness": "Faithfulness",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
    "generation_time_seconds": "응답 생성 시간(초)",
}
QUALITY_METRICS = tuple(
    metric
    for metric in METRIC_LABELS
    if metric != "generation_time_seconds"
)
ALL_METRICS = tuple(METRIC_LABELS)

VARIANT_COLORS = {
    PromptVariant.BASIC.value: "#2563EB",
    PromptVariant.ROLE.value: "#7C3AED",
    PromptVariant.CONSTRAINT.value: "#059669",
    PromptVariant.FEW_SHOT.value: "#EA580C",
}


@dataclass(frozen=True)
class BenchmarkQuestion:
    """Excel 질문·모범답안 한 행."""

    question_id: str
    vehicle: str
    car: str
    answerability: str
    difficulty: str
    question_type: str
    question: str
    reference_answer: str
    key_terms: str
    source_pdf: str
    source_page: str
    source_section: str
    evaluation_focus: str

    def to_dict(self) -> dict[str, str]:
        return {
            "question_id": self.question_id,
            "vehicle": self.vehicle,
            "car": self.car,
            "answerability": self.answerability,
            "difficulty": self.difficulty,
            "question_type": self.question_type,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "key_terms": self.key_terms,
            "source_pdf": self.source_pdf,
            "source_page": self.source_page,
            "source_section": self.source_section,
            "evaluation_focus": self.evaluation_focus,
        }


def dataset_sha256(path: Path) -> str:
    """재개 실행에서 데이터셋 변경 여부를 확인할 해시를 계산한다."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []

    return [
        "".join(node.text or "" for node in item.findall(".//x:t", XML_NAMESPACES))
        for item in root.findall("x:si", XML_NAMESPACES)
    ]


def _worksheet_path(
    archive: zipfile.ZipFile,
    sheet_name: str,
) -> str:
    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook.findall(".//x:sheets/x:sheet", XML_NAMESPACES):
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get(f"{{{OFFICE_REL_NS}}}id")
            break

    if not relationship_id:
        available = [
            sheet.attrib.get("name", "")
            for sheet in workbook.findall(".//x:sheets/x:sheet", XML_NAMESPACES)
        ]
        raise ValueError(
            f'Excel에서 "{sheet_name}" 시트를 찾지 못했습니다. '
            f"사용 가능한 시트: {', '.join(available)}"
        )

    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    for relation in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        if relation.attrib.get("Id") != relationship_id:
            continue
        target = relation.attrib.get("Target", "")
        if target.startswith("/"):
            return target.lstrip("/")
        return posixpath.normpath(posixpath.join("xl", target))

    raise ValueError(f"{sheet_name} 시트의 XML 경로를 찾지 못했습니다.")


def _column_index(cell_reference: str) -> int:
    letters = "".join(character for character in cell_reference if character.isalpha())
    if not letters:
        raise ValueError(f"잘못된 Excel 셀 주소입니다: {cell_reference}")

    index = 0
    for character in letters.upper():
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1


def _cell_text(
    cell: ElementTree.Element,
    shared_strings: Sequence[str],
) -> str:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        return "".join(
            node.text or ""
            for node in cell.findall(".//x:is/x:t", XML_NAMESPACES)
        ).strip()

    value_node = cell.find("x:v", XML_NAMESPACES)
    if value_node is None or value_node.text is None:
        return ""

    value = value_node.text
    if cell_type == "s":
        try:
            return shared_strings[int(value)].strip()
        except (IndexError, ValueError) as exc:
            raise ValueError("Excel 공유 문자열 인덱스가 올바르지 않습니다.") from exc
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    return value.strip()


def _worksheet_rows(
    archive: zipfile.ZipFile,
    worksheet_path: str,
) -> list[list[str]]:
    shared_strings = _shared_strings(archive)
    worksheet = ElementTree.fromstring(archive.read(worksheet_path))
    rows: list[list[str]] = []

    for row in worksheet.findall(".//x:sheetData/x:row", XML_NAMESPACES):
        values_by_column: dict[int, str] = {}
        for cell in row.findall("x:c", XML_NAMESPACES):
            reference = cell.attrib.get("r", "")
            values_by_column[_column_index(reference)] = _cell_text(
                cell,
                shared_strings,
            )

        if not values_by_column:
            rows.append([])
            continue

        width = max(values_by_column) + 1
        rows.append(
            [values_by_column.get(column, "") for column in range(width)]
        )

    return rows


def load_benchmark_questions(
    path: Path,
    sheet_name: str = DATASET_SHEET_NAME,
) -> list[BenchmarkQuestion]:
    """추가 Excel 패키지 없이 xlsx의 질문·모범답안 시트를 읽는다."""

    if not path.is_file():
        raise FileNotFoundError(f"질문 데이터셋을 찾을 수 없습니다: {path}")

    try:
        with zipfile.ZipFile(path) as archive:
            worksheet_path = _worksheet_path(archive, sheet_name)
            rows = _worksheet_rows(archive, worksheet_path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f"올바른 xlsx 파일이 아닙니다: {path}") from exc

    if len(rows) < 2:
        raise ValueError("질문 데이터셋에 데이터 행이 없습니다.")

    headers = [header.strip() for header in rows[0]]
    required_headers = {"ID", "차량", "질문", "모범답안"}
    missing_headers = sorted(required_headers.difference(headers))
    if missing_headers:
        raise ValueError(
            "질문 데이터셋에 필수 열이 없습니다: "
            + ", ".join(missing_headers)
        )

    questions: list[BenchmarkQuestion] = []
    seen_ids: set[str] = set()
    for row_number, values in enumerate(rows[1:], start=2):
        padded = values + [""] * max(0, len(headers) - len(values))
        record = dict(zip(headers, padded))

        question_id = record.get("ID", "").strip()
        question = record.get("질문", "").strip()
        reference_answer = record.get("모범답안", "").strip()
        vehicle = record.get("차량", "").strip()
        if not any((question_id, question, reference_answer, vehicle)):
            continue
        if not all((question_id, question, reference_answer, vehicle)):
            raise ValueError(
                f"Excel {row_number}행의 ID·차량·질문·모범답안 중 빈 값이 있습니다."
            )
        if question_id in seen_ids:
            raise ValueError(f"중복 질문 ID가 있습니다: {question_id}")
        seen_ids.add(question_id)

        try:
            car = VEHICLE_TO_CAR[vehicle]
        except KeyError as exc:
            raise ValueError(
                f"Excel {row_number}행의 차량을 검색 코드로 바꿀 수 없습니다: "
                f"{vehicle}"
            ) from exc

        questions.append(
            BenchmarkQuestion(
                question_id=question_id,
                vehicle=vehicle,
                car=car,
                answerability=record.get("답변 가능 여부", "").strip(),
                difficulty=record.get("난이도", "").strip(),
                question_type=record.get("질문 유형", "").strip(),
                question=question,
                reference_answer=reference_answer,
                key_terms=record.get("핵심 채점어", "").strip(),
                source_pdf=record.get("근거 PDF", "").strip(),
                source_page=record.get("근거 페이지(PDF)", "").strip(),
                source_section=record.get("근거 섹션", "").strip(),
                evaluation_focus=record.get("평가 초점", "").strip(),
            )
        )

    if not questions:
        raise ValueError("유효한 질문·모범답안이 없습니다.")
    return questions


def select_random_questions(
    questions: Sequence[BenchmarkQuestion],
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_RANDOM_SEED,
) -> list[BenchmarkQuestion]:
    """동일한 seed에서 같은 질문 표본이 나오도록 단순 무작위 추출한다."""

    if sample_size < 1:
        raise ValueError("표본 수는 1 이상이어야 합니다.")
    if sample_size > len(questions):
        raise ValueError(
            f"표본 수({sample_size})가 전체 질문 수({len(questions)})보다 큽니다."
        )

    return random.Random(seed).sample(list(questions), sample_size)


def _finite_values(values: Iterable[Any]) -> list[float]:
    finite: list[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite.append(numeric)
    return finite


def summarize_results(
    results: Sequence[Mapping[str, Any]],
    variants: Sequence[PromptVariant] | None = None,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """질문별 결과를 프롬프트별 지표 평균과 유효 표본 수로 집계한다."""

    selected_variants = list(variants or PromptVariant)
    summary: dict[str, dict[str, dict[str, float | int]]] = {}

    for variant in selected_variants:
        variant_summary: dict[str, dict[str, float | int]] = {}
        for metric in ALL_METRICS:
            values = _finite_values(
                result.get("variants", {})
                .get(variant.value, {})
                .get(metric)
                for result in results
            )
            variant_summary[metric] = {
                "mean": statistics.fmean(values) if values else float("nan"),
                "valid_count": len(values),
            }
        summary[variant.value] = variant_summary

    return summary


def write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    """중단 시 JSON이 깨지지 않도록 임시 파일을 거쳐 저장한다."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("평가 체크포인트 형식이 올바르지 않습니다.")
    return payload


def _display_number(value: Any, *, time_metric: bool = False) -> str:
    values = _finite_values([value])
    if not values:
        return "N/A"
    return f"{values[0]:.2f}" if time_metric else f"{values[0]:.3f}"


def write_summary_csv(
    path: Path,
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    variants: Sequence[PromptVariant] | None = None,
) -> None:
    selected_variants = list(variants or PromptVariant)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["variant", "label", *METRIC_LABELS.values()]
        )
        for variant in selected_variants:
            metrics = summary[variant.value]
            writer.writerow(
                [
                    variant.value,
                    PROMPT_LABELS[variant],
                    *[
                        _display_number(
                            metrics[metric]["mean"],
                            time_metric=metric == "generation_time_seconds",
                        )
                        for metric in ALL_METRICS
                    ],
                ]
            )


def write_detail_csv(
    path: Path,
    results: Sequence[Mapping[str, Any]],
    variants: Sequence[PromptVariant] | None = None,
) -> None:
    selected_variants = list(variants or PromptVariant)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "question_id",
                "vehicle",
                "car",
                "answerability",
                "difficulty",
                "question_type",
                "question",
                "reference_answer",
                "variant",
                "variant_label",
                "answer",
                *ALL_METRICS,
            ],
        )
        writer.writeheader()
        for result in results:
            for variant in selected_variants:
                variant_result = result.get("variants", {}).get(
                    variant.value,
                    {},
                )
                writer.writerow(
                    {
                        "question_id": result.get("question_id", ""),
                        "vehicle": result.get("vehicle", ""),
                        "car": result.get("car", ""),
                        "answerability": result.get("answerability", ""),
                        "difficulty": result.get("difficulty", ""),
                        "question_type": result.get("question_type", ""),
                        "question": result.get("question", ""),
                        "reference_answer": result.get(
                            "reference_answer",
                            "",
                        ),
                        "variant": variant.value,
                        "variant_label": PROMPT_LABELS[variant],
                        "answer": variant_result.get("answer", ""),
                        **{
                            metric: variant_result.get(metric, "")
                            for metric in ALL_METRICS
                        },
                    }
                )


def _best_variants(
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric: str,
    variants: Sequence[PromptVariant],
) -> set[str]:
    values = {
        variant.value: finite[0]
        for variant in variants
        if (
            finite := _finite_values(
                [summary[variant.value][metric]["mean"]]
            )
        )
    }
    if not values:
        return set()
    best_value = (
        min(values.values())
        if metric == "generation_time_seconds"
        else max(values.values())
    )
    return {
        variant
        for variant, value in values.items()
        if math.isclose(value, best_value, rel_tol=1e-9, abs_tol=1e-12)
    }


def build_html_report(
    *,
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    variants: Sequence[PromptVariant] | None = None,
) -> str:
    """외부 라이브러리 없이 열 수 있는 비교 대시보드 HTML을 만든다."""

    selected_variants = list(variants or PromptVariant)
    sample_size = int(metadata.get("sample_size", len(results)))
    top_k = int(metadata.get("top_k", 0))
    seed = metadata.get("seed", "")
    generated_at = html.escape(str(metadata.get("updated_at", "")))
    retriever_backend = html.escape(
        str(metadata.get("retriever_backend", ""))
    )
    provider = html.escape(str(metadata.get("answer_provider", "")))
    model = html.escape(str(metadata.get("answer_model", "")))
    evaluation_model = html.escape(
        str(metadata.get("evaluation_model", ""))
    )

    legend = "".join(
        (
            '<span class="legend-item">'
            f'<i style="background:{VARIANT_COLORS[variant.value]}"></i>'
            f"{html.escape(PROMPT_LABELS[variant])}</span>"
        )
        for variant in selected_variants
    )

    quality_sections: list[str] = []
    for metric in QUALITY_METRICS:
        bars: list[str] = []
        best = _best_variants(summary, metric, selected_variants)
        for variant in selected_variants:
            metric_summary = summary[variant.value][metric]
            finite = _finite_values([metric_summary["mean"]])
            value = finite[0] if finite else float("nan")
            width = min(100.0, max(0.0, value * 100)) if finite else 0
            best_badge = '<span class="best">BEST</span>' if variant.value in best else ""
            bars.append(
                '<div class="bar-row">'
                f'<div class="bar-label">{html.escape(PROMPT_LABELS[variant])}</div>'
                '<div class="track">'
                f'<div class="bar" style="width:{width:.2f}%;'
                f'background:{VARIANT_COLORS[variant.value]}"></div>'
                "</div>"
                f'<div class="bar-value">{_display_number(value)}{best_badge}</div>'
                "</div>"
            )
        quality_sections.append(
            '<section class="metric-card">'
            f"<h3>{html.escape(METRIC_LABELS[metric])}</h3>"
            + "".join(bars)
            + "</section>"
        )

    time_values = {
        variant.value: _finite_values(
            [summary[variant.value]["generation_time_seconds"]["mean"]]
        )
        for variant in selected_variants
    }
    finite_times = [
        values[0] for values in time_values.values() if values
    ]
    max_time = max(finite_times, default=1.0) or 1.0
    best_times = _best_variants(
        summary,
        "generation_time_seconds",
        selected_variants,
    )
    time_bars: list[str] = []
    for variant in selected_variants:
        values = time_values[variant.value]
        value = values[0] if values else float("nan")
        width = (value / max_time * 100) if values else 0
        best_badge = (
            '<span class="best">FASTEST</span>'
            if variant.value in best_times
            else ""
        )
        time_bars.append(
            '<div class="bar-row">'
            f'<div class="bar-label">{html.escape(PROMPT_LABELS[variant])}</div>'
            '<div class="track">'
            f'<div class="bar" style="width:{width:.2f}%;'
            f'background:{VARIANT_COLORS[variant.value]}"></div>'
            "</div>"
            f'<div class="bar-value">{_display_number(value, time_metric=True)}초'
            f"{best_badge}</div></div>"
        )

    header_cells = "".join(
        f"<th>{html.escape(METRIC_LABELS[metric])}</th>"
        for metric in ALL_METRICS
    )
    table_rows: list[str] = []
    for variant in selected_variants:
        cells: list[str] = []
        for metric in ALL_METRICS:
            metric_summary = summary[variant.value][metric]
            is_best = variant.value in _best_variants(
                summary,
                metric,
                selected_variants,
            )
            cell_class = ' class="best-cell"' if is_best else ""
            display = _display_number(
                metric_summary["mean"],
                time_metric=metric == "generation_time_seconds",
            )
            valid_count = int(metric_summary["valid_count"])
            cells.append(
                f"<td{cell_class}>{display}"
                f'<small>n={valid_count}</small></td>'
            )
        table_rows.append(
            "<tr>"
            f'<th class="variant-name">{html.escape(PROMPT_LABELS[variant])}</th>'
            + "".join(cells)
            + "</tr>"
        )

    question_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(result.get('question_id', '')))}</td>"
        f"<td>{html.escape(str(result.get('vehicle', '')))}</td>"
        f"<td>{html.escape(str(result.get('answerability', '')))}</td>"
        f"<td>{html.escape(str(result.get('difficulty', '')))}</td>"
        f"<td>{html.escape(str(result.get('question', '')))}</td>"
        "</tr>"
        for result in results
    )

    return f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG 프롬프트 4종 평가 결과</title>
<style>
:root {{
  color-scheme: light;
  --ink: #172033;
  --muted: #637083;
  --line: #e4e9f0;
  --surface: #ffffff;
  --canvas: #f4f7fb;
  --accent: #0f766e;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  color: var(--ink);
  background: var(--canvas);
  font-family: "Malgun Gothic", "Apple SD Gothic Neo", sans-serif;
}}
.container {{ max-width: 1480px; margin: 0 auto; padding: 40px 24px 64px; }}
.hero {{
  padding: 34px;
  border-radius: 22px;
  color: #fff;
  background: linear-gradient(135deg, #14213d 0%, #0f766e 100%);
  box-shadow: 0 18px 45px rgba(20, 33, 61, .18);
}}
.eyebrow {{ margin: 0 0 10px; opacity: .76; font-size: 13px; letter-spacing: .12em; }}
h1 {{ margin: 0; font-size: clamp(28px, 4vw, 46px); }}
.hero p {{ max-width: 820px; margin: 14px 0 0; line-height: 1.65; opacity: .88; }}
.kpis {{
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin: 20px 0;
}}
.kpi, .panel, .metric-card {{
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 16px;
  box-shadow: 0 7px 24px rgba(28, 42, 66, .06);
}}
.kpi {{ padding: 18px 20px; }}
.kpi strong {{ display: block; font-size: 22px; }}
.kpi span {{ color: var(--muted); font-size: 12px; }}
.panel {{ margin-top: 18px; padding: 24px; }}
.panel h2 {{ margin: 0 0 8px; font-size: 21px; }}
.note {{ margin: 5px 0 18px; color: var(--muted); font-size: 13px; line-height: 1.55; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 12px 20px; margin: 16px 0 20px; }}
.legend-item {{ display: inline-flex; align-items: center; gap: 7px; font-size: 12px; }}
.legend-item i {{ width: 11px; height: 11px; border-radius: 3px; }}
.metrics-grid {{
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 14px;
}}
.metric-card {{ padding: 18px; box-shadow: none; }}
.metric-card h3 {{ margin: 0 0 15px; font-size: 15px; }}
.metric-card.time {{ margin-top: 14px; }}
.bar-row {{
  display: grid;
  grid-template-columns: 155px minmax(120px, 1fr) 92px;
  gap: 10px;
  align-items: center;
  min-height: 30px;
}}
.bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-size: 11px; }}
.track {{ height: 11px; overflow: hidden; border-radius: 999px; background: #edf1f6; }}
.bar {{ height: 100%; min-width: 2px; border-radius: inherit; }}
.bar-value {{ font-variant-numeric: tabular-nums; font-size: 12px; }}
.best {{
  display: inline-block;
  margin-left: 5px;
  padding: 2px 5px;
  color: #0f766e;
  background: #dff7f1;
  border-radius: 999px;
  font-size: 8px;
  font-weight: 700;
}}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ padding: 11px 10px; border-bottom: 1px solid var(--line); text-align: center; }}
thead th {{ position: sticky; top: 0; color: #fff; background: #25324a; }}
.variant-name {{ min-width: 160px; text-align: left; }}
td small {{ display: block; margin-top: 4px; color: var(--muted); font-size: 9px; }}
.best-cell {{ color: #08766e; background: #effbf8; font-weight: 700; }}
.questions th:last-child, .questions td:last-child {{ min-width: 420px; text-align: left; }}
.questions td {{ vertical-align: top; }}
.footer {{ margin-top: 20px; color: var(--muted); text-align: right; font-size: 11px; }}
@media (max-width: 900px) {{
  .kpis, .metrics-grid {{ grid-template-columns: 1fr; }}
  .bar-row {{ grid-template-columns: 120px minmax(100px, 1fr) 82px; }}
}}
</style>
</head>
<body>
<main class="container">
  <section class="hero">
    <p class="eyebrow">RAG PROMPT BENCHMARK</p>
    <h1>프롬프트 4종 성능 비교</h1>
    <p>동일하게 무작위 추출한 질문과 동일한 top-{top_k}
    {retriever_backend} 검색 문맥을 사용해
    답변 품질, 근거 충실도, 검색 품질, 생성 시간을 비교한 결과입니다.</p>
  </section>

  <section class="kpis">
    <article class="kpi"><strong>{sample_size}</strong><span>무작위 질문 수</span></article>
    <article class="kpi"><strong>{len(selected_variants)}</strong><span>비교 프롬프트 수</span></article>
    <article class="kpi"><strong>{html.escape(str(seed))}</strong><span>재현용 random seed</span></article>
    <article class="kpi"><strong>{len(results)}</strong><span>평가 완료 질문 수</span></article>
  </section>

  <section class="panel">
    <h2>평균 품질 지표</h2>
    <p class="note">모든 품질 지표는 0~1 범위이며 높을수록 좋습니다.
    Context Precision과 Context Recall은 프롬프트 적용 전 동일한 검색 결과를
    평가하므로 {len(selected_variants)}개 기법에서 같은 값이 나오는 것이
    정상입니다.</p>
    <div class="legend">{legend}</div>
    <div class="metrics-grid">{''.join(quality_sections)}</div>
    <section class="metric-card time">
      <h3>평균 응답 생성 시간 · 낮을수록 좋음</h3>
      {''.join(time_bars)}
    </section>
  </section>

  <section class="panel">
    <h2>프롬프트별 평균값</h2>
    <p class="note">n은 오류·결측치를 제외하고 평균 계산에 사용된 질문 수입니다.</p>
    <div class="table-wrap">
      <table>
        <thead><tr><th>프롬프트</th>{header_cells}</tr></thead>
        <tbody>{''.join(table_rows)}</tbody>
      </table>
    </div>
  </section>

  <section class="panel">
    <h2>선정된 질문</h2>
    <p class="note">같은 25개 질문이 모든 프롬프트에 공통으로 적용되었습니다.</p>
    <div class="table-wrap">
      <table class="questions">
        <thead><tr><th>ID</th><th>차량</th><th>답변 가능 여부</th><th>난이도</th><th>질문</th></tr></thead>
        <tbody>{question_rows}</tbody>
      </table>
    </div>
  </section>

  <p class="footer">검색: {retriever_backend} top-{top_k}
  · 답변 모델: {provider} / {model} · 평가 모델: {evaluation_model}
  · 생성: {generated_at}</p>
</main>
</body>
</html>
"""


def write_html_report(
    path: Path,
    *,
    summary: Mapping[str, Mapping[str, Mapping[str, Any]]],
    results: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    variants: Sequence[PromptVariant] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_html_report(
            summary=summary,
            results=results,
            metadata=metadata,
            variants=variants,
        ),
        encoding="utf-8",
    )


def utc_timestamp() -> str:
    """체크포인트와 보고서에 사용할 초 단위 ISO 시각."""

    return datetime.now().astimezone().isoformat(timespec="seconds")
