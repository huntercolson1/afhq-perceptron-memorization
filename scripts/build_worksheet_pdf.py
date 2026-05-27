#!/usr/bin/env python3
"""Build the downloadable worksheet PDF from the Markdown source."""

from __future__ import annotations

import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    markdown_path = project_root / "docs" / "worksheet.md"
    html_path = project_root / "docs" / "perceptron_vc_dimension_worksheet.html"
    pdf_path = project_root / "docs" / "perceptron_vc_dimension_worksheet.pdf"

    subprocess.run(
        [
            "pandoc",
            str(markdown_path),
            "-t",
            "html",
            "-s",
            "-o",
            str(html_path),
            "--metadata",
            "title=Perceptron VC-Dimension Worksheet",
        ],
        check=True,
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(pdf_path),
            format="Letter",
            margin={"top": "0.65in", "right": "0.65in", "bottom": "0.65in", "left": "0.65in"},
            print_background=True,
        )
        browser.close()

    print(f"Wrote {pdf_path}")


if __name__ == "__main__":
    main()
