<a name="readme-top"></a>

# Red Wine Quality Exploratory Data Analysis

This repository contains the source code, dataset, and written report for an
exploratory data analysis (EDA) of the UCI Red Wine Quality dataset. The work
focuses on understanding the physicochemical properties that distinguish higher
quality wines and provides supporting visualizations and statistics that are
summarized in an accompanying LaTeX report.

- **Status:** Course assignment submission
- **Primary language:** Python 3
- **Key deliverables:** `src/eda_analysis.py`, `report/eda_report.tex`,
  `data/red_wine_quality.csv`

## Table of Contents

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Repository Layout](#repository-layout)
4. [Getting Started](#getting-started)
5. [Running the EDA Script](#running-the-eda-script)

## Project Overview

The project explores how physicochemical measurements such as acidity, sugar,
and alcohol content relate to a wine's quality rating on a 0–10 scale. The
analysis covers:

- Loading and cleaning the dataset into a tidy Pandas `DataFrame`.
- Computing descriptive statistics and correlations between features and the
  quality label.
- Producing tables and figures that support the written analysis found in
  `report/eda_report.tex`.

The results emphasize relationships that practitioners can use to reason about
quality outcomes, such as which chemical properties most strongly influence the
rating assigned by experts.

## Dataset

The repository includes `data/red_wine_quality.csv`, a copy of the Red Wine
Quality dataset published on the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/wine+quality).

- **Observations:** 1,599 red wine samples from Portugal
- **Features:** 11 physicochemical measurements (e.g., acidity, sulfur dioxide)
  plus a subjective quality score
- **License:** Originally released for educational use; please review the UCI
  terms before redistribution.

The `src/eda_analysis.py` helper converts the dataset's column names to
`snake_case` for easier downstream analysis and ensures the data remains
read-only when executed.

## Repository Layout

```
├── data/                   # Source dataset used for the analysis
├── report/                 # LaTeX report describing findings and visuals
├── src/                    # Python utilities for loading and summarizing data
├── requirements.txt        # Python dependencies (currently only pandas)
├── CONTRIBUTING.md         # Guidelines for proposing improvements
└── README.md               # Project overview (this document)
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- A virtual environment (recommended)
- `pip` for dependency installation

### Installation

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies with:

   ```bash
   pip install -r requirements.txt
   ```

If you plan to build the report locally, make sure you also have a LaTeX
distribution (for example, TeX Live or MiKTeX) installed.

## Running the EDA Script

Execute the helper script to reproduce the summary statistics referenced in the
report:

```bash
python src/eda_analysis.py
```

The script prints:

1. The dataset shape (rows × columns).
2. Descriptive statistics (mean, median, standard deviation) for each feature.
3. Mean values of selected chemical properties grouped by quality rating.
4. Correlation scores between each feature and the quality label.

These outputs are intended for console inspection and to inform the LaTeX
report; the script does not write to disk.

## Working with the LaTeX Report

The report in `report/eda_report.tex` compiles into a PDF that summarizes the
findings along with relevant plots and tables. To build the report locally:

```bash
cd report
pdflatex eda_report.tex
```

You may need to run `pdflatex` multiple times to resolve references and
citations depending on your LaTeX setup.

## Contributing

Contributions that improve the analysis, documentation, or code are welcome.
Please review [`CONTRIBUTING.md`](CONTRIBUTING.md) for details on how to report
issues, propose enhancements, and follow the project's code of conduct.

## License

This project is distributed under the terms of the
[`LICENSE`](LICENSE). Review the license to understand the conditions for using
and sharing the code and accompanying materials.

## Using GitHub Under Protest

This project is currently hosted on GitHub. This is not ideal; GitHub is a
proprietary, trade-secret system that is not Free and Open Source Software
(FOSS). We are deeply concerned about using a proprietary system like GitHub to
develop our FOSS project. I have a [website](https://bellKevin.me) where the
project contributors are actively discussing how we can move away from GitHub
in the long term. We urge you to read about the
[Give up GitHub](https://GiveUpGitHub.org) campaign from
[the Software Freedom Conservancy](https://sfconservancy.org) to understand some
of the reasons why GitHub is not a good place to host FOSS projects.

If you are a contributor who personally has already quit using GitHub, please
email me at **bellKevin@pm.me** for how to send us contributions without using
GitHub directly.

Any use of this project's code by GitHub Copilot, past or present, is done
without our permission. We do not consent to GitHub's use of this project's code
in Copilot.

![Logo of the GiveUpGitHub campaign](https://sfconservancy.org/img/GiveUpGitHub.png)

<p align="right"><a href="#readme-top">back to top</a></p>
