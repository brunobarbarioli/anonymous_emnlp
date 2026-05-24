"""
Consolidated Pydantic schemas for agent tools.

This module provides standardized input schemas for all tools used
by the replication agents. Centralizing schemas here avoids duplication
and ensures consistency across different agent implementations.
"""
from pydantic import BaseModel, Field


class CodeExecutionInput(BaseModel):
    """Input schema for code execution tool."""
    code: str = Field(description="The code to execute")
    language: str = Field(
        description="Programming language: 'python', 'r', or 'stata'"
    )
    description: str = Field(
        description="Description of what this code does"
    )


class FileReadInput(BaseModel):
    """Input schema for file reading tool."""
    file_path: str = Field(
        description="Path to the file to read (use 'data/' prefix for data files)"
    )


class WriteFileInput(BaseModel):
    """Input schema for file writing tool."""
    file_path: str = Field(
        description="Path to the file to write (relative to the run workspace unless absolute)"
    )
    content: str = Field(
        description="Content to write"
    )


class PDFExtractionInput(BaseModel):
    """Input schema for PDF text extraction tool."""
    pdf_path: str = Field(
        description="Path to the PDF file"
    )


class CompareValuesInput(BaseModel):
    """Input schema for value comparison tool."""
    name: str = Field(
        description="Name of the metric being compared (e.g., 'Model1_coefficient')"
    )
    original_value: float = Field(
        description="Original value from the paper"
    )
    reproduced_value: float = Field(
        description="Reproduced value from your analysis"
    )
    metric_id: str = Field(
        default="",
        description="Stable metric identifier. If omitted, the metric name will be slugified."
    )
    table_name: str = Field(
        default="",
        description="Table or figure name containing the metric"
    )
    page: int = Field(
        default=0,
        description="1-based page number containing the original metric"
    )
    row_label: str = Field(
        default="",
        description="Optional row label for the metric"
    )
    column_label: str = Field(
        default="",
        description="Optional column label for the metric"
    )
    provenance: str = Field(
        default="",
        description="Free-form provenance note describing where the original value came from"
    )


class CompareMetricInput(BaseModel):
    """Input schema for manifest-backed metric comparison."""
    metric_id: str = Field(
        description="Stable manifest metric identifier such as 'Table2_M4_aligned'"
    )
    reproduced_value: float = Field(
        description="Reproduced full-precision value from the generated output"
    )
    provenance: str = Field(
        default="",
        description="Optional provenance note describing where the reproduced value was extracted from"
    )


class MetricTargetInput(BaseModel):
    """Input schema for registering a target metric before execution."""
    metric_id: str = Field(
        description="Stable metric identifier such as 'Table2_model1_treatment'"
    )
    display_name: str = Field(
        description="Human-readable name for the metric"
    )
    original_value: float = Field(
        description="Original value from the paper"
    )
    item_id: str = Field(
        default="",
        description="Inventory item identifier such as 'Table2' or 'Claim03'"
    )
    table_name: str = Field(
        default="",
        description="Table or figure name"
    )
    page: int = Field(
        default=0,
        description="1-based page number"
    )
    row_label: str = Field(
        default="",
        description="Optional row label"
    )
    column_label: str = Field(
        default="",
        description="Optional column label"
    )
    provenance: str = Field(
        default="",
        description="Where the original value was obtained from"
    )
    notes: str = Field(
        default="",
        description="Additional notes about this target metric"
    )


class MarkInventoryItemInput(BaseModel):
    """Input schema for locking an exploratory inventory item."""
    item_id: str = Field(
        description="Inventory item identifier such as 'Table2' or 'Claim03'"
    )
    expected_target_count: int = Field(
        description="Total number of required numeric targets for this inventory item"
    )
    notes: str = Field(
        default="",
        description="Optional note explaining the inventory decision"
    )


class ListDirectoryInput(BaseModel):
    """Input schema for directory listing tool."""
    directory: str = Field(
        default="data",
        description="Directory to list (default: 'data')"
    )


class WebSearchInput(BaseModel):
    """Input schema for web search tool."""
    query: str = Field(
        description="Search query string"
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return"
    )


class DatasetInfoInput(BaseModel):
    """Input schema for dataset information tool."""
    file_path: str = Field(
        description="Path to the dataset file"
    )
    sample_rows: int = Field(
        default=5,
        description="Number of sample rows to display"
    )


class SaveResultInput(BaseModel):
    """Input schema for persisting an analysis result in the report summary."""
    name: str = Field(description="Short name for the saved result")
    description: str = Field(description="What this result captures")
    code: str = Field(description="Code used to produce the result")
    language: str = Field(description="Language used for the code snippet")
    output: str = Field(description="Execution output or result summary")


class SaveResultsInput(BaseModel):
    """Input schema for saving results tool."""
    results: dict = Field(
        description="Results dictionary to save"
    )
    filename: str = Field(
        description="Output filename (without extension)"
    )
    format: str = Field(
        default="json",
        description="Output format: 'json', 'csv', or 'both'"
    )


class RunOriginalScriptInput(BaseModel):
    """Input schema for running an original replication script."""
    script_path: str = Field(
        description="Path to the script file (e.g., 'data/an_main.R')"
    )
    path_substitutions: dict = Field(
        default_factory=dict,
        description="Path substitutions to apply (e.g., {'01_data/': 'data/', '../data/': 'data/'})"
    )


class RunPlannedStepInput(BaseModel):
    """Input schema for executing a planned STATA step."""
    step_id: str = Field(
        description="Planned step identifier such as 'step_01_master_do'"
    )
    retry_recipe_id: str = Field(
        default="",
        description="Optional recovery recipe identifier to annotate a retry attempt"
    )


class InspectStepLogInput(BaseModel):
    """Input schema for reading a planned step log."""
    step_id: str = Field(
        description="Planned step identifier whose wrapper log should be inspected"
    )


class ProbeDatasetSchemaInput(BaseModel):
    """Input schema for probing a STATA dataset schema."""
    dataset_path: str = Field(
        description="Path to a .dta dataset file, usually under the source package"
    )


class ExtractGeneratedOutputInput(BaseModel):
    """Input schema for indexing generated outputs for one paper item."""
    item_id: str = Field(
        default="",
        description="Optional paper item identifier such as 'Table1' or 'Figure2'"
    )
    path_hint: str = Field(
        default="",
        description="Optional filename or relative path hint for narrowing the generated output search"
    )


class FocusPaperItemInput(BaseModel):
    """Input schema for focusing the exploratory agent on one paper item."""
    item_id: str = Field(
        description="Paper item identifier such as 'Table2' or 'Figure4'"
    )


class PaperMetadataInput(BaseModel):
    """Input schema for reporting paper metadata and replication package assessment."""
    paper_summary: str = Field(
        description="A concise summary of the paper's research question, methodology, and key findings (3-5 sentences)"
    )
    doi: str = Field(
        default="",
        description="The DOI of the paper if found in the text (e.g., '10.1234/example')"
    )
    citation: str = Field(
        default="",
        description="Full citation of the paper (authors, year, title, journal)"
    )
    has_raw_data: bool = Field(
        description="Whether raw (unprocessed) data files are present in the replication package"
    )
    has_cleaning_code: bool = Field(
        description="Whether code that cleans/processes the raw data is present"
    )
    has_clean_data: bool = Field(
        description="Whether clean/processed data ready for analysis is present"
    )
    has_analysis_code: bool = Field(
        description="Whether code to generate the results (tables/figures) is present"
    )
