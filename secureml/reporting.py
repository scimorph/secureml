"""
Reporting functionality for SecureML.

This module provides tools for generating reports from audit trails and compliance checks,
which can be used to demonstrate compliance with privacy regulations.
"""

import os
import json
import datetime
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import jinja2
import base64
import matplotlib.pyplot as plt
from io import BytesIO
from pathlib import Path

from .compliance import ComplianceReport
from .audit import get_audit_logs

# Default report templates directory
DEFAULT_TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Create templates directory if it doesn't exist
if not os.path.exists(DEFAULT_TEMPLATES_DIR):
    os.makedirs(DEFAULT_TEMPLATES_DIR)

# Initialize Jinja2 environment
JINJA_ENV = jinja2.Environment(
    loader=jinja2.FileSystemLoader(DEFAULT_TEMPLATES_DIR),
    autoescape=jinja2.select_autoescape(['html', 'xml'])
)


class ReportGenerator:
    """
    Class for generating compliance and audit reports.
    
    The ReportGenerator class provides methods for creating
    HTML and PDF reports from audit logs and compliance checks.
    """
    
    def __init__(
        self, 
        templates_dir: Optional[str] = None,
        custom_css: Optional[str] = None
    ):
        """
        Initialize a report generator.
        
        Args:
            templates_dir: Directory containing report templates
            custom_css: Custom CSS for HTML reports
        """
        self.templates_dir = templates_dir or DEFAULT_TEMPLATES_DIR
        self.custom_css = custom_css
        
        # Create a Jinja2 environment
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.templates_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self) -> None:
        """Create default report templates if they don't exist."""
        compliance_template_path = os.path.join(self.templates_dir, "compliance_report.html")
        audit_template_path = os.path.join(self.templates_dir, "audit_report.html")
        css_path = os.path.join(self.templates_dir, "report_style.css")
        
        # Default CSS
        default_css = """
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        h1, h2, h3, h4 {
            color: #2c3e50;
        }
        
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        
        .report-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
        }
        
        .report-summary {
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin-bottom: 20px;
        }
        
        .report-section {
            margin-bottom: 30px;
        }
        
        table {
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }
        
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #f2f2f2;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .issue {
            background-color: #ffe6e6;
        }
        
        .warning {
            background-color: #fff3cd;
        }
        
        .passed {
            background-color: #d4edda;
        }
        
        .high {
            color: #dc3545;
            font-weight: bold;
        }
        
        .medium {
            color: #fd7e14;
            font-weight: bold;
        }
        
        .low {
            color: #6c757d;
        }
        
        .timestamp {
            color: #6c757d;
            font-size: 0.9em;
        }
        
        .chart-container {
            margin: 20px 0;
            text-align: center;
        }
        
        footer {
            margin-top: 40px;
            padding-top: 10px;
            border-top: 1px solid #ddd;
            text-align: center;
            font-size: 0.9em;
            color: #6c757d;
        }
        """
        
        # Default compliance report template
        default_compliance_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Compliance Report - {{ report.regulation }}</title>
            <style>
                {{ css }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <div>
                    <h1>Compliance Report</h1>
                    <p>Regulation: {{ report.regulation }}</p>
                    <p>Generated: {{ generation_time }}</p>
                </div>
                {% if logo_base64 %}
                <div>
                    <img src="data:image/png;base64,{{ logo_base64 }}" alt="Logo" style="max-height: 60px;">
                </div>
                {% endif %}
            </div>
            
            <div class="report-summary">
                <h2>Summary</h2>
                <p>Status: <strong>{{ "Compliant" if report.summary().compliant else "Non-compliant" }}</strong></p>
                <p>Passed Checks: {{ report.passed_checks|length }}</p>
                <p>Issues: {{ report.issues|length }}</p>
                <p>Warnings: {{ report.warnings|length }}</p>
            </div>
            
            {% if charts.issues_by_severity_base64 %}
            <div class="chart-container">
                <img src="data:image/png;base64,{{ charts.issues_by_severity_base64 }}" alt="Issues by Severity">
            </div>
            {% endif %}
            
            {% if report.issues %}
            <div class="report-section">
                <h2>Issues</h2>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Issue</th>
                        <th>Severity</th>
                        <th>Recommendation</th>
                    </tr>
                    {% for issue in report.issues %}
                    <tr class="issue">
                        <td>{{ issue.component }}</td>
                        <td>{{ issue.issue }}</td>
                        <td class="{{ issue.severity }}">{{ issue.severity|upper }}</td>
                        <td>{{ issue.recommendation }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if report.warnings %}
            <div class="report-section">
                <h2>Warnings</h2>
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Warning</th>
                        <th>Recommendation</th>
                    </tr>
                    {% for warning in report.warnings %}
                    <tr class="warning">
                        <td>{{ warning.component }}</td>
                        <td>{{ warning.warning }}</td>
                        <td>{{ warning.recommendation }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if report.passed_checks %}
            <div class="report-section">
                <h2>Passed Checks</h2>
                <ul>
                    {% for check in report.passed_checks %}
                    <li class="passed">{{ check }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            <footer>
                <p>Generated with SecureML - {{ secureml_version }}</p>
            </footer>
        </body>
        </html>
        """
        
        # Default audit report template
        default_audit_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Audit Report - {{ title }}</title>
            <style>
                {{ css }}
            </style>
        </head>
        <body>
            <div class="report-header">
                <div>
                    <h1>Audit Trail Report</h1>
                    <p>{{ title }}</p>
                    <p>Generated: {{ generation_time }}</p>
                </div>
                {% if logo_base64 %}
                <div>
                    <img src="data:image/png;base64,{{ logo_base64 }}" alt="Logo" style="max-height: 60px;">
                </div>
                {% endif %}
            </div>
            
            <div class="report-summary">
                <h2>Summary</h2>
                <p>Operation: {{ metadata.operation }}</p>
                {% if metadata.start_time and metadata.end_time %}
                <p>Time Period: {{ metadata.start_time }} to {{ metadata.end_time }}</p>
                {% endif %}
                <p>Total Events: {{ logs|length }}</p>
                {% if metadata.regulations %}
                <p>Relevant Regulations: {{ metadata.regulations|join(', ') }}</p>
                {% endif %}
            </div>
            
            {% if charts.events_by_type_base64 %}
            <div class="chart-container">
                <img src="data:image/png;base64,{{ charts.events_by_type_base64 }}" alt="Events by Type">
            </div>
            {% endif %}
            
            <div class="report-section">
                <h2>Event Timeline</h2>
                <table>
                    <tr>
                        <th>Timestamp</th>
                        <th>Event Type</th>
                        <th>Details</th>
                    </tr>
                    {% for event in logs %}
                    <tr>
                        <td class="timestamp">{{ event.timestamp }}</td>
                        <td>{{ event.event_type }}</td>
                        <td>
                            <ul>
                            {% for key, value in event.items() %}
                                {% if key not in ['timestamp', 'event_type'] %}
                                <li><strong>{{ key }}:</strong> {{ value }}</li>
                                {% endif %}
                            {% endfor %}
                            </ul>
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            
            <footer>
                <p>Generated with SecureML - {{ secureml_version }}</p>
            </footer>
        </body>
        </html>
        """
        
        # Write the templates if they don't exist
        if not os.path.exists(compliance_template_path):
            with open(compliance_template_path, "w") as f:
                f.write(default_compliance_template)
        
        if not os.path.exists(audit_template_path):
            with open(audit_template_path, "w") as f:
                f.write(default_audit_template)
        
        if not os.path.exists(css_path):
            with open(css_path, "w") as f:
                f.write(default_css)
    
    def _get_css(self) -> str:
        """Get the CSS for the reports."""
        if self.custom_css:
            return self.custom_css
        
        css_path = os.path.join(self.templates_dir, "report_style.css")
        if os.path.exists(css_path):
            with open(css_path, "r") as f:
                return f.read()
        
        return ""
    
    def _generate_compliance_charts(self, report: ComplianceReport) -> Dict[str, str]:
        """
        Generate charts for the compliance report.
        
        Args:
            report: The compliance report
            
        Returns:
            Dictionary of base64-encoded chart images
        """
        charts = {}
        
        # Create a chart for issues by severity
        if report.issues:
            # Count issues by severity
            severity_counts = {"high": 0, "medium": 0, "low": 0}
            for issue in report.issues:
                severity = issue["severity"].lower()
                if severity in severity_counts:
                    severity_counts[severity] += 1
            
            # Create a pie chart
            plt.figure(figsize=(8, 5))
            colors = ['#dc3545', '#fd7e14', '#6c757d']
            labels = [f"{k.capitalize()} ({v})" for k, v in severity_counts.items() if v > 0]
            sizes = [v for k, v in severity_counts.items() if v > 0]
            
            if sizes:  # Only create the chart if there's data
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
                plt.axis('equal')
                plt.title('Issues by Severity')
                
                # Convert to base64
                buffer = BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                charts["issues_by_severity_base64"] = base64.b64encode(buffer.read()).decode('utf-8')
        
        return charts
    
    def _generate_audit_charts(self, logs: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate charts for the audit report.
        
        Args:
            logs: The audit logs
            
        Returns:
            Dictionary of base64-encoded chart images
        """
        charts = {}
        
        if logs:
            # Count events by type
            event_types = {}
            for log in logs:
                event_type = log.get("event_type", "unknown")
                event_types[event_type] = event_types.get(event_type, 0) + 1
            
            # Create a bar chart
            plt.figure(figsize=(10, 6))
            colors = plt.cm.tab10.colors
            plt.bar(event_types.keys(), event_types.values(), color=colors[:len(event_types)])
            plt.xticks(rotation=45, ha="right")
            plt.title('Events by Type')
            plt.tight_layout()
            
            # Convert to base64
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            charts["events_by_type_base64"] = base64.b64encode(buffer.read()).decode('utf-8')
        
        return charts
    
    def generate_compliance_report(
        self,
        report: ComplianceReport,
        output_file: str,
        logo_path: Optional[str] = None,
        include_charts: bool = True,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate a compliance report.
        
        Args:
            report: The compliance report
            output_file: Path to write the report to
            logo_path: Path to a logo image
            include_charts: Whether to include charts
            additional_context: Additional context data for the template
        
        Returns:
            Path to the generated report
        """
        # Load the template
        template = self.jinja_env.get_template("compliance_report.html")
        
        # Prepare the context
        context = {
            "report": report,
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "css": self._get_css(),
            "secureml_version": "0.3.1"
        }
        
        # Add logo if provided
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_base64 = base64.b64encode(f.read()).decode('utf-8')
                context["logo_base64"] = logo_base64
        
        # Generate charts if requested
        if include_charts:
            context["charts"] = self._generate_compliance_charts(report)
        
        # Add additional context
        if additional_context:
            context.update(additional_context)
        
        # Render the template
        html = template.render(**context)
        
        # Write the output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        # Convert to PDF if output is PDF
        if output_file.endswith(".pdf"):
            try:
                from weasyprint import HTML
                pdf_file = output_file
                html_file = output_file.replace(".pdf", ".html")
                
                # Write the HTML first
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html)
                
                # Convert to PDF
                HTML(html_file).write_pdf(pdf_file)
                
                # Remove the temporary HTML file
                os.remove(html_file)
            except ImportError:
                raise ImportError(
                    "WeasyPrint is required for PDF generation. "
                    "Install it with: pip install weasyprint"
                )
        
        return output_file
    
    def generate_audit_report(
        self,
        logs: Union[List[Dict[str, Any]], str, Dict[str, Any]],
        output_file: str,
        title: str = "Audit Trail Report",
        logo_path: Optional[str] = None,
        include_charts: bool = True,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate an audit report.
        
        Args:
            logs: The audit logs, operation ID, or operation name
            output_file: Path to write the report to
            title: Title for the report
            logo_path: Path to a logo image
            include_charts: Whether to include charts
            additional_context: Additional context data for the template
        
        Returns:
            Path to the generated report
        """
        # Get logs if string or dict provided
        if isinstance(logs, str):
            # Assume it's an operation ID
            actual_logs = get_audit_logs(operation_id=logs)
        elif isinstance(logs, dict) and not isinstance(logs, list):
            # Assume it has query parameters
            actual_logs = get_audit_logs(**logs)
        else:
            actual_logs = logs
        
        # Sort logs by timestamp
        actual_logs = sorted(actual_logs, key=lambda x: x.get("timestamp", ""))
        
        # Prepare metadata
        metadata = {
            "operation": title
        }
        
        # Extract operation name and regulations from logs if available
        if actual_logs:
            first_log = actual_logs[0]
            if "operation_name" in first_log:
                metadata["operation"] = first_log["operation_name"]
            if "regulations" in first_log:
                metadata["regulations"] = first_log["regulations"]
            
            # Get time range
            if "timestamp" in first_log:
                metadata["start_time"] = first_log["timestamp"]
                metadata["end_time"] = actual_logs[-1].get("timestamp", "")
        
        # Load the template
        template = self.jinja_env.get_template("audit_report.html")
        
        # Prepare the context
        context = {
            "logs": actual_logs,
            "title": title,
            "metadata": metadata,
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "css": self._get_css(),
            "secureml_version": "0.3.1"
        }
        
        # Add logo if provided
        if logo_path and os.path.exists(logo_path):
            with open(logo_path, "rb") as f:
                logo_base64 = base64.b64encode(f.read()).decode('utf-8')
                context["logo_base64"] = logo_base64
        
        # Generate charts if requested
        if include_charts:
            context["charts"] = self._generate_audit_charts(actual_logs)
        
        # Add additional context
        if additional_context:
            context.update(additional_context)
        
        # Render the template
        html = template.render(**context)
        
        # Write the output
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html)
        
        # Convert to PDF if output is PDF
        if output_file.endswith(".pdf"):
            try:
                from weasyprint import HTML
                pdf_file = output_file
                html_file = output_file.replace(".pdf", ".html")
                
                # Write the HTML first
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html)
                
                # Convert to PDF
                HTML(html_file).write_pdf(pdf_file)
                
                # Remove the temporary HTML file
                os.remove(html_file)
            except ImportError:
                raise ImportError(
                    "WeasyPrint is required for PDF generation. "
                    "Install it with: pip install weasyprint"
                )
        
        return output_file


# Enhance the ComplianceReport class to include report generation
def generate_report(
    self,
    output_file: str,
    format: str = "html",
    logo_path: Optional[str] = None,
    include_charts: bool = True
) -> str:
    """
    Generate a report from the compliance check results.
    
    Args:
        output_file: Path to write the report to
        format: Report format ('html' or 'pdf')
        logo_path: Path to a logo image
        include_charts: Whether to include charts in the report
    
    Returns:
        Path to the generated report
    """
    # Determine the output file extension
    if not output_file.endswith(f".{format}"):
        output_file = f"{output_file}.{format}"
    
    # Create a report generator
    generator = ReportGenerator()
    
    # Generate the report
    return generator.generate_compliance_report(
        report=self,
        output_file=output_file,
        logo_path=logo_path,
        include_charts=include_charts
    )

# Add the generate_report method to ComplianceReport
from secureml.compliance import ComplianceReport
ComplianceReport.generate_report = generate_report 