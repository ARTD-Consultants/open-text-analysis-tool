"""Simplified report generation utilities."""

import os
import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import json

from ..models.analysis_result import AnalysisSession

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates simple analysis reports in Excel and JSON formats."""
    
    def __init__(self, output_dir: str = "output"):
        """Initialize report generator."""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_simple_report(
        self,
        session: AnalysisSession,
        theme_summary: Dict[str, int],
        report_title: str = "Qualitative Analysis Report"
    ) -> Dict[str, str]:
        """
        Generate simplified reports - Excel and JSON only.
        
        Returns:
            Dictionary mapping format to file path
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_files = {}
        
        # Generate Excel summary
        excel_path = os.path.join(self.output_dir, f"analysis_summary_{timestamp}.xlsx")
        self._generate_simple_excel_report(session, theme_summary, excel_path)
        report_files["excel"] = excel_path
        
        # Generate statistics summary
        stats_path = os.path.join(self.output_dir, f"statistics_{timestamp}.json")
        self._generate_simple_statistics_file(session, theme_summary, stats_path)
        report_files["statistics"] = stats_path
        
        logger.info(f"Generated report files: {list(report_files.keys())}")
        return report_files
    
    def _generate_simple_excel_report(
        self,
        session: AnalysisSession,
        theme_summary: Dict[str, int],
        file_path: str
    ) -> None:
        """Generate simplified Excel report."""
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            # Theme summary sheet
            theme_df = pd.DataFrame(
                [(theme, count) for theme, count in sorted(theme_summary.items(), key=lambda x: x[1], reverse=True)],
                columns=['Theme', 'Frequency']
            )
            theme_df.to_excel(writer, sheet_name='Theme Summary', index=False)
            
            # Session statistics
            stats = session.get_session_statistics()
            stats_df = pd.DataFrame([
                ['Total Entries', stats['total_entries']],
                ['Unique Themes', len(theme_summary)],
                ['Processing Time (seconds)', stats.get('session_duration', 0)],
                ['Average Themes per Entry', stats['average_themes_per_entry']],
                ['Overall Confidence', stats['overall_confidence']]
            ], columns=['Metric', 'Value'])
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        logger.info(f"Excel report saved to: {file_path}")
    
    def _generate_simple_statistics_file(
        self,
        session: AnalysisSession,
        theme_summary: Dict[str, int],
        file_path: str
    ) -> None:
        """Generate JSON statistics file."""
        
        stats = session.get_session_statistics()
        stats_data = {
            "analysis_summary": {
                "total_entries": stats['total_entries'],
                "unique_themes": len(theme_summary),
                "processing_time_seconds": stats.get('session_duration', 0),
                "average_themes_per_entry": stats['average_themes_per_entry'],
                "overall_confidence": stats['overall_confidence']
            },
            "theme_frequencies": theme_summary,
            "top_10_themes": dict(sorted(theme_summary.items(), key=lambda x: x[1], reverse=True)[:10]),
            "generated_timestamp": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logger.info(f"Statistics file saved to: {file_path}")