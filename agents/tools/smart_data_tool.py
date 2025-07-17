"""
Smart Data Tool - A tool for dynamically fetching government data.

This tool wraps the SmartDataAgent, allowing other agents to easily access
real-time government data for rate validation and other tasks.
"""

from typing import Dict, Any, List, Optional
import structlog

from ..smart_data_agent import SmartDataAgent, DataFetchingResult

logger = structlog.get_logger(__name__)

class SmartDataTool:
    """
    A tool for dynamically fetching relevant government data using the SmartDataAgent.
    """

    def __init__(self, smart_data_agent: SmartDataAgent):
        """
        Initializes the SmartDataTool.

        Args:
            smart_data_agent: An instance of the SmartDataAgent.
        """
        self.name = "smart_data_tool"
        self.description = "Fetches real-time government data based on document entities."
        self.smart_data_agent = smart_data_agent

    async def __call__(
        self,
        document_type: str,
        raw_text: str,
        state_code: Optional[str] = None,
        max_sources: int = 3,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Fetches relevant government data for a given document.

        Args:
            document_type: The type of document being analyzed.
            raw_text: The raw text extracted from the document.
            state_code: The state code for region-specific data.
            max_sources: The maximum number of sources to scrape.

        Returns:
            A dictionary containing the scraped data.
        """
        logger.info(
            "Executing SmartDataTool",
            document_type=document_type,
            state_code=state_code,
        )
        try:
            result: DataFetchingResult = await self.smart_data_agent.fetch_relevant_data(
                document_type=document_type,
                raw_text=raw_text,
                state_code=state_code,
                max_sources=max_sources,
            )
            return {
                "success": result.success,
                "scraped_data": result.scraped_data,
                "sources_successful": result.sources_successful,
                "error": result.error,
            }
        except Exception as e:
            logger.error("SmartDataTool execution failed", error=str(e), exc_info=True)
            return {"success": False, "error": str(e)}