# Copyright (c) Stanford University.
# Licensed under the MIT License.
"""LangGraph implementation of the core STORM pipeline.

This module defines ``STORMGraph`` which mimics the four stage
pipeline implemented by :class:`STORMWikiRunner` using ``langgraph``.
Each STORM stage is wrapped as a node in a LangGraph ``StateGraph``
so the pipeline can be executed as a single graph.

The class is meant as a lightweight prototype demonstrating how the
existing modular design based on ``dspy`` can be translated to
``langgraph`` while preserving the same public API for the runner.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from langgraph.graph import StateGraph, END

from .storm_wiki.modules.article_generation import StormArticleGenerationModule
from .storm_wiki.modules.article_polish import StormArticlePolishingModule
from .storm_wiki.modules.knowledge_curation import StormKnowledgeCurationModule
from .storm_wiki.modules.outline_generation import StormOutlineGenerationModule
from .storm_wiki.modules.callback import BaseCallbackHandler
from .storm_wiki.modules.storm_dataclass import (
    StormArticle,
    StormInformationTable,
)


class STORMGraph:
    """Simple LangGraph wrapper around the four STORM modules."""

    def __init__(
        self,
        knowledge_curation: StormKnowledgeCurationModule,
        outline_generation: StormOutlineGenerationModule,
        article_generation: StormArticleGenerationModule,
        article_polishing: StormArticlePolishingModule,
    ) -> None:
        self.kc = knowledge_curation
        self.og = outline_generation
        self.ag = article_generation
        self.ap = article_polishing
        self._graph = self._build_graph()

    # ------------------------------------------------------------------
    # graph construction
    def _build_graph(self):
        workflow: StateGraph = StateGraph(
            {
                "topic": str,
                "ground_truth_url": str,
                "callback_handler": Optional[BaseCallbackHandler],
                "information_table": Optional[StormInformationTable],
                "outline": Optional[StormArticle],
                "draft_article": Optional[StormArticle],
                "article": Optional[StormArticle],
            }
        )

        workflow.add_node("curate", self._curate)
        workflow.add_node("outline", self._outline)
        workflow.add_node("article", self._article)
        workflow.add_node("polish", self._polish)

        workflow.add_edge("curate", "outline")
        workflow.add_edge("outline", "article")
        workflow.add_edge("article", "polish")
        workflow.add_edge("polish", END)
        workflow.set_entry_point("curate")
        return workflow.compile()

    # ------------------------------------------------------------------
    # node implementations
    def _curate(self, state: Dict[str, Any]):
        handler = state.get("callback_handler")
        table = self.kc.research(
            topic=state["topic"],
            ground_truth_url=state.get("ground_truth_url", ""),
            callback_handler=handler,
            max_perspective=getattr(self.kc, "max_perspective", 0),
            disable_perspective=False,
        )
        return {"information_table": table}

    def _outline(self, state: Dict[str, Any]):
        handler = state.get("callback_handler")
        outline = self.og.generate_outline(
            topic=state["topic"],
            information_table=state["information_table"],
            callback_handler=handler,
        )
        return {"outline": outline}

    def _article(self, state: Dict[str, Any]):
        handler = state.get("callback_handler")
        draft = self.ag.generate_article(
            topic=state["topic"],
            information_table=state["information_table"],
            article_with_outline=state["outline"],
            callback_handler=handler,
        )
        return {"draft_article": draft}

    def _polish(self, state: Dict[str, Any]):
        handler = state.get("callback_handler")
        article = self.ap.polish_article(
            topic=state["topic"],
            draft_article=state["draft_article"],
        )
        return {"article": article}

    # ------------------------------------------------------------------
    # public API
    def run(
        self,
        topic: str,
        *,
        ground_truth_url: str = "",
        callback_handler: Optional[BaseCallbackHandler] = None,
    ) -> StormArticle:
        """Run the full STORM pipeline as a LangGraph."""

        state = {
            "topic": topic,
            "ground_truth_url": ground_truth_url,
            "callback_handler": callback_handler,
        }
        result = self._graph.invoke(state)
        return result["article"]
