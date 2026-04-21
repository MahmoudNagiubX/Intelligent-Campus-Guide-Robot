"""Conversation controller, response composition, and live runtime graph."""

__all__ = [
    "ConversationController",
    "NavigatorPipecatRuntime",
    "ResponseComposer",
]


def __getattr__(name: str):
    """Load heavier pipeline objects lazily to avoid import cycles."""
    if name == "ConversationController":
        from app.pipeline.controller import ConversationController

        return ConversationController
    if name == "NavigatorPipecatRuntime":
        from app.pipeline.pipecat_graph import NavigatorPipecatRuntime

        return NavigatorPipecatRuntime
    if name == "ResponseComposer":
        from app.pipeline.response_composer import ResponseComposer

        return ResponseComposer
    raise AttributeError(name)
