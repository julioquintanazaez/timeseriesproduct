from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from enum import Enum

class Sentiment(str, Enum):
    pos = "pos"
    neg = "neg"
    neu = "neu"


class BridgeOutput(BaseModel):
    community_A: int
    community_B: int
    bridging_comment_ids: List[str]
    total_connections: int
    avg_weight: float


class AnalysisOutput(BaseModel):
    num_comments: int
    num_edges: int
    global_strength: float
    sentiment: str
    sentiment_type: Sentiment
    center_comment_id: Optional[str]
    communities: Dict[int, BridgeOutput, str, float]
    bridges_between_communities: List[BridgeOutput]