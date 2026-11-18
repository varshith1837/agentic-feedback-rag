"""Strict contracts for extraction, plans, user filters, and checked outputs."""
from __future__ import annotations
from datetime import date
from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
from .config import ASPECTS

Aspect = Literal[
    'delivery_delay', 'product_not_received', 'damaged_item',
    'wrong_or_missing_item', 'product_quality', 'not_as_described',
    'seller_service', 'refund_or_return', 'packaging', 'price_value', 'other'
]
Polarity = Literal['negative', 'neutral', 'positive']

class AspectMention(BaseModel):
    model_config = ConfigDict(extra='forbid')
    aspect: Aspect
    polarity: Polarity
    severity: int = Field(ge=1, le=3)
    evidence_pt: str = Field(min_length=1, max_length=500)

class ReviewFacts(BaseModel):
    model_config = ConfigDict(extra='forbid')
    review_key: str
    english_text: str = Field(max_length=5000)
    overall_sentiment: Polarity
    aspects: list[AspectMention] = Field(max_length=20)

class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra='forbid')
    reviews: list[ReviewFacts]

class Filters(BaseModel):
    model_config = ConfigDict(extra='forbid')
    start_date: date | None = None
    end_date: date | None = None
    category: str | None = Field(default=None, max_length=150)
    state: str | None = Field(default=None, max_length=4)
    seller_id: str | None = Field(default=None, max_length=64)

    @model_validator(mode='after')
    def valid_range(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError('start_date cannot be later than end_date')
        return self

class ToolCall(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: Literal['list_aspects', 'revenue_at_risk', 'trend', 'compare', 'top_examples', 'semantic_search']
    group_by: Literal['aspect', 'category', 'seller', 'state', 'month'] = 'aspect'
    aspect: Aspect | None = None
    n: int = Field(default=5, ge=1, le=10)
    query: str = Field(default='', max_length=300)
    period_a_start: date | None = None
    period_a_end: date | None = None
    period_b_start: date | None = None
    period_b_end: date | None = None

    @model_validator(mode='after')
    def valid_periods(self):
        if self.name == 'compare':
            vals = (self.period_a_start, self.period_a_end, self.period_b_start, self.period_b_end)
            if not all(vals):
                raise ValueError('compare requires four period boundaries')
            if self.period_a_start > self.period_a_end or self.period_b_start > self.period_b_end:
                raise ValueError('compare period start exceeds end')
        if self.name == 'semantic_search' and not self.query.strip():
            raise ValueError('semantic_search requires a query')
        return self

class AgentPlan(BaseModel):
    model_config = ConfigDict(extra='forbid')
    calls: list[ToolCall] = Field(default_factory=list, min_length=0, max_length=4)
    refuse: bool = False
    reason: str = Field(default='', max_length=200)

class CheckedClaim(BaseModel):
    text: str
    status: Literal['supported', 'removed']
    reason: str
    tool_id: str | None = None
    citations: list[str] = Field(default_factory=list)
