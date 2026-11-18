"""Fail-closed grounding checks: SQL number, scoped citation, optional NLI.

NLI is applied to *textual paraphrases*, not to aggregate arithmetic. Numerical
claims are validated against deterministic SQL provenance instead.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .db import Database
from .tools import Tools
from .schemas import Filters

NUMERIC_FIELDS = frozenset({
  'sample_at_risk_brl','estimated_at_risk_brl','affected_orders_sample',
  'affected_orders_estimated','avg_review_score','total_revenue_brl',
  'sample_exposure_pct','estimated_exposure_pct',
  'period_a_brl','period_b_brl','delta_brl'
})

class NLIModel:
    """Lazy optional English 3-way NLI. Requires additional model download."""
    def __init__(self, model_name: str = 'cross-encoder/nli-deberta-v3-small'):
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
        except ImportError as e:
            raise RuntimeError('Install optional NLI dependencies: pip install -e ".[nli]"') from e
        self.torch=torch
        self.tokenizer=AutoTokenizer.from_pretrained(model_name)
        self.model=AutoModelForSequenceClassification.from_pretrained(model_name).eval()
        self.labels={int(i):str(v).lower() for i,v in self.model.config.id2label.items()}
        if not any('entail' in label for label in self.labels.values()):
            raise RuntimeError('NLI model labels do not declare entailment; refusing to guess class order')

    def entailment(self, premise: str, hypothesis: str) -> float:
        inputs=self.tokenizer(premise,hypothesis,return_tensors='pt',
                              truncation=True,max_length=512)
        with self.torch.no_grad():
            logits=self.model(**inputs).logits[0]
            scores=self.torch.softmax(logits,dim=-1)
        return max(float(scores[i]) for i,v in self.labels.items() if 'entail' in v)

class Verifier:
    def __init__(self, db: Database, nli=None, threshold: float = 0.70):
        self.db=db
        self.tools=Tools(db)
        self.nli=nli
        self.threshold=threshold

    @staticmethod
    def number_matches(row: dict, field: str, claimed: float,
                       tol: float = .011) -> bool:
        if field not in NUMERIC_FIELDS or field not in row:
            return False
        actual=row[field]
        return isinstance(actual,(int,float)) and abs(float(actual)-float(claimed)) <= tol

    def check_citation(self, citation_id: str, aspect: str | None = None,
                       filters: Filters | None = None, group_by: str | None = None,
                       group_key: str | None = None) -> dict | None:
        return self.tools.source(citation_id,aspect,filters,group_by,group_key)

    def check_paraphrase(self, text: str, citation_id: str,
                         aspect: str | None = None, filters: Filters | None = None) -> tuple[bool,str]:
        source=self.check_citation(citation_id,aspect,filters)
        if source is None:
            return False,'Citation absent or outside active filters'
        if not self.nli:
            return False,'Paraphrased review claims require optional NLI; no NLI configured'
        english=source.get('english_text') or ''
        if not english:
            return False,'No verified English translation for NLI'
        confidence=self.nli.entailment(english,text)
        return (confidence >= self.threshold,
                f'English NLI entailment={confidence:.3f} (threshold={self.threshold:.2f})')

    def verified_metric(self, row: dict, field: str, claimed: float,
                        citations: list[str], aspect: str | None,
                        filters: Filters | None, group_by: str,
                        group_key: str) -> tuple[bool,str,list[dict]]:
        if not self.number_matches(row,field,claimed):
            return False,'Claimed number differs from tool output',[]
        sources=[]
        for cid in citations:
            src=self.check_citation(cid,aspect,filters,group_by,group_key)
            if src is None:
                return False,f'Citation {cid} not found in this slice',[]
            sources.append(src)
        # Numeric evidence comes from SQL; cited reviews illustrate, not prove sums.
        return True,'Number confirmed by SQL; citations match filtered reviews',sources
