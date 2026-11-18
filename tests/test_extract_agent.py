from __future__ import annotations
import json
import pytest
from pydantic import ValidationError
from revenue_risk.extract import validate_batch,run_extract
from revenue_risk.schemas import ReviewFacts,AspectMention,Filters
from revenue_risk.agent import Agent,local_plan
from revenue_risk.verify import Verifier
from revenue_risk.llm import parse_json
from revenue_risk.graph import neighborhood


def test_evidence_quote_is_exact(toy):
    row=toy.one('SELECT review_key,original_text,review_score FROM reviews WHERE review_key=?',('r1::o1',))
    payload={'reviews':[{'review_key':'r1::o1','english_text':'Wrong',
                         'overall_sentiment':'negative','aspects':[{'aspect':'damaged_item',
                         'polarity':'negative','severity':3,'evidence_pt':'does not occur'}]}]}
    with pytest.raises(ValueError,match='Unsupported evidence'):
        validate_batch(payload,[row])

def test_invalid_llm_batch_fallback(toy):
    class BadClient:
        model='mock'
        def json(self,*args,**kwargs):return {'reviews':[]}
    stats=run_extract(toy,'groq',llm=BadClient(),limit=1,batch_size=1,pause=0,overwrite=True)
    assert stats['processed']==1 and stats['fallback']==1
    assert toy.one("SELECT COUNT(*) n FROM extractions WHERE status='fallback'")['n']==1


def test_number_check_rejects_hallucinated_amount(toy):
    row=__import__('revenue_risk.tools',fromlist=['Tools']).Tools(toy).revenue_at_risk('aspect','damaged_item')[0]
    v=Verifier(toy)
    assert v.number_matches(row,'sample_at_risk_brl',170.0)
    assert not v.number_matches(row,'sample_at_risk_brl',99999.0)
    assert not v.number_matches(row,'arbitrary_field',170.0)


def test_citation_outside_scope_rejected(toy):
    rows=toy.query('SELECT citation_id FROM reviews WHERE order_id=?',('o1',))
    assert Verifier(toy).check_citation(rows[0]['citation_id'],'damaged_item',Filters(state='RJ')) is None


def test_unsupported_paraphrase_fails_closed_without_nli(toy):
    cid=toy.one('SELECT citation_id FROM reviews WHERE order_id=?',('o1',))['citation_id']
    assert not Verifier(toy).check_paraphrase('Customer wants refund',cid,'damaged_item')[0]


def test_agent_refuses_out_of_scope(toy):
    assert Agent(toy).ask('What do customers think of the mobile app?')['refused']
    assert Agent(toy).ask('How many customers churned?')['refused']
    assert Agent(toy).ask('What happened in 2025?')['refused']


def test_agent_returns_verified_metrics(toy):
    answer=Agent(toy).ask('What revenue is at risk from damaged items?')
    assert not answer['refused']
    assert answer['claims'][0]['status']=='supported'
    assert answer['claims'][0]['numeric_value']==170.0
    assert answer['sources']


def test_graph_contains_multiple_edge_types(toy):
    graph=neighborhood(toy,'damaged_item',2)
    assert graph.number_of_edges()>0
    assert {'review','order','seller','aspect','category'} <= {v['kind'] for _,v in graph.nodes(data=True)}


def test_loose_json_fence_parsing():
    assert parse_json('```json\n{"a":1}\n```')=={'a':1}
    with pytest.raises(ValueError):parse_json('[1,2]')


def test_invalid_tool_params_rejected():
    from revenue_risk.schemas import ToolCall
    with pytest.raises(ValidationError):
        ToolCall(name='compare',aspect='damaged_item')
