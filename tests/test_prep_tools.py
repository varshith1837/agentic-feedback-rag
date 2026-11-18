from __future__ import annotations
import pytest
from pydantic import ValidationError
from revenue_risk.prep import allocate_counts
from revenue_risk.tools import Tools,ToolError
from revenue_risk.schemas import Filters,ToolCall

def test_payment_aggregation_and_review_dedup(toy):
    assert toy.one('SELECT order_value FROM orders WHERE order_id=?',('o1',))['order_value']==120
    assert toy.one('SELECT COUNT(*) n FROM reviews')['n']==2
    assert toy.one('SELECT delivery_delay_days FROM orders WHERE order_id=?',('o1',))['delivery_delay_days']==2

def test_order_level_unique_per_aspect(toy):
    rows=Tools(toy).revenue_at_risk('aspect','damaged_item')
    assert len(rows)==1
    assert rows[0]['sample_at_risk_brl']==170.0
    assert rows[0]['affected_orders_sample']==2
    assert rows[0]['total_revenue_brl']==170.0
    assert rows[0]['sample_exposure_pct']==100.0

def test_seller_and_category_item_allocation(toy):
    seller={r['group_key']:r['sample_at_risk_brl'] for r in Tools(toy).revenue_at_risk('seller','damaged_item')}
    assert seller=={'s1':95.0,'s2':75.0}
    cats={r['group_key']:r['sample_at_risk_brl'] for r in Tools(toy).revenue_at_risk('category','damaged_item')}
    assert cats=={'health_beauty':95.0,'books':75.0}

def test_multi_aspect_count_does_not_double_count_order(toy):
    states=Tools(toy).revenue_at_risk('state')
    assert sum(r['sample_at_risk_brl'] for r in states)==170.0

def test_score_month_weights_toy(toy):
    assert all(r['weight']==1 for r in toy.query('SELECT weight FROM sample_reviews'))
    assert allocate_counts({'a':10,'b':20},30)=={'a':10,'b':20}
    chosen=allocate_counts({'a':10,'b':20},5)
    assert sum(chosen.values())==5 and all(x>0 for x in chosen.values())

def test_validation_invalid_aspect_and_dates(toy):
    with pytest.raises(ValidationError):
        ToolCall(name='revenue_at_risk',aspect='made_up')
    with pytest.raises(ValidationError):
        Filters(start_date='2018-05-02',end_date='2018-05-01')
    with pytest.raises(ToolError):
        Tools(toy).revenue_at_risk('drop table')

def test_filtered_citations(toy):
    t=Tools(toy)
    examples=t.top_examples('damaged_item',Filters(state='SP'))
    assert len(examples)==1
    assert t.source(examples[0]['citation_id'],'damaged_item',Filters(state='SP')) is not None
    assert t.source(examples[0]['citation_id'],'damaged_item',Filters(state='RJ')) is None

def test_trend_and_compare(toy):
    trend=Tools(toy).trend('damaged_item')
    assert [r['group_key'] for r in trend]==['2018-01','2018-02']
    comp=Tools(toy).compare('damaged_item',('2018-01-01','2018-01-31','2018-02-01','2018-02-28'))
    assert comp[0]['period_a_brl']==120.0 and comp[0]['period_b_brl']==50.0
