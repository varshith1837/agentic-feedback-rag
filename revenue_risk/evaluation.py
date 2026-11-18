"""30 predeclared questions, independently calculated Python golden aggregates.

Do not interpret automated citation eligibility as human citation precision.
"""
from __future__ import annotations
import csv
from datetime import date
import json
import time
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from .agent import Agent
from .config import ROOT
from .db import Database


@lru_cache(maxsize=2)
def _gold_base(db: Database):
    orders=db.query('SELECT order_id,order_value,customer_state,purchase_date,purchase_month FROM orders')
    mentions=db.query('''SELECT DISTINCT r.order_id,m.aspect FROM reviews r
             JOIN sample_reviews s ON s.review_key=r.review_key
             JOIN extractions e ON e.review_key=r.review_key
             JOIN mentions m ON m.review_key=r.review_key
             WHERE m.polarity='negative' AND e.status IN ('llm','rules_demo')''')
    items=db.query('SELECT order_id,category,seller_id,item_value FROM order_items')
    return orders,mentions,items

def reference_gold(db: Database, group_by: str, aspect: str | None,
                   start_date: str | None=None, end_date: str | None=None) -> dict[str,float]:
    """Independent calculation directly from raw relational rows (not tools SQL)."""
    orders,mentions,items = _gold_base(db)
    allowed={r['order_id']:r for r in orders if (not start_date or r['purchase_date'] and r['purchase_date']>=start_date)
             and (not end_date or r['purchase_date'] and r['purchase_date']<=end_date)}
    exposure=defaultdict(float)
    if group_by=='aspect':
        for m in set((r['order_id'],r['aspect']) for r in mentions):
            if (aspect is None or m[1]==aspect) and m[0] in allowed:
                exposure[m[1]]+=allowed[m[0]]['order_value']
    else:
        bad={r['order_id'] for r in mentions if aspect is None or r['aspect']==aspect}
        if group_by in ('category','seller'):
            for i in items:
                if i['order_id'] in allowed and i['order_id'] in bad:
                    exposure[i['category' if group_by=='category' else 'seller_id']]+=i['item_value']
        else:
            field='customer_state' if group_by=='state' else 'purchase_month'
            for oid in bad & set(allowed):
                r=allowed[oid]
                exposure[r[field]]+=r['order_value']
    return {k:round(v+1e-8,2) for k,v in exposure.items()}


def create_label_template(db: Database, file: Path, n: int = 60):
    rows=db.query('''SELECT r.review_key,r.review_score,r.original_text
           FROM sample_reviews s JOIN reviews r ON r.review_key=s.review_key
           ORDER BY r.review_key LIMIT ?''',(n,))
    file.parent.mkdir(parents=True,exist_ok=True)
    with file.open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f)
        w.writerow(['review_key','review_score','original_portuguese','human_gold_aspects_semicolon',
                    'human_sentiment','reviewer_initials','notes'])
        for r in rows:
            w.writerow([r['review_key'],r['review_score'],r['original_text'],'','','',''])


def run_evaluation(db: Database, provider: str='rules', model: str | None=None,
                   output: str='results/evaluation.json') -> dict:
    questions=json.loads((ROOT/'eval/questions.json').read_text(encoding='utf-8'))
    assert len(questions)==30
    agent=Agent(db,provider,model)
    log=[]
    for q in questions:
        result=agent.ask(q['question'],max_rows=5)
        selected=result['calls'][0]['tool'] if result['calls'] else None
        correct_tool=selected==q['tool']
        if q['kind']=='refusal':
            correct=bool(result['refused']);numeric=None
        else:
            correct=correct_tool;numeric=None
            if q['kind'] in ('aggregation','trend','compare') and result['tables']:
                observed=result['tables'][0]['rows']
                f=result['scope']
                if q['kind']=='compare':
                    year=2017 if '2017' in q['question'] else 2018
                    pa=reference_gold(db,'aspect',q['aspect'],f'{year}-01-01',f'{year}-06-30')
                    pb=reference_gold(db,'aspect',q['aspect'],f'{year}-07-01',f'{year}-12-31')
                    expected=round(pb.get(q['aspect'],0)-pa.get(q['aspect'],0),2)
                    numeric=all(abs(r['delta_brl']-expected)<.02 for r in observed)
                else:
                    group=q.get('group_by','month' if q['kind']=='trend' else 'aspect')
                    expected=reference_gold(db,group,q.get('aspect'),f['start_date'],f['end_date'])
                    numeric=bool(observed) and all(
                        abs(r['sample_at_risk_brl']-expected.get(r['group_key'],0))<.02
                        for r in observed[:10])
        citations=[s['citation_id'] for s in result['sources']]
        valid_citations=all(db.one('SELECT 1 n FROM reviews WHERE citation_id=?',(cid,)) for cid in citations)
        log.append({'id':q['id'],'kind':q['kind'],'question':q['question'],
                    'expected_tool':q['tool'],'selected_tool':selected,
                    'tool_correct':correct_tool,'refusal_correct':correct if q['kind']=='refusal' else None,
                    'numeric_exact':numeric,'citations_eligible':valid_citations,
                    'citation_count':len(citations),'refused':result['refused'],
                    'elapsed_seconds':result.get('duration_seconds',0),'llm_calls':result.get('llm_calls',0)})
    with_numeric=[r for r in log if r['numeric_exact'] is not None]
    traps=[r for r in log if r['kind']=='refusal']
    score={
        'metadata':{'evaluation_question_count':len(questions),'provider':provider,'model':model,
                    'annotation_source':db.query('SELECT status,COUNT(*) n FROM extractions GROUP BY status'),
                    'warning':'Offline rules_demo results are NOT an LLM assessment; no human labels exist.'},
        'tool_selection_accuracy':round(sum(r['tool_correct'] for r in log)/len(log),4),
        'numeric_exactness_on_answered':round(sum(r['numeric_exact'] for r in with_numeric)/len(with_numeric),4) if with_numeric else None,
        'numeric_questions_checked':len(with_numeric),
        'refusal_accuracy':round(sum(r['refusal_correct'] for r in traps)/len(traps),4),
        'citation_eligibility_rate':round(sum(r['citations_eligible'] for r in log)/len(log),4),
        'human_citation_precision':'NOT MEASURED – hand-check about 10 answers',
        'human_faithfulness_with_without_checker':'NOT MEASURED – requires independent labeling and an ablation',
        'extraction_precision_recall':'NOT MEASURED – label eval/hand_label_60.csv first',
        'total_llm_calls':sum(r['llm_calls'] for r in log),
        'mean_seconds_per_question':round(sum(r['elapsed_seconds'] for r in log)/len(log),3),
        'question_results':log
    }
    out=Path(output)
    if not out.is_absolute(): out=ROOT/out
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(score,ensure_ascii=False,indent=2),encoding='utf-8')
    create_label_template(db,ROOT/'eval/hand_label_60.csv')
    return {k:v for k,v in score.items() if k!='question_results'} | {'results_path':str(out)}
