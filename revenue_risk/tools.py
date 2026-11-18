"""Allowlisted, parameterized SQL tools. No LLM-generated SQL is ever executed."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from .config import ASPECTS
from .db import Database
from .schemas import Filters, ToolCall

class ToolError(ValueError):
    pass

def _order_filter(f: Filters, alias: str = 'o') -> tuple[str,list]:
    cond, args = [], []
    if f.start_date:
        cond.append(f'{alias}.purchase_date >= ?');args.append(f.start_date.isoformat())
    if f.end_date:
        cond.append(f'{alias}.purchase_date <= ?');args.append(f.end_date.isoformat())
    if f.state:
        cond.append(f'{alias}.customer_state = ?');args.append(f.state.upper())
    return (' AND '.join(cond) if cond else '1=1'),args

def _item_filter(f: Filters, alias: str = 'i') -> tuple[str,list]:
    cond,args=[],[]
    if f.category:
        cond.append(f'{alias}.category = ?');args.append(f.category)
    if f.seller_id:
        cond.append(f'{alias}.seller_id = ?');args.append(f.seller_id)
    return (' AND '.join(cond) if cond else '1=1'),args

def _risk_query(group_by: str, aspect: str | None, filters: Filters) -> tuple[str,list]:
    if group_by not in ('aspect','category','seller','state','month'):
        raise ToolError('group_by not permitted')
    if aspect is not None and aspect not in ASPECTS:
        raise ToolError('Unknown aspect')
    ow,oa = _order_filter(filters)
    iw,ia = _item_filter(filters)
    restrict_items = bool(filters.category or filters.seller_id)
    # Dates refer to purchase date; seller/category exposure uses item price+freight.
    # Overall/state/month exposure uses order payments unless narrowed to items.
    if group_by in ('category','seller'):
        key = 'category' if group_by=='category' else 'seller_id'
        base = f'''SELECT i.order_id, i.{key} AS group_key, SUM(i.item_value) AS value
                   FROM item_scope i GROUP BY i.order_id,i.{key}'''
    elif group_by in ('state','month'):
        key = 'customer_state' if group_by == 'state' else 'purchase_month'
        if restrict_items:
            base = f'''SELECT o.order_id,o.{key} AS group_key,SUM(i.item_value) AS value
                        FROM order_scope o JOIN item_scope i ON i.order_id=o.order_id
                        GROUP BY o.order_id,o.{key}'''
        else:
            base = f'''SELECT o.order_id,o.{key} AS group_key,o.order_value AS value
                        FROM order_scope o'''
    else:
        if restrict_items:
            base = '''SELECT o.order_id,'all' AS group_key,SUM(i.item_value) AS value
                      FROM order_scope o JOIN item_scope i ON i.order_id=o.order_id
                      GROUP BY o.order_id'''
        else:
            base = '''SELECT o.order_id,'all' AS group_key,o.order_value AS value FROM order_scope o'''
    # Risk orders are unique per order/aspect before financial joins.
    # Score×month weight expands to population of WRITTEN reviews only.
    aspect_clause = 'AND m.aspect=?' if aspect else ''
    aspect_args = [aspect] if aspect else []
    risks = f'''SELECT r.order_id,m.aspect,MAX(s.weight) AS weight,MIN(r.review_score) AS score
                FROM reviews r JOIN sample_reviews s ON s.review_key=r.review_key
                JOIN extractions e ON e.review_key=r.review_key
                JOIN mentions m ON m.review_key=r.review_key
                JOIN order_scope o ON o.order_id=r.order_id
                WHERE e.status IN ('llm','rules_demo') AND m.polarity='negative' {aspect_clause}
                GROUP BY r.order_id,m.aspect'''
    if group_by == 'aspect':
        risk_group = '''SELECT order_id,aspect AS group_key,MAX(weight) AS weight,MIN(score) AS score
                        FROM risk_raw GROUP BY order_id,aspect'''
        risk_join = 'b.order_id=r.order_id'
        risk_group_key = 'r.group_key'
        totals_key = 'r.group_key'
        denominator = '''SELECT SUM(value) AS total FROM base'''
        select_sql = f'''SELECT {risk_group_key} AS group_key,
            ROUND(SUM(b.value),2) AS sample_at_risk_brl,
            ROUND(SUM(b.value*r.weight),2) AS estimated_at_risk_brl,
            COUNT(DISTINCT r.order_id) AS affected_orders_sample,
            ROUND(SUM(r.weight),2) AS affected_orders_estimated,
            ROUND(AVG(r.score),2) AS avg_review_score,
            ROUND((SELECT total FROM denominator),2) AS total_revenue_brl
            FROM risk_group r JOIN base b ON {risk_join}
            GROUP BY {risk_group_key} ORDER BY sample_at_risk_brl DESC,group_key LIMIT 100'''
    else:
        risk_group = '''SELECT order_id,'any' AS group_key,MAX(weight) AS weight,MIN(score) AS score
                        FROM risk_raw GROUP BY order_id'''
        denominator = '''SELECT group_key,SUM(value) AS total FROM base GROUP BY group_key'''
        select_sql = '''SELECT b.group_key AS group_key,
            ROUND(SUM(b.value),2) AS sample_at_risk_brl,
            ROUND(SUM(b.value*r.weight),2) AS estimated_at_risk_brl,
            COUNT(DISTINCT r.order_id) AS affected_orders_sample,
            ROUND(SUM(r.weight),2) AS affected_orders_estimated,
            ROUND(AVG(r.score),2) AS avg_review_score,
            ROUND(d.total,2) AS total_revenue_brl
            FROM risk_group r JOIN base b ON b.order_id=r.order_id
            JOIN denominator d ON d.group_key=b.group_key
            GROUP BY b.group_key,d.total ORDER BY sample_at_risk_brl DESC,b.group_key LIMIT 100'''
    sql = f'''WITH order_scope AS (SELECT * FROM orders o WHERE {ow}),
         item_scope AS (SELECT i.* FROM order_items i JOIN order_scope o ON o.order_id=i.order_id WHERE {iw}),
         base AS ({base}), risk_raw AS ({risks}), risk_group AS ({risk_group}),
         denominator AS ({denominator}) {select_sql}'''
    return sql, oa+ia+aspect_args

class Tools:
    def __init__(self, db: Database):
        self.db = db

    def list_aspects(self) -> list[dict]:
        observed = {r['aspect']:r['count'] for r in self.db.query('''
           SELECT m.aspect,COUNT(DISTINCT m.review_key) AS count FROM mentions m
           JOIN extractions e ON e.review_key=m.review_key
           WHERE m.polarity='negative' AND e.status IN ('llm','rules_demo')
           GROUP BY m.aspect''')}
        return [{'aspect':name,'negative_mentions':observed.get(name,0)} for name in ASPECTS]

    def revenue_at_risk(self, group_by: str = 'aspect', aspect: str | None = None,
                        filters: Filters | None = None) -> list[dict]:
        sql,args = _risk_query(group_by,aspect,filters or Filters())
        rows=self.db.query(sql,args)
        for r in rows:
            denom = r['total_revenue_brl'] or 0
            r['sample_exposure_pct']=round(100*r['sample_at_risk_brl']/denom,2) if denom > 0 else None
            r['estimated_exposure_pct']=round(100*r['estimated_at_risk_brl']/denom,2) if denom > 0 else None
            r['group_by']=group_by
            r['aspect_filter']=aspect
            r['unit']='BRL'
        return rows

    def trend(self, aspect: str | None = None, filters: Filters | None = None) -> list[dict]:
        return sorted(self.revenue_at_risk('month',aspect,filters), key=lambda r: r['group_key'])

    def compare(self, aspect: str | None, periods: tuple[str,str,str,str],
                filters: Filters | None = None) -> list[dict]:
        base=(filters or Filters()).model_dump(mode='json')
        fa=Filters.model_validate({**base,'start_date':periods[0],'end_date':periods[1]})
        fb=Filters.model_validate({**base,'start_date':periods[2],'end_date':periods[3]})
        a=self.revenue_at_risk('aspect',aspect,fa)
        b=self.revenue_at_risk('aspect',aspect,fb)
        by_a={r['group_key']:r for r in a};by_b={r['group_key']:r for r in b}
        out=[]
        for key in sorted(set(by_a)|set(by_b)):
            va=by_a.get(key,{}).get('sample_at_risk_brl',0)
            vb=by_b.get(key,{}).get('sample_at_risk_brl',0)
            out.append({'group_key':key,'period_a_brl':va,'period_b_brl':vb,
                        'delta_brl':round(vb-va,2),'period_a':f'{periods[0]}–{periods[1]}',
                        'period_b':f'{periods[2]}–{periods[3]}','unit':'BRL'})
        return out

    def top_examples(self, aspect: str | None = None, filters: Filters | None = None,
                     n: int = 5, group_by: str | None = None,
                     group_key: str | None = None) -> list[dict]:
        if aspect and aspect not in ASPECTS:
            raise ToolError('Unknown aspect')
        f = filters or Filters()
        ow, oa = _order_filter(f)
        iw, ia = _item_filter(f)
        cond=[];params=list(oa)
        if aspect:
            cond.append('m.aspect=?');params.append(aspect)
        if group_by == 'aspect' and group_key:
            cond.append('m.aspect=?');params.append(group_key)
        if group_by == 'state' and group_key:
            cond.append('o.customer_state=?');params.append(group_key)
        if group_by == 'month' and group_key:
            cond.append('o.purchase_month=?');params.append(group_key)
        item_conditions=[];item_params=[]
        if f.category:
            item_conditions.append('i.category=?');item_params.append(f.category)
        if f.seller_id:
            item_conditions.append('i.seller_id=?');item_params.append(f.seller_id)
        if group_by == 'category' and group_key:
            item_conditions.append('i.category=?');item_params.append(group_key)
        if group_by == 'seller' and group_key:
            item_conditions.append('i.seller_id=?');item_params.append(group_key)
        if item_conditions:
            cond.append('EXISTS (SELECT 1 FROM order_items i WHERE i.order_id=r.order_id AND '+
                        ' AND '.join(item_conditions)+')')
            params.extend(item_params)
        where=' AND '.join(cond) if cond else '1=1'
        sql=f'''SELECT DISTINCT r.citation_id,r.review_key,r.order_id,r.review_score,
                  r.original_text,e.english_text,o.order_value,o.customer_state,o.purchase_date,
                  m.aspect,m.polarity,m.evidence_pt,e.status AS extraction_status
                  FROM reviews r JOIN orders o ON o.order_id=r.order_id
                  JOIN sample_reviews s ON s.review_key=r.review_key
                  JOIN extractions e ON e.review_key=r.review_key
                  JOIN mentions m ON m.review_key=r.review_key
                  WHERE {ow} AND {where} AND m.polarity='negative'
                    AND e.status IN ('llm','rules_demo')
                  ORDER BY r.review_score ASC,r.citation_id LIMIT ?'''
        params.append(min(max(n,1),10))
        return self.db.query(sql,params)

    def source(self, citation_id: str, aspect: str | None = None,
               filters: Filters | None = None, group_by: str | None = None,
               group_key: str | None = None) -> dict | None:
        if not citation_id.startswith('R-'):
            return None
        # Citation must appear in eligible examples for the same filtered slice.
        candidates=self.top_examples(aspect,filters,10,group_by,group_key)
        candidate=next((r for r in candidates if r['citation_id']==citation_id),None)
        if candidate:
            return candidate
        # Check beyond top-10 without widening filters: allow only verified query results.
        # The default agent only cites its own top-10 tool results.
        return None

    def semantic_search(self, query: str, n: int = 5) -> list[dict]:
        """Optional multilingual E5 semantic retrieval (CPU, model download required)."""
        if not query.strip():
            raise ToolError('Query cannot be empty')
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ToolError('Semantic search optional: pip install -e ".[semantic]"') from e
        rows=self.db.query('''SELECT r.citation_id,r.review_key,r.original_text,r.review_score,
             e.english_text,r.order_id FROM reviews r JOIN sample_reviews s ON s.review_key=r.review_key
             LEFT JOIN extractions e ON e.review_key=r.review_key LIMIT 8000''')
        if not rows:
            return []
        model=SentenceTransformer('intfloat/multilingual-e5-small')
        texts=['passage: '+r['original_text'][:600] for r in rows]
        emb=model.encode(texts,normalize_embeddings=True,show_progress_bar=False)
        q=model.encode(['query: '+query],normalize_embeddings=True)[0]
        sim=emb @ q
        order=np.argsort(-sim)[:min(10,max(1,n))]
        return [dict(rows[int(i)],similarity=float(sim[i])) for i in order]

    def dispatch(self, call: ToolCall, filters: Filters | None = None) -> list[dict]:
        f=filters or Filters()
        if call.name=='list_aspects': return self.list_aspects()
        if call.name=='revenue_at_risk':return self.revenue_at_risk(call.group_by,call.aspect,f)
        if call.name=='trend':return self.trend(call.aspect,f)
        if call.name=='top_examples':return self.top_examples(call.aspect,f,call.n)
        if call.name=='semantic_search':return self.semantic_search(call.query,call.n)
        if call.name=='compare':
            dates=(call.period_a_start,call.period_a_end,call.period_b_start,call.period_b_end)
            if not all(dates):raise ToolError('compare requires both periods')
            return self.compare(call.aspect,tuple(d.isoformat() for d in dates),f)
        raise ToolError('Unknown tool')
