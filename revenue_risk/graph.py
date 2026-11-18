"""Small NetworkX neighborhood drawing from relational graph edges."""
from __future__ import annotations
import networkx as nx
from .db import Database
from .tools import Tools


def neighborhood(db: Database, aspect: str='delivery_delay', max_reviews: int=12) -> nx.Graph:
    G=nx.Graph()
    G.add_node('aspect:'+aspect,label=aspect.replace('_',' '),kind='aspect')
    reviews=Tools(db).top_examples(aspect,n=min(max_reviews,10))
    for r in reviews:
        review='review:'+r['citation_id']
        order='order:'+r['order_id'][:10]
        G.add_node(review,label=r['citation_id'][:9],kind='review')
        G.add_node(order,label='order '+r['order_id'][:7],kind='order')
        G.add_edge('aspect:'+aspect,review,relation='mentions')
        G.add_edge(review,order,relation='about')
        items=db.query('''SELECT DISTINCT seller_id,category FROM order_items
                          WHERE order_id=? LIMIT 3''',(r['order_id'],))
        for item in items:
            seller='seller:'+item['seller_id'][:12]
            category='category:'+item['category']
            G.add_node(seller,label='seller '+item['seller_id'][:7],kind='seller')
            G.add_node(category,label=item['category'].replace('_',' ')[:22],kind='category')
            G.add_edge(order,seller,relation='fulfilled_by')
            G.add_edge(seller,category,relation='sells_category')
    return G
