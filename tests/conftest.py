"""Minimal in-memory public-data-shaped fixture; never needs external downloads."""
from __future__ import annotations
import csv,io,zipfile
from pathlib import Path
import pytest
from revenue_risk.db import Database
from revenue_risk.prep import prepare
from revenue_risk.extract import persist
from revenue_risk.schemas import ReviewFacts,AspectMention

@pytest.fixture
def toy(tmp_path):
    zip_path=tmp_path/'tiny_olist.zip'
    specs={
     'olist_order_payments_dataset.csv':([ 'order_id','payment_value'],
         [['o1','100'],['o1','20'],['o2','50']]),
     'olist_orders_dataset.csv':(['order_id','customer_id','order_status','order_purchase_timestamp','order_delivered_customer_date','order_estimated_delivery_date'],
         [['o1','c1','delivered','2018-01-15 10:00:00','2018-02-01','2018-01-30'],
          ['o2','c2','delivered','2018-02-01','2018-02-07','2018-02-08']]),
     'olist_order_items_dataset.csv':(['order_id','order_item_id','product_id','seller_id','price','freight_value'],
         [['o1','1','p1','s1','40','5'],['o1','2','p2','s2','65','10'],['o2','1','p1','s1','45','5']]),
     'olist_order_reviews_dataset.csv':(['review_id','order_id','review_score','review_comment_title','review_comment_message','review_creation_date'],
         [['r1','o1','1','','O produto chegou quebrado, com atraso.','2018-02-01'],
          ['r1','o1','1','','O produto chegou quebrado, com atraso.','2018-02-01'],
          ['r2','o2','2','','Produto chegou quebrado.','2018-02-08']]),
     'olist_products_dataset.csv':(['product_id','product_category_name'],[['p1','beleza_saude'],['p2','livros']]),
     'product_category_name_translation.csv':(['product_category_name','product_category_name_english'],
                                             [['beleza_saude','health_beauty'],['livros','books']]),
     'olist_sellers_dataset.csv':(['seller_id','seller_state'],[['s1','SP'],['s2','RJ']]),
     'olist_customers_dataset.csv':(['customer_id','customer_state'],[['c1','SP'],['c2','RJ']]),
    }
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,(columns,rows) in specs.items():
            f=io.StringIO();w=csv.writer(f);w.writerow(columns);w.writerows(rows)
            z.writestr(name,f.getvalue())
    db=Database(tmp_path/'test.sqlite')
    prepare(db,zip_path,sample_size=10)
    persist(db,[ReviewFacts(review_key='r1::o1',english_text='Item arrived damaged and late',
                     overall_sentiment='negative',aspects=[
                     AspectMention(aspect='damaged_item',polarity='negative',severity=3,
                                   evidence_pt='quebrado'),
                     AspectMention(aspect='delivery_delay',polarity='negative',severity=2,evidence_pt='atraso')]),
                ReviewFacts(review_key='r2::o2',english_text='Item arrived damaged',
                     overall_sentiment='negative',aspects=[
                     AspectMention(aspect='damaged_item',polarity='negative',severity=2,
                                   evidence_pt='quebrado')])], 'llm','fixture','fixture')
    yield db
    db.close()
