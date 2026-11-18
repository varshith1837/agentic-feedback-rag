"""Ingest supplied Olist ZIP, protect order-value accounting, and sample reviews."""
from __future__ import annotations
import csv
import hashlib
import io
import math
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from .db import Database

TABLES = (
    'mentions','extractions','sample_reviews','reviews','order_items','orders',
    'llm_cache'
)

def records(z: zipfile.ZipFile, filename: str):
    with z.open(filename) as binary:
        stream = io.TextIOWrapper(binary, encoding='utf-8-sig', newline='')
        yield from csv.DictReader(stream)

def numeric(v: str | None) -> float:
    try:
        return float(v or 0)
    except ValueError:
        return 0.0

def iso_date(v: str | None) -> str | None:
    return v[:10] if v and len(v) >= 10 else None

def initialize(db: Database, reset: bool = True) -> None:
    if reset:
        for table in TABLES:
            db.execute(f'DROP TABLE IF EXISTS {table}')
    for sql in (
        '''CREATE TABLE IF NOT EXISTS orders (
          order_id VARCHAR PRIMARY KEY, purchase_date VARCHAR, purchase_month VARCHAR,
          order_status VARCHAR, customer_state VARCHAR, order_value DOUBLE,
          delivery_delay_days INTEGER)''',
        '''CREATE TABLE IF NOT EXISTS order_items (
          order_id VARCHAR, item_id INTEGER, product_id VARCHAR, seller_id VARCHAR,
          category VARCHAR, seller_state VARCHAR, item_value DOUBLE,
          PRIMARY KEY (order_id,item_id))''',
        '''CREATE TABLE IF NOT EXISTS reviews (
          review_key VARCHAR PRIMARY KEY, citation_id VARCHAR UNIQUE, review_id VARCHAR,
          order_id VARCHAR, review_score INTEGER, original_text VARCHAR, review_date VARCHAR)''',
        '''CREATE TABLE IF NOT EXISTS sample_reviews (
          review_key VARCHAR PRIMARY KEY, weight DOUBLE NOT NULL, stratum VARCHAR)''',
        '''CREATE TABLE IF NOT EXISTS extractions (
          review_key VARCHAR PRIMARY KEY, english_text VARCHAR, overall_sentiment VARCHAR,
          status VARCHAR, provider VARCHAR, model VARCHAR, processed_at VARCHAR)''',
        '''CREATE TABLE IF NOT EXISTS mentions (
          review_key VARCHAR, aspect VARCHAR, polarity VARCHAR, severity INTEGER,
          evidence_pt VARCHAR, PRIMARY KEY (review_key,aspect,polarity))''',
        '''CREATE TABLE IF NOT EXISTS llm_cache (
          request_hash VARCHAR PRIMARY KEY, response_json VARCHAR)''',
    ):
        db.execute(sql)
    for sql in (
        'CREATE INDEX IF NOT EXISTS idx_items_order ON order_items(order_id)',
        'CREATE INDEX IF NOT EXISTS idx_items_category ON order_items(category)',
        'CREATE INDEX IF NOT EXISTS idx_reviews_order ON reviews(order_id)',
        'CREATE INDEX IF NOT EXISTS idx_mentions_aspect ON mentions(aspect,polarity)',
        'CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(purchase_date)',
    ):
        db.execute(sql)
    db.commit()

def allocate_counts(groups: dict, capacity: int, weights: dict | None = None) -> dict:
    """Capacity-constrained largest-remainder allocation with nonzero coverage."""
    groups = {k: int(v) for k, v in groups.items() if v > 0}
    if capacity >= sum(groups.values()):
        return groups.copy()
    if capacity <= 0 or not groups:
        return {k: 0 for k in groups}
    weights = weights or {k: v for k, v in groups.items()}
    result = {k: 0 for k in groups}
    # When possible, include one element from every nonempty stratum.
    if capacity >= len(groups):
        for k in groups:
            result[k] = 1
        capacity -= len(groups)
    for _ in range(capacity):
        eligible = [k for k in groups if result[k] < groups[k]]
        if not eligible:
            break
        k = max(eligible, key=lambda x: (weights.get(x, 0) / (result[x] + 1), str(x)))
        result[k] += 1
    return result

def sample_reviews(db: Database, n: int = 4000, seed: int = 42) -> dict:
    rows = db.query('''SELECT r.review_key,r.review_score,substr(COALESCE(o.purchase_month,'unknown'),1,7) AS month
                      FROM reviews r JOIN orders o ON o.order_id=r.order_id''')
    by_score = defaultdict(list)
    for row in rows:
        by_score[row['review_score']].append(row)
    total = min(n, len(rows))
    # Oversample 1-2-star written reviews, retain contrast from higher scores.
    weights = {s: len(v) * (3.0 if s <= 2 else 1.0) for s, v in by_score.items()}
    quotas = allocate_counts({s: len(v) for s, v in by_score.items()}, total, weights)
    picked = []
    rng = random.Random(seed)
    for score, group in sorted(by_score.items()):
        by_month = defaultdict(list)
        for r in group:
            by_month[r['month']].append(r)
        month_counts = allocate_counts({m: len(v) for m, v in by_month.items()}, quotas[score])
        for month, members in sorted(by_month.items()):
            rng.shuffle(members)
            count = month_counts[month]
            if count:
                weight = len(members) / count
                for r in members[:count]:
                    picked.append((r['review_key'], weight, f'{score}:{month}'))
    db.execute('DELETE FROM sample_reviews')
    db.execute('DELETE FROM extractions')
    db.execute('DELETE FROM mentions')
    db.many('INSERT INTO sample_reviews(review_key,weight,stratum) VALUES (?,?,?)', picked)
    db.commit()
    return {'written_reviews': len(rows), 'sampled_reviews': len(picked), 'score_counts': dict(quotas),
            'sample_weights': 'score × purchase month expansion to written reviews only'}

def prepare(db: Database, data_zip: str | Path, sample_size: int = 4000, seed: int = 42) -> dict:
    path = Path(data_zip)
    if not path.exists():
        raise FileNotFoundError(f'Olist ZIP not found at {path}. Copy the supplied brazil.zip to data/.')
    expected = {
        'olist_order_payments_dataset.csv', 'olist_orders_dataset.csv',
        'olist_order_items_dataset.csv', 'olist_order_reviews_dataset.csv',
        'olist_products_dataset.csv', 'olist_customers_dataset.csv',
        'olist_sellers_dataset.csv', 'product_category_name_translation.csv'
    }
    with zipfile.ZipFile(path) as z:
        missing = expected - set(z.namelist())
        if missing:
            raise ValueError(f'Missing dataset files: {sorted(missing)}')
        translations = {r['product_category_name']: r['product_category_name_english']
                        for r in records(z, 'product_category_name_translation.csv')}
        products = {r['product_id']: translations.get(r['product_category_name'], r['product_category_name'] or 'unknown')
                    for r in records(z, 'olist_products_dataset.csv')}
        sellers = {r['seller_id']: r['seller_state'] for r in records(z, 'olist_sellers_dataset.csv')}
        customers = {r['customer_id']: r['customer_state'] for r in records(z, 'olist_customers_dataset.csv')}
        payments = defaultdict(float)
        for r in records(z, 'olist_order_payments_dataset.csv'):
            payments[r['order_id']] += numeric(r['payment_value'])
        initialize(db, reset=True)
        order_batch = []
        for r in records(z, 'olist_orders_dataset.csv'):
            delivered, estimated = iso_date(r['order_delivered_customer_date']), iso_date(r['order_estimated_delivery_date'])
            delay = None
            if delivered and estimated:
                from datetime import date
                delay = (date.fromisoformat(delivered)-date.fromisoformat(estimated)).days
            purchased = iso_date(r['order_purchase_timestamp'])
            order_batch.append((r['order_id'], purchased, purchased[:7] if purchased else None,
                                r['order_status'], customers.get(r['customer_id'], 'unknown'),
                                round(payments.get(r['order_id'], 0.0), 2), delay))
        db.many('INSERT INTO orders VALUES (?,?,?,?,?,?,?)', order_batch)
        del order_batch, payments, customers
        item_batch = []
        for r in records(z, 'olist_order_items_dataset.csv'):
            item_batch.append((r['order_id'], int(r['order_item_id']), r['product_id'],
                               r['seller_id'], products.get(r['product_id'], 'unknown'),
                               sellers.get(r['seller_id'], 'unknown'),
                               round(numeric(r['price'])+numeric(r['freight_value']), 2)))
        db.many('INSERT INTO order_items VALUES (?,?,?,?,?,?,?)', item_batch)
        del item_batch, products, sellers
        best = {}
        for r in records(z, 'olist_order_reviews_dataset.csv'):
            title, msg = (r['review_comment_title'] or '').strip(), (r['review_comment_message'] or '').strip()
            original = '\n'.join(v for v in (title, msg) if v)
            if not original:
                continue
            key = r['review_id'] + '::' + r['order_id']
            row = (key, 'R-' + hashlib.sha256(key.encode()).hexdigest()[:12], r['review_id'],
                   r['order_id'], int(r['review_score']), original, iso_date(r['review_creation_date']))
            if key not in best or len(original) > len(best[key][5]):
                best[key] = row
        # Keep reviews only when an order record exists. No synthetic order or dollar values.
        existing = {r['order_id'] for r in db.query('SELECT order_id FROM orders')}
        db.many('INSERT INTO reviews VALUES (?,?,?,?,?,?,?)', [r for r in best.values() if r[3] in existing])
        db.commit()
    counts = sample_reviews(db, n=sample_size, seed=seed)
    counts['orders'] = db.one('SELECT COUNT(*) n FROM orders')['n']
    counts['order_items'] = db.one('SELECT COUNT(*) n FROM order_items')['n']
    counts['total_paid_brl'] = round(db.one('SELECT SUM(order_value) s FROM orders')['s'] or 0.0,2)
    return counts
