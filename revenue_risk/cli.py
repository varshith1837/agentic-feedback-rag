"""Simple command line for dataset preparation, extraction, Q&A and evaluation."""
from __future__ import annotations
import argparse
import json
import sys
from .config import DB_PATH, DATA_ZIP, ASPECTS
from .db import Database,require_prepared
from .prep import prepare
from .extract import run_extract
from .schemas import Filters
from .agent import Agent


def build_parser():
    p=argparse.ArgumentParser(prog='revenue-risk',description='Olist feedback graph and cited analytics')
    p.add_argument('--db',default=str(DB_PATH),help='.sqlite or .duckdb database path')
    subs=p.add_subparsers(dest='cmd',required=True)
    prep=subs.add_parser('prepare',help='Load full Olist ZIP and choose stratified sample')
    prep.add_argument('--zip',default=str(DATA_ZIP))
    prep.add_argument('--sample-size',type=int,default=4000)
    prep.add_argument('--seed',type=int,default=42)
    extract=subs.add_parser('extract',help='Extract problems from written reviews (resumable)')
    extract.add_argument('--provider',choices=['rules','groq','ollama'],default='rules')
    extract.add_argument('--model')
    extract.add_argument('--limit',type=int)
    extract.add_argument('--batch-size',type=int,default=8)
    extract.add_argument('--pause',type=float,default=.3)
    extract.add_argument('--overwrite',action='store_true',help='Re-extract completed LLM reviews')
    ask=subs.add_parser('ask',help='Ask a business question')
    ask.add_argument('question')
    ask.add_argument('--provider',choices=['rules','groq','ollama'],default='rules')
    ask.add_argument('--model')
    ask.add_argument('--start-date');ask.add_argument('--end-date')
    ask.add_argument('--category');ask.add_argument('--state');ask.add_argument('--seller-id')
    ask.add_argument('--no-checker',action='store_true')
    ask.add_argument('--enable-nli',action='store_true',help='Load optional English NLI model for review paraphrases')
    evalp=subs.add_parser('evaluate',help='Run 30-question validation suite')
    evalp.add_argument('--provider',choices=['rules','groq','ollama'],default='rules')
    evalp.add_argument('--model')
    evalp.add_argument('--out',default='results/evaluation.json')
    subs.add_parser('status',help='Dataset/coverage overview')
    return p

def main(argv=None):
    args=build_parser().parse_args(argv)
    try:
        with Database(args.db) as db:
            if args.cmd=='prepare':
                stats=prepare(db,args.zip,args.sample_size,args.seed)
            else:
                require_prepared(db)
                if args.cmd=='extract':
                    stats=run_extract(db,args.provider,args.model,args.limit,args.batch_size,
                                      args.pause,args.overwrite)
                elif args.cmd=='ask':
                    f=Filters(start_date=args.start_date,end_date=args.end_date,
                              category=args.category,state=args.state,seller_id=args.seller_id)
                    if args.enable_nli:
                        from .verify import NLIModel
                    nli=NLIModel() if args.enable_nli else None
                    stats=Agent(db,args.provider,args.model,nli=nli).ask(args.question,f,not args.no_checker)
                elif args.cmd=='evaluate':
                    from .evaluation import run_evaluation
                    stats=run_evaluation(db,args.provider,args.model,args.out)
                elif args.cmd=='status':
                    stats={
                     'orders':db.one('SELECT COUNT(*) n FROM orders')['n'],
                     'all_written_reviews':db.one('SELECT COUNT(*) n FROM reviews')['n'],
                     'sample_reviews':db.one('SELECT COUNT(*) n FROM sample_reviews')['n'],
                     'extractions':db.query('SELECT status,COUNT(*) n FROM extractions GROUP BY status'),
                     'negative_aspect_mentions':db.one("SELECT COUNT(*) n FROM mentions WHERE polarity='negative'")['n'],
                     'db_engine':db.engine}
        print(json.dumps(stats,indent=2,ensure_ascii=False,default=str))
        return 0
    except (ValueError,RuntimeError,FileNotFoundError) as exc:
        print(f'Error: {exc}',file=sys.stderr)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
