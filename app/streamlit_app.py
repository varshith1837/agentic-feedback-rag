"""Streamlit deployment UI. Run: streamlit run app/streamlit_app.py"""
from __future__ import annotations
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from revenue_risk.config import DB_PATH, ASPECTS,load_dotenv
from revenue_risk.db import Database,require_prepared
from revenue_risk.schemas import Filters
from revenue_risk.tools import Tools
from revenue_risk.agent import Agent
from revenue_risk.graph import neighborhood

st.set_page_config(page_title='Revenue-at-Risk | Cited Feedback',page_icon='🔎',layout='wide')
load_dotenv()
st.title('Revenue-at-Risk Feedback Graph')
st.caption('Historical Olist orders (2016–2018) · evidence-linked business analytics · BRL')

try:
    with Database() as conn:
        require_prepared(conn)
        metadata={
            'orders':conn.one('SELECT COUNT(*) n FROM orders')['n'],
            'sample':conn.one('SELECT COUNT(*) n FROM sample_reviews')['n'],
            'processed':conn.one('SELECT COUNT(*) n FROM extractions')['n'],
            'llm':conn.one("SELECT COUNT(*) n FROM extractions WHERE status='llm'")['n'],
            'rules':conn.one("SELECT COUNT(*) n FROM extractions WHERE status='rules_demo'")['n'],
            'fallback':conn.one("SELECT COUNT(*) n FROM extractions WHERE status='fallback'")['n'],
            'categories':[r['category'] for r in conn.query('SELECT DISTINCT category FROM order_items ORDER BY category')],
            'states':[r['customer_state'] for r in conn.query('SELECT DISTINCT customer_state FROM orders ORDER BY customer_state')],
        }
except Exception as exc:
    st.error(f'No prepared database available: {exc}')
    st.code('python -m revenue_risk.cli prepare\npython -m revenue_risk.cli extract --provider rules')
    st.stop()

with st.sidebar:
    st.header('Questions & filters')
    provider=st.selectbox('Agent planner', ['rules','groq','ollama'],help='Rules = offline routing. Groq/Ollama = LLM tool selection.')
    if provider=='groq' and not os.getenv('GROQ_API_KEY'):
        st.warning('GROQ_API_KEY missing. Set it in .env or choose rules.')
    if provider=='ollama':
        st.caption('Ollama must be running with the configured model.')
    category=st.selectbox('Product category',['All']+metadata['categories'])
    state=st.selectbox('Customer state',['All']+metadata['states'])
    start=st.date_input('From purchase date',value=None)
    end=st.date_input('Through purchase date',value=None)
    use_checker=st.checkbox('Number and citation checker',value=True)
    use_nli=st.checkbox('Enable optional NLI text checking',value=False,
        help='Requires pip install -e .[nli], model download, and LLM-extracted English translations.')
    st.caption('NLI is only used for LLM-generated text paraphrases; SQL verifies financial claims.')
    st.divider()
    st.subheader('Try a question')
    sample_q=st.selectbox('Example prompts',[ '— Select —',
        'Which three problem types put the most revenue at risk overall?',
        'Which product categories are most exposed to delivery delays?',
        'Which states have the highest late-delivery revenue at risk?',
        'Show examples of packaging complaints',
        'Is damaged item risk getting worse over time?',
        'What do customers say about the mobile app?'])

m1,m2,m3,m4=st.columns(4)
m1.metric('Orders',f"{metadata['orders']:,}")
m2.metric('Sampled reviews',f"{metadata['sample']:,}")
m3.metric('Extracted reviews',f"{metadata['processed']:,}")
m4.metric('LLM / rules / fallback',f"{metadata['llm']:,} / {metadata['rules']:,} / {metadata['fallback']:,}")
if metadata['rules']:
    st.warning('The bundled starter data uses conservative **rules-demo labels**, not validated LLM annotations. '
               'Run Groq/Ollama extraction for stronger issue labels, then hand-evaluate them. '
               'Do not interpret this demonstration as measured model accuracy.')
if not metadata['processed']:
    st.info('No issue extraction yet. Run `python -m revenue_risk.cli extract --provider groq` '
            'or use `--provider rules` for an explicitly labeled offline demo.')

filters=Filters(start_date=start or None,end_date=end or None,
                category=None if category=='All' else category,
                state=None if state=='All' else state)

chart_tab,graph_tab,help_tab=st.tabs(['Revenue exposure','Feedback graph','About the metric'])
with chart_tab:
    with Database() as db:
        metrics=Tools(db).revenue_at_risk('aspect',filters=filters)
    if metrics:
        chart=pd.DataFrame(metrics).set_index('group_key')[['sample_at_risk_brl']]
        st.bar_chart(chart,y_label='BRL – sample exposure',height=330)
        st.dataframe(pd.DataFrame(metrics),hide_index=True,use_container_width=True)
    else:
        st.info('No negative mentions found in the extracted subset with these filters.')
with graph_tab:
    selected=st.selectbox('Problem type',ASPECTS)
    with Database() as db:
        graph=neighborhood(db,selected,max_reviews=8)
    if graph.number_of_edges():
        import matplotlib.pyplot as plt
        import networkx as nx
        fig,ax=plt.subplots(figsize=(11,5))
        pos=nx.spring_layout(graph,seed=17)
        names=nx.get_node_attributes(graph,'label')
        colors={'aspect':'#f59e0b','review':'#60a5fa','order':'#94a3b8','seller':'#34d399','category':'#c4b5fd'}
        nx.draw_networkx(graph,pos,ax=ax,labels=names,node_size=850,font_size=7,
                         node_color=[colors.get(graph.nodes[n]['kind'],'#dddddd') for n in graph.nodes],
                         edge_color='#9ca3af')
        ax.axis('off');st.pyplot(fig,clear_figure=True)
    else:st.info('No linked review examples currently extracted for this aspect.')
with help_tab:
    st.markdown('''**Definition.** Revenue at risk here is historical order value **associated with**
    negative written feedback—not churn, forecast lost sales, or revenue causally lost.

    **Order-level:** each affected order contributes its **total payment once per aspect**.
    **Seller/category:** each relevant order item contributes **item price + freight**;
    this is an attribution proxy and can differ from the exact customer payment.
    **Estimated:** score × purchase-month sampling weights expand negative mentions to
    the written-review population; this is a rough survey-style estimate, not an experiment.

    **Evidence:** SQL produces totals, original Portuguese reviews illustrate issue labels.
    Reviews alone cannot prove an aggregated financial sum. The checker validates both
    numeric provenance and source eligibility. Paraphrases need additional NLI verification.''')

st.divider()
st.subheader('Ask the feedback agent')
if 'history' not in st.session_state:
    st.session_state.history=[]
for event in st.session_state.history:
    with st.chat_message('user'):
        st.write(event['question'])
    with st.chat_message('assistant'):
        st.markdown(event['answer'])
        if event['data']:
            with st.expander('Query results and provenance'):
                for table in event['data'].get('tables',[]):
                    st.write('Tool:',table['tool'])
                    st.dataframe(pd.DataFrame(table['rows']),hide_index=True)
            with st.expander('Original Portuguese evidence'):
                for s in event['data'].get('sources',[]):
                    st.markdown(f"**[{s['citation_id']}]** · {s.get('review_score')} stars · order `{s.get('order_id')}`")
                    st.write(s.get('original_text'))
                    if s.get('english_text'):st.caption('LLM translation: '+s['english_text'])
            with st.expander('Verifier audit'):
                st.json([{'status':c['status'],'reason':c['reason'],'citations':c.get('citations',[])}
                         for c in event['data'].get('claims',[])])

prompt=st.chat_input('Ask about revenue exposure, problem types, sellers, or customer feedback')
if not prompt and sample_q!='— Select —' and st.button('Ask selected example'):
    prompt=sample_q
if prompt:
    try:
        nli=None
        if use_nli:
            from revenue_risk.verify import NLIModel
            @st.cache_resource(show_spinner='Loading multilingual / English claim checking model...')
            def load_nli():
                return NLIModel()
            nli=load_nli()
        with Database() as db:
            answer=Agent(db,provider=provider,nli=nli).ask(prompt,filters,checker=use_checker)
        claims=[c for c in answer['claims'] if c['status']=='supported']
        text=('**I cannot find support for this in the data.** '+answer['refusal_reason']
              if answer['refused'] else '\n\n'.join('• '+c['text']+' '+
               ' '.join('`['+cid+']`' for cid in c['citations']) for c in claims))
        text+='\n\n*'+answer['metric_note']+'*'
        st.session_state.history.append({'question':prompt,'answer':text,'data':answer})
        st.rerun()
    except Exception as exc:
        st.error(f'Query could not be completed: {exc}')
