"""Offline HTTP protocol tests: no live Groq/Ollama keys or servers required."""
from __future__ import annotations
import json
import httpx
from revenue_risk.llm import LLMClient


def test_groq_json_mode_and_cache(monkeypatch,toy):
    monkeypatch.setenv('GROQ_API_KEY','this-is-a-test-key')
    outbound=[]
    def fake_post(url,**kwargs):
        outbound.append((url,kwargs))
        return httpx.Response(200,request=httpx.Request('POST',url),json={
            'choices':[{'message':{'content':'{"calls":[],"refuse":true,"reason":"unsupported"}'}}]})
    monkeypatch.setattr(httpx,'post',fake_post)
    client=LLMClient('groq',model='mock')
    messages=[{'role':'system','content':'JSON only'}]
    assert client.json(messages,toy)['refuse']
    assert client.json(messages,toy)['refuse']
    assert len(outbound)==1  # verified request signature was cached
    assert outbound[0][0]=='https://api.groq.com/openai/v1/chat/completions'
    assert outbound[0][1]['json']['response_format']=={'type':'json_object'}
    assert outbound[0][1]['headers']['Authorization']=='Bearer this-is-a-test-key'
    client.invalidate(toy,messages)
    assert toy.one('SELECT COUNT(*) n FROM llm_cache')['n']==0


def test_ollama_nonstreaming_json_mode(monkeypatch,toy):
    outbound=[]
    def fake_post(url,**kwargs):
        outbound.append((url,kwargs))
        return httpx.Response(200,request=httpx.Request('POST',url),json={
           'message':{'content':'{"reviews":[]}'},'done':True})
    monkeypatch.setattr(httpx,'post',fake_post)
    client=LLMClient('ollama',model='qwen2.5:7b')
    assert client.json([{'role':'user','content':'Return JSON'}],toy)=={'reviews':[]}
    assert outbound[0][0].endswith('/api/chat')
    assert outbound[0][1]['json']['format']=='json'
    assert outbound[0][1]['json']['stream'] is False


def test_transport_failure_keeps_rows_pending(monkeypatch,toy):
    from revenue_risk.extract import run_extract
    from revenue_risk.llm import LLMError
    class Gone:
        model='mock'
        def json(self,*args,**kwargs):raise LLMError('Provider unavailable')
    before=toy.one("SELECT COUNT(*) n FROM extractions WHERE status='llm'")['n']
    report=run_extract(toy,'groq',llm=Gone(),limit=2,batch_size=1,pause=0,overwrite=True)
    assert report['processed']==0
    assert 'stopped_reason' in report
    after=toy.one("SELECT COUNT(*) n FROM extractions WHERE status='llm'")['n']
    assert after==before
