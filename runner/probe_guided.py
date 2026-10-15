"""Which request shape actually constrains this server's output to the three labels?"""
import json, sys, urllib.request
BASE, MODEL = sys.argv[1], sys.argv[2]
SYS = "You are a referee. Answer with exactly one word: Correct, Incorrect, or Unknown."
USR = "The man drank the soup."
variants = {
    "plain": {},
    "guided_choice": {"guided_choice": ["Correct", "Incorrect", "Unknown"]},
    "structured_outputs": {"structured_outputs": {"choice": ["Correct", "Incorrect", "Unknown"]}},
    "chat_template_kwargs": {"chat_template_kwargs": {"enable_thinking": False}},
    "both": {"guided_choice": ["Correct", "Incorrect", "Unknown"],
             "chat_template_kwargs": {"enable_thinking": False}},
}
for name, extra in variants.items():
    body = {"model": MODEL, "max_tokens": 6, "temperature": 0.0, "logprobs": True,
            "top_logprobs": 5,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": USR}], **extra}
    req = urllib.request.Request(f"{BASE}/chat/completions", method="POST",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=60))
        ch = d["choices"][0]
        toks = [t["token"] for t in ch["logprobs"]["content"]]
        print(f"{name:<22} content={ch['message']['content']!r:<40} tokens={toks}")
        print(f"{'':<22} top@0={[(t['token'], round(2.718**t['logprob'],3)) for t in ch['logprobs']['content'][0]['top_logprobs'][:4]]}")
    except Exception as e:
        body_txt = getattr(e, "read", lambda: b"")()[:200]
        print(f"{name:<22} ERROR {e} {body_txt}")
