import json

with open("scripts/chatgpt/captions.json", "r") as f:
    captions = json.load(f)

fn = lambda d: d["n_tokens_short"]
for c in captions:
    if c.get('info', None) == 'Empty Graph!':
        continue
    c['captions'].sort(key=fn)
    c['captions'] = [d for d in c['captions'] if d["n_tokens_short"] <= 80][-1]

with open("scripts/chatgpt/captions_filtered.json", "w") as f:
    captions = json.dump(captions, f)