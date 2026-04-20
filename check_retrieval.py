from app.retrieval.search import search

tests = [
    ('robotics lab',         'en'),
    ('software engineering', 'en'),
    ('الروبوتات ورؤية الآلة',  'ar'),
    ('قسم هندسة البرمجيات',   'ar'),
    ('C207',                 'en'),
    ('معمل الإجهاد',          'ar'),
]

for query, lang in tests:
    r = search(query, lang=lang)
    name = r.canonical_name or 'NONE'
    conf = round(r.confidence, 2)
    via  = r.matched_via or '-'
    print(f'[{lang}] {query:<32} -> {r.status.value:<12} {name:<35} conf={conf} via={via}')
