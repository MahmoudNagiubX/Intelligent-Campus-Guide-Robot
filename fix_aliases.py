from app.storage.db import get_db

conn = get_db()

dept = conn.execute("SELECT id FROM departments WHERE code='SET' AND lang='en'").fetchone()
if dept:
    conn.execute("INSERT OR IGNORE INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias, lang) VALUES ('department', ?, 'software engineering', 'software engineering', 'en')", (dept['id'],))
    conn.execute("INSERT OR IGNORE INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias, lang) VALUES ('department', ?, 'set department', 'set department', 'en')", (dept['id'],))
    conn.commit()
    print('EN aliases added')

dept_ar = conn.execute("SELECT id FROM departments WHERE code='SET' AND lang='ar'").fetchone()
if dept_ar:
    conn.execute("INSERT OR IGNORE INTO aliases (canonical_type, canonical_id, alias_text, normalized_alias, lang) VALUES ('department', ?, 'هندسة البرمجيات', 'هندسة البرمجيات', 'ar')", (dept_ar['id'],))
    conn.commit()
    print('AR aliases added')
