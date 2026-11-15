"""Register audited metadata-only sources; never fetch restricted corpus content."""
import csv,sqlite3
from catalog import ROOT

def main():
    db=sqlite3.connect(ROOT/'data/catalog/catalog.sqlite')
    with (ROOT/'research/sources.csv').open() as f:
        for r in csv.DictReader(f):
            if r['source_id'] in {'stk','sud','commons-laura'}:continue
            db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?)',tuple(r[k] for k in ['source_id','url','revision','license_id','terms_url','access','notes']))
    db.commit();db.close()
if __name__=='__main__':main()
