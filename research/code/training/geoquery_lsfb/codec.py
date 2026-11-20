"""Lossless graph codec with a train-only HamNoSys dictionary and literal fallback."""
from urllib.parse import quote,unquote
from .audit_baseline import validate_graph,ARTICULATIONS

CODES={'right-hand-articulation':'R','left-hand-articulation':'L','two-hand-articulation':'B','adjacent':'A','start-coincides':'S','end-coincides':'E','during':'D'}
REVERSE={v:k for k,v in CODES.items()}

class GraphCodec:
 def __init__(self,hns):
  if len(set(hns))!=len(hns):raise ValueError('Duplicate dictionary entries')
  self.hns=list(hns);self.index={s:i for i,s in enumerate(hns)}
 @classmethod
 def fit(cls,graphs):return cls(sorted({row[2] for g in graphs for row in g if row[0] in ARTICULATIONS}))
 def encode(self,graph):
  validate_graph(graph);ids={row[1]:i for i,row in enumerate(r for r in graph if r[0] in ARTICULATIONS)};lines=[]
  for relation,a,b in graph:
   if relation in ARTICULATIONS:
    h='h'+str(self.index[b]) if b in self.index else 'u'+'.'.join(format(ord(c),'x') for c in b)
    lines.append(CODES[relation]+'|'+quote(a,safe="-'")+'|'+h)
   else:lines.append(CODES[relation]+'|'+str(ids[a])+'|'+str(ids[b]))
  return ' '.join(lines)
 def decode(self,text):
  rows=[];ids=[]
  for token in text.split():
   parts=token.split('|')
   if len(parts)!=3 or parts[0] not in REVERSE:raise ValueError('Invalid graph token')
   code,a,b=parts;relation=REVERSE[code]
   if relation in ARTICULATIONS:
    name=unquote(a)
    if b.startswith('h'):
     try:value=self.hns[int(b[1:])]
     except (ValueError,IndexError):raise ValueError('Unknown HNS dictionary index')
     if not b[1:].isdigit():raise ValueError('Invalid dictionary index')
    elif b.startswith('u'):
     try:value=''.join(chr(int(c,16)) for c in b[1:].split('.'))
     except (ValueError,OverflowError):raise ValueError('Invalid literal HNS')
    else:raise ValueError('Missing HNS representation')
    ids.append(name);rows.append([relation,name,value])
   else:
    if not a.isdigit() or not b.isdigit():raise ValueError('Invalid edge index')
    rows.append([relation,int(a),int(b)])
  for row in rows:
   if row[0] not in ARTICULATIONS:
    try:row[1],row[2]=ids[row[1]],ids[row[2]]
    except IndexError:raise ValueError('Dangling edge index')
  validate_graph(rows);return rows
