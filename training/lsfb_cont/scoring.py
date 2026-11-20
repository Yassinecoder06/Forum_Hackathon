"""Versioned diagnostics: never discard the content of malformed predictions."""
import re
try:
    from .prepare import parse_plan
except ImportError:
    from prepare import parse_plan

QUOTED = r'"(?:[^"\\]|\\.)*"'
CHUNK = re.compile(r'(?:FINGERSPELL\s+'+QUOTED+r'\s+'+QUOTED+r'|SIGN\s+'+QUOTED+r')')

def distance(a, b):
    previous = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(x != y)))
        previous = current
    return previous[-1]

def operations(text):
    return [repr(sorted(o.items())) for o in parse_plan(text)]

def recover(text):
    result, end = [], 0
    for match in CHUNK.finditer(text):
        gap = text[end:match.start()].strip(' ;\n\t')
        result.extend('<INVALID:'+t+'>' for t in gap.split())
        result.extend(operations(match.group()))
        end = match.end()
    gap = text[end:].strip(' \n\t')
    result.extend('<INVALID:'+t+'>' for t in gap.split())
    return result

def score(predictions, references):
    if len(predictions) != len(references) or not references:
        raise ValueError('Expected equal, nonempty prediction/reference lists')
    invalid = exact = edits = raw_edits = total = raw_total = predicted = 0
    for p, r in zip(predictions, references):
        ref = operations(r)
        try:
            pred = operations(p)
            exact += pred == ref
        except (ValueError, TypeError, KeyError):
            invalid += 1
            pred = recover(p)
        edits += distance(ref, pred)
        predicted += len(pred)
        total += len(ref)
        rt, pt = re.findall(r'\w+|[^\w\s]', r), re.findall(r'\w+|[^\w\s]', p)
        raw_edits += distance(rt, pt)
        raw_total += len(rt)
    return dict(scoring_version='recovered-operations-plus-invalid-spans-v2',
                operation_wer=edits/max(1,total), raw_token_wer=raw_edits/max(1,raw_total),
                exact_match=exact/len(references), invalid_plan_rate=invalid/len(references),
                mean_predicted_operations=predicted/len(references),
                mean_reference_operations=total/len(references), examples=len(references))
