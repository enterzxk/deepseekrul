"""Cross-check the committed first-passage implementation against an
independent reference, without importing the test module (it pulls torch/pytest).

We exec only the three pure-numpy functions, taken verbatim from the file.
"""
import ast
import random
import numpy as np

SRC = 'paper_b/hc/test_d_engine.py'
text = open(SRC, encoding='utf-8').read()
tree = ast.parse(text)

want = {'trajectory', 'first_passage', 'arrival'}
segments = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        segments.append(ast.get_source_segment(text, node))
assert len(segments) >= 2, 'expected >=2 functions, got %d' % len(segments)

ns = {'np': np}
exec('\n\n'.join(segments), ns)
arrival = ns.get('arrival')
if arrival is None:
    arrival = lambda D0, nu: ns['first_passage'](ns['trajectory'](D0, nu))
print('found:', sorted(k for k in ns if k in want))


def ref_first_passage(D0, nu):
    D = float(D0)
    tot = 0.0
    for v in nu:
        v = float(v)
        if v > 0:
            tot += min(1.0, max(0.0, (1.0 - D) / v))
        else:
            tot += 1.0 if D < 1.0 else 0.0
        D += v
    return tot


random.seed(101)
worst, n = 0.0, 0
for _ in range(5000):
    K = random.randint(1, 150)
    D0 = random.choice([0.0, random.random(), random.random() * 0.99, 1.0, 1.2])
    style = random.choice(['rand', 'slow', 'fast', 'zero', 'decel'])
    nu = []
    for k in range(K):
        if style == 'zero' and random.random() < 0.5:
            nu.append(0.0)
        elif style == 'slow':
            nu.append(random.random() * 0.002)
        elif style == 'fast':
            nu.append(random.random() * 0.5)
        elif style == 'decel':
            nu.append(max(0.0, 0.3 * (0.9 ** k)))
        else:
            nu.append(random.random() * 0.05)
    got = float(arrival(np.array([[D0]]), np.array([nu]))[0])
    worst = max(worst, abs(got - ref_first_passage(D0, nu)))
    n += 1
print('cases:', n, ' worst |impl - reference| =', worst)

edge = [
    ('already failed D0=1.2', 1.2, [0.01] * 5),
    ('exact node hit', 0.0, [0.5, 0.5, 0.5]),
    ('non-integer hit', 0.0, [0.4, 0.4, 0.4]),
    ('plateau then jump', 0.0, [0.0, 0.0, 1.0, 0.0]),
    ('never reaches', 0.0, [0.1] * 5),
    ('decel after 0.90', 0.90, [0.01] * 10),
]
for name, D0, nu in edge:
    got = float(arrival(np.array([[D0]]), np.array([nu]))[0])
    print('%-22s impl=%.6f  ref=%.6f' % (name, got, ref_first_passage(D0, nu)))
