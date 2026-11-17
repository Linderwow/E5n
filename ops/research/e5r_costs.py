r"""E5R account factory (e5r_factory.py) re-run at the real fill cost - information, 2026-11-17.
    python e5r_costs.py [ticks]   default 4.5
"""
import os, sys
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import e5b_lab as L
import e5x_lab as X
import e5r_factory as F

TK = float(sys.argv[1]) if len(sys.argv) > 1 else 4.5
O = X.SIDE; X.SIDE = lambda t: O(t - 2.35 + TK)
L.OUT = os.path.join(HERE, f'E5R_COSTS_{TK:g}.txt'); L.say(f"(fill cost {TK} ticks a side instead of 2.35)")
F.main()
