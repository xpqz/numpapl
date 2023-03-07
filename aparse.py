import sys
from apl import APL
p = APL(parse_only=True)
print(p.parse(sys.argv[1]))
