import sys
from aplparser import APLParser
p = APLParser(parse_only=True)
print(p.parse(sys.argv[1]))
