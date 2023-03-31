# Notes

## Parser vs evaluator

The current approach doesn't quite work. Instead,

1. Enforce BQN's naming convention: we then know what type of value each name holds.
2. Parse to the tuple-based, almost-lisp AST.
3. Walk the tree to evaluate.

4. ~Stranding is broken!~ FIXED