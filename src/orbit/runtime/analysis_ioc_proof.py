"""Closed, bounded source proof for static browser destinations. Never execute JS.

The parser consumes every script token. The effect checker admits only the
specified DOM/read/storage operations and proven pure helpers; unknown syntax
or effects break the proof. A discovered assignment alone is only an objective.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser

from orbit.runtime.analysis_deobfuscate import MAX_INPUT_CHARS, TransformStage, _fcc_mask_code
from orbit.runtime.analysis_indicators import uris_in

KIND = 'js_destination_proof'
MAX_NODES = 8192
MAX_DEPTH = 48


class Unproved(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    kind: str
    value: object
    children: tuple
    start: int
    end: int


@dataclass(frozen=True)
class Destination:
    start: int
    end: int
    property: str
    state: str
    reason: str
    value: str | None = None
    links: tuple[tuple[str, int, int, str], ...] = ()

    def stage(self, source):
        if self.state != 'RESOLVED_EXACT':
            return None
        offset = len(source.encode()[:self.start].decode())
        return TransformStage(KIND, 0, self.property, source[:offset].count('\n') + 1,
                              offset, 0, source, self.value,
                              hashlib.sha256(source.encode()).hexdigest(),
                              hashlib.sha256(self.value.encode()).hexdigest())


@dataclass(frozen=True)
class DestinationScan:
    """A positive closed-subset certificate, separate from per-sink proofs."""
    source_sha256: str
    objectives: tuple[Destination, ...]
    complete: bool
    reason: str = ''


# Plain HTML structure whose document membership we can account for without
# implementing browser tree construction. Unknown/inert/foreign contexts never
# supply DOM origins. Discovery uses a stricter subset: CSS and URL attributes
# can introduce other destinations even when they do not invalidate a DOM ID.
_DOM_TAGS = frozenset(('html head body title style script div span main section '
    'article header footer nav aside p h1 h2 h3 h4 h5 h6 a area br meta').split())
_PASSIVE_TAGS = _DOM_TAGS - {'style', 'meta'}
_VOID_TAGS = frozenset('area base br col embed hr img input link meta param source track wbr'.split())
_PASSIVE_ATTRS = frozenset('id class title lang dir role'.split())


class Scripts(HTMLParser):
    def __init__(self, source):
        super().__init__(convert_charrefs=False)
        self.source = source
        self.lines = [0] + [m.end() for m in re.finditer('\n', source)]
        self.begin = None
        self.ranges = []
        self.external = False
        self.inert = []
        self.elements = {}
        self.open_elements = []
        self.dom_complete = True
        self.discovery_issues = []

    def incomplete(self, reason):
        if reason not in self.discovery_issues:
            self.discovery_issues.append(reason)

    def pos(self):
        line, column = self.getpos()
        return self.lines[line - 1] + column

    def handle_starttag(self, tag, attrs):
        # HTMLParser applies generic character-reference decoding to attributes;
        # HTML's attribute-context rules differ for ambiguous ampersands. Until
        # that context is proved, decoded IDs cannot authorize a DOM lookup.
        if '&' in self.get_starttag_text():
            self.dom_complete = False
            self.incomplete('HTML attribute character-reference semantics are unsupported')
        if tag not in _DOM_TAGS:
            self.dom_complete = False
        if tag not in _PASSIVE_TAGS:
            self.incomplete('HTML contains a context outside the closed discovery subset')
        if any(k not in _PASSIVE_ATTRS and not (tag == 'script' and k == 'type') for k, _v in attrs):
            self.incomplete('HTML attributes are not fully classified for destination discovery')
        if tag in ('iframe', 'frame', 'object', 'embed') or any(k == 'srcdoc' for k, _v in attrs):
            self.external = True
        if tag in ('template', 'noscript', 'svg', 'math'):
            self.inert.append(tag)
        pairs = dict(attrs)
        if len(pairs) != len(attrs):
            self.external = True  # duplicate attribute semantics not interpreted
        # Tree-construction repairs can invalidate lexical nesting/leaf claims.
        parents = [entry[0] for entry in self.open_elements]
        if ((tag == 'a' and 'a' in parents)
                or ('p' in parents and tag not in ('span', 'a', 'br', 'script'))):
            self.dom_complete = False
        for entry in self.open_elements:
            entry[3] = False
        element = [tag, self.pos(), self.pos() + len(self.get_starttag_text()), True]
        if pairs.get('id') and not self.inert:
            self.elements.setdefault(pairs['id'], []).append(element)
        if tag not in _VOID_TAGS:
            self.open_elements.append(element)
        if any(k.startswith('on') or (v or '').lstrip().lower().startswith('javascript:') for k, v in attrs):
            self.external = True
        if tag == 'script':
            if (self.inert or set(pairs) - _PASSIVE_ATTRS - {'type'}
                    or (pairs.get('type') or '').lower() not in
                    ('', 'text/javascript', 'application/javascript')):
                self.external = True
            else:
                self.begin = self.pos() + len(self.get_starttag_text())

    def handle_endtag(self, tag):
        if tag == 'script' and self.begin is not None:
            self.ranges.append((self.begin, self.pos()))
            self.begin = None
        if tag in self.inert:
            self.inert.remove(tag)
        if self.open_elements and self.open_elements[-1][0] == tag:
            self.open_elements.pop()
        elif tag not in _VOID_TAGS:
            self.dom_complete = False

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            # HTML ignores a self-closing flag on ordinary elements. Do not
            # pretend that an XML-style lexical closure proves HTML structure.
            self.dom_complete = False
            self.incomplete('non-void self-closing HTML structure is unsupported')

    def handle_decl(self, decl):
        if decl.lower() != 'doctype html':
            self.dom_complete = False
            self.incomplete('unsupported HTML declaration')

    def unknown_decl(self, data):
        self.dom_complete = False
        self.incomplete('unclassified HTML declaration')

    def handle_pi(self, data):
        self.dom_complete = False
        self.incomplete('unclassified HTML processing instruction')


def _script_view(source):
    if not source.lstrip().startswith('<'):
        return source, False, {}, [(0, len(source))], ()
    parser = Scripts(source)
    parser.feed(source)
    parser.close()
    if parser.open_elements:
        parser.dom_complete = False
    if not parser.dom_complete:
        parser.incomplete('HTML document membership is not proven by the supported structure')
    view = list(' ' * len(source))
    for lo, hi in parser.ranges:
        view[lo:hi] = source[lo:hi]
    # HTMLParser is not the HTML script-data state machine: escaped script
    # modes can change where a closing tag takes effect. U+0000 is replaced by
    # HTML preprocessing. Neither may silently supply byte-exact JS premises.
    if '\x00' in source or any('<!--' in source[lo:hi] for lo, hi in parser.ranges):
        parser.external = True
        parser.incomplete('HTML tokenizer state does not preserve the supported script view')
    return (''.join(view), parser.external or parser.begin is not None,
            parser.elements if parser.dom_complete else {}, parser.ranges,
            tuple(parser.discovery_issues))


def script_view(source):
    return _script_view(source)[:4]


# No escape interpretation, ASI, regex literals, legacy numbers or catch-all
# token. Unsupported spelling stops the whole proof; locations are preserved.
_TOKEN = re.compile(r'''[ \t\r\n]+|//[^\r\n]*|/\*[\s\S]*?\*/|(?P<str>"[^"\\\r\n]*"|'[^'\\\r\n]*')|(?P<id>[A-Za-z_$][A-Za-z0-9_$]*)|(?P<num>0|[1-9][0-9]*)|(?P<op>\+\+|\+=|[{}()[\].,;:+^<!=])''')
_RESERVED = frozenset('break case catch class const continue debugger default delete do else enum export extends false finally for function if implements import in instanceof interface let new null package private protected public return static super switch this throw true try typeof var void while with yield await'.split())


class Parser:
    def __init__(self, text):
        self.text, self.tokens, self.i, self.depth = text, [], 0, 0
        pos = 0
        while pos < len(text):
            m = _TOKEN.match(text, pos)
            if not m:
                raise Unproved(f'unsupported source syntax at character {pos}')
            if m.lastgroup:
                if m.lastgroup == 'num' and len(m.group()) > 15:
                    raise Unproved('numeric literal exceeds exact supported bound')
                self.tokens.append((m.lastgroup, m.group(), pos, m.end()))
            pos = m.end()
            if len(self.tokens) > MAX_NODES:
                raise Unproved('static syntax bound reached')
        self.tokens.append(('eof', '', len(text), len(text)))

    def peek(self, value=None):
        return self.tokens[self.i][1] == value if value is not None else self.tokens[self.i]

    def take(self, value=None):
        token = self.peek()
        if value is not None and token[1] != value:
            raise Unproved(f'expected {value!r} at character {token[2]}')
        self.i += 1
        return token

    def identifier(self):
        token = self.take()
        if token[0] != 'id' or token[1] in _RESERVED or token[1] in ('eval', 'arguments'):
            raise Unproved('unsupported identifier/binding')
        return token

    def node(self, kind, value, children, start):
        return Node(kind, value, tuple(children), start, self.tokens[self.i - 1][3])

    def program(self, end=''):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise Unproved('static nesting bound reached')
        result = []
        while not self.peek(end):
            if self.peek()[0] == 'eof':
                raise Unproved('unterminated block')
            result.append(self.statement())
        self.depth -= 1
        return tuple(result)

    def block(self):
        start = self.take('{')[2]
        children = self.program('}')
        self.take('}')
        return self.node('block', None, children, start)

    def function(self, declaration):
        start = self.take('function')[2]
        name = self.identifier()[1] if declaration else ''
        self.take('(')
        params = []
        if not self.peek(')'):
            while True:
                params.append(self.identifier()[1])
                if not self.peek(','):
                    break
                self.take(',')
        if len(set(params)) != len(params):
            raise Unproved('duplicate parameter')
        self.take(')')
        return self.node('function', (name, tuple(params)), (self.block(),), start)

    def statement(self):
        start = self.peek()[2]
        kind = self.peek()[1]
        if kind == '{':
            return self.block()
        if kind in ('var', 'const', 'let'):
            self.take()
            name = self.identifier()[1]
            self.take('=')  # no implicit undefined or multi-declarator proof
            value = self.expression()
            self.take(';')
            return self.node('var', (kind, name), (value,), start)
        if kind == 'function':
            return self.function(True)
        if kind in ('if', 'while'):
            self.take()
            self.take('(')
            condition = self.expression()
            self.take(')')
            yes = self.block()  # no dangling else / unbraced dominance guess
            children = [condition, yes]
            if kind == 'if' and self.peek('else'):
                self.take('else')
                children.append(self.statement() if self.peek('if') else self.block())
            return self.node(kind, None, children, start)
        if kind == 'return':
            token = self.take()
            if '\n' in self.text[token[3]:self.peek()[2]] or '\r' in self.text[token[3]:self.peek()[2]]:
                raise Unproved('return automatic-semicolon syntax is unsupported')
            value = self.expression()
            if self.peek(';'):
                self.take(';')
            elif not self.peek('}'):
                raise Unproved('explicit return terminator required')
            return self.node('return', None, (value,), start)
        expr = self.expression()
        if self.peek(';'):
            self.take(';')
        elif expr.kind != 'postfix' or not self.peek('}'):
            raise Unproved('explicit statement terminator required')
        return self.node('expr', None, (expr,), start)

    def expression(self, minimum=0):
        self.depth += 1
        if self.depth > MAX_DEPTH:
            raise Unproved('expression nesting bound reached')
        token = self.peek()
        start = token[2]
        if token[1] in ('+', '!'):
            op = self.take()[1]
            lhs = self.node('unary', op, (self.expression(6),), start)
        elif token[1] == '(':
            self.take('(')
            lhs = self.expression()
            self.take(')')
        elif token[1] == 'function':
            lhs = self.function(False)
        elif token[1] == '{':
            self.take('{')
            fields, values = [], []
            while not self.peek('}'):
                fields.append(self.identifier()[1])
                self.take(':')
                values.append(self.expression(2))
                if not self.peek(','):
                    break
                self.take(',')
            self.take('}')
            if len(set(fields)) != len(fields):
                raise Unproved('duplicate object key')
            if '__proto__' in fields:
                raise Unproved('special object prototype key is unsupported')
            lhs = self.node('object', tuple(fields), values, start)
        elif token[0] in ('str', 'num'):
            self.take()
            lhs = self.node(token[0], token[1][1:-1] if token[0] == 'str' else int(token[1]), (), start)
        elif token[1] in ('true', 'false'):
            self.take()
            lhs = self.node('boolean', token[1], (), start)
        else:
            lhs = self.node('id', self.identifier()[1], (), start)
        while True:
            op = self.peek()[1]
            if op == '.':
                self.take('.')
                lhs = self.node('member', self.identifier()[1], (lhs,), start)
            elif op == '[':
                self.take('[')
                key = self.expression()
                self.take(']')
                lhs = self.node('index', None, (lhs, key), start)
            elif op == '(':
                self.take('(')
                args = []
                if not self.peek(')'):
                    while True:
                        args.append(self.expression(2))
                        if not self.peek(','):
                            break
                        self.take(',')
                self.take(')')
                lhs = self.node('call', None, (lhs, *args), start)
            elif op == '++':
                if any(c in self.text[lhs.end:self.peek()[2]] for c in '\r\n'):
                    raise Unproved('postfix automatic-semicolon syntax is unsupported')
                self.take()
                lhs = self.node('postfix', op, (lhs,), start)
            else:
                priority = {'=':1, '+=':1, '<':2, '^':3, '+':4}.get(op, -1)
                if priority < minimum:
                    break
                self.take()
                rhs = self.expression(priority if op in ('=', '+=') else priority + 1)
                lhs = self.node('binary', op, (lhs, rhs), start)
        self.depth -= 1
        return lhs


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


@dataclass(frozen=True)
class Value:
    kind: str
    text: str | None = None
    links: tuple = ()


@dataclass
class Scope:
    parent: Scope | None
    bindings: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)


class Proof:
    """Closed effect and binding checker; never runs statements or callbacks."""
    def __init__(self, source, nodes, elements, scripts):
        self.source = source
        self.nodes = nodes
        self.global_scope = Scope(None)
        self.context = {}
        self.blocks = {}
        self.writes = {}
        self.functions = {}
        self.pure = set()
        self.sinks = []
        self.sink_origins = {}
        self.effects_checked = False
        self.excluded_branches = []
        self.ambiguous_branches = []
        self.discovery_issues = []
        self.elements = elements
        self.scripts = scripts
        self.work = 0
        self.calls = []
        self._collect(nodes, self.global_scope, ())
        for scope in self.context.values():
            if scope.params.keys() & scope.bindings.keys():
                raise Unproved('parameter/local declaration overlap')
            for definitions in scope.bindings.values():
                if len(definitions) > 1 and any(d.kind == 'var' and d.value[0] != 'var' for d in definitions):
                    raise Unproved('duplicate lexical declaration')
        # Browser roots/builtins are only trusted when this entire parsed source
        # cannot rebind them. No alias to them is an allowed primitive value.
        for name in ('String', 'document', 'window', 'navigator', 'location', 'localStorage'):
            if any(name in s.bindings or name in s.params for s in self.context.values()):
                raise Unproved(f'builtin {name} is shadowed')
            if any(key[1] == name for key in self.writes):
                raise Unproved(f'builtin {name} is assigned')

    def _collect(self, nodes, scope, blocks):
        for node in nodes:
            self.context[id(node)] = scope
            self.blocks[id(node)] = blocks
            if node.kind == 'function':
                name, params = node.value
                if name:
                    scope.bindings.setdefault(name, []).append(node)
                    self.functions[name] = node
                child = Scope(scope, params={p: Value('parameter') for p in params})
                self._collect(node.children, child, ())
            else:
                if node.kind == 'var':
                    scope.bindings.setdefault(node.value[1], []).append(node)
                if node.kind in ('binary', 'postfix') and node.value in ('=', '+=', '++'):
                    lhs = node.children[0]
                    if lhs.kind == 'id':
                        self.writes.setdefault((id(scope), lhs.value), []).append(node)
                child_blocks = (*blocks, node.start) if node.kind == 'block' else blocks
                self._collect(node.children, scope, child_blocks)

    def link(self, role, node):
        lo, hi = len(self.source[:node.start].encode()), len(self.source[:node.end].encode())
        return (role, lo, hi, hashlib.sha256(self.source[node.start:node.end].encode()).hexdigest())

    def binding(self, name, use):
        scope = self.context[id(use)]
        chain = []
        while scope:
            chain.append(scope)
            if name in scope.params:
                return scope.params[name]
            if name in scope.bindings:
                definitions = scope.bindings[name]
                if len(definitions) != 1:
                    # Repeated primitive declarations can be inspected for
                    # effects, but never supply an exact destination value.
                    # Each initializer is checked independently by check().
                    if all(d.kind == 'var' and self.primitive_initializer(d.children[0]) for d in definitions):
                        return Value('primitive')
                    raise Unproved(f'ambiguous definitions for {name}')
                node = definitions[0]
                # Any assignment from this scope or a nested one may change the
                # captured value. Resolve ownership structurally, not by spelling.
                for (_sid, written), writes in self.writes.items():
                    if written != name:
                        continue
                    for write in writes:
                        owner = self.context[id(write)]
                        while owner is not scope and owner is not None:
                            if name in owner.params or name in owner.bindings:
                                break
                            owner = owner.parent
                        if owner is scope:
                            raise Unproved(f'mutated binding {name}')
                block = self.blocks[id(node)]
                use_blocks = self.blocks[id(use)]
                if node.kind != 'function' and (node.start >= use.start or use_blocks[:len(block)] != block):
                    raise Unproved(f'initializer does not dominate {name}')
                if node.kind != 'function':
                    for call in self.calls:
                        call_scope = self.context[id(call)]
                        while call_scope is not None and call_scope is not scope:
                            call_scope = call_scope.parent
                        if call_scope is scope and node.start >= call.start:
                            raise Unproved(f'initializer does not dominate invocation of {name}')
                if node.kind == 'function' and block:
                    raise Unproved('conditional function declaration')
                if node.kind == 'function' and node.start > use.start:
                    if not any(lo <= use.start < node.start < hi for lo, hi in self.scripts):
                        raise Unproved('function declaration belongs to a later script')
                return node
            scope = scope.parent
        raise Unproved(f'no local binding for {name}')

    @staticmethod
    def primitive_initializer(node):
        if node.kind in ('str', 'num'):
            return True
        if node.kind != 'call' or len(node.children) != 2 or node.children[1].kind != 'str':
            return False
        callee = node.children[0]
        return (callee.kind == 'member' and callee.value == 'getItem'
                and callee.children[0].kind == 'member' and callee.children[0].value == 'localStorage'
                and callee.children[0].children[0].kind == 'id'
                and callee.children[0].children[0].value == 'window')

    def function_body(self, fn):
        block = fn.children[0]
        return self.source[block.start + 1:block.end - 1]

    def pure_family(self, fn):
        # Reuse the exact arithmetic recognizers, but only AFTER complete syntax
        # and binding parsing. Their parameters and full bodies must match.
        from orbit.runtime.analysis_ioc_xor import provider, loop
        params = fn.value[1]
        body = self.function_body(fn)
        roles = provider(params, body)
        if roles:
            return ('provider', roles)
        # Loop is matched with the provider roles when its caller is proven.
        if len(params) == 3:
            for other in self.functions.values():
                roles = provider(other.value[1], self.function_body(other))
                if roles and loop(params, body, roles):
                    return ('loop', roles)
        statements = fn.children[0].children
        if not params and statements and statements[-1].kind == 'return' and all(s.kind == 'var' for s in statements[:-1]):
            for statement in statements:
                for expr in walk(statement.children[0]):
                    if expr.kind not in ('str', 'num', 'id', 'binary', 'call'):
                        raise Unproved('return helper contains an unsupported effect')
                    if expr.kind == 'binary' and expr.value != '+':
                        raise Unproved('return helper mutates state')
                    if expr.kind == 'call' and expr.children[0].kind != 'id':
                        raise Unproved('return helper contains an indirect call')
            return ('return', None)
        raise Unproved('helper body is not in the closed pure subset')

    def value(self, node, depth=0):
        self.work += 1
        if depth > MAX_DEPTH or self.work > MAX_NODES:
            raise Unproved('proof dependency bound reached')
        kind = node.kind
        if kind == 'str':
            return Value('string', node.value, (self.link('literal', node),))
        if kind == 'num':
            return Value('number', str(node.value), (self.link('literal', node),))
        if kind == 'boolean':
            return Value('boolean', node.value)
        if kind == 'id':
            if node.value in ('true', 'false'):
                return Value('boolean', node.value)
            if node.value == 'location':
                return Value('dom', '@location')
            bound = self.binding(node.value, node)
            if isinstance(bound, Value):
                return bound
            if bound.kind == 'function':
                return Value('function', node.value, (self.link('helper', bound),))
            result = self.value(bound.children[0], depth + 1)
            return Value(result.kind, result.text, (*result.links, self.link('binding', bound)))
        if kind == 'unary' and node.value == '!':
            self.value(node.children[0], depth + 1)
            return Value('boolean')
        if kind == 'binary' and node.value == '+':
            a, b = (self.value(c, depth + 1) for c in node.children)
            if a.kind != 'string' or b.kind != 'string' or a.text is None or b.text is None:
                raise Unproved('concatenation needs exact strings')
            if len(a.text) + len(b.text) > 2048:
                raise Unproved('destination size bound reached')
            return Value('string', a.text + b.text, (*a.links, *b.links, self.link('concat', node)))
        if kind == 'member':
            root, prop = node.children[0], node.value
            if root.kind == 'id' and (root.value, prop) == ('navigator', 'userAgent'):
                return Value('runtime_string')
            if root.kind == 'id' and (root.value, prop) == ('window', 'location'):
                return Value('dom', '@location')
            receiver = self.value(root, depth + 1)
            if receiver.kind == 'event' and prop == 'target':
                return Value('dom', receiver.text, receiver.links)
            if receiver.kind == 'dom' and prop == 'style':
                return Value('style')
            raise Unproved('unrecognized property read')
        if kind == 'call':
            self.calls.append(node)
            try:
                return self.call(node, depth + 1)
            finally:
                self.calls.pop()
        raise Unproved('expression is outside the closed value proof')

    def call(self, node, depth):
        callee, *args = node.children
        if callee.kind == 'id':
            fn = self.binding(callee.value, callee)
            if not isinstance(fn, Node) or fn.kind != 'function':
                raise Unproved('callee is not a unique local declaration')
            category, roles = self.pure_family(fn)
            if category == 'loop':
                if len(args) != 3 or args[0].kind != 'id':
                    raise Unproved('decoder arguments are not complete')
                p = self.binding(args[0].value, args[0])
                if not isinstance(p, Node) or self.pure_family(p) != ('provider', roles):
                    raise Unproved('decoder/provider link not proven')
                data, output = (self.value(a, depth + 1) for a in args[1:])
                if data.kind != 'string' or data.text is None or output != Value('string', '', output.links):
                    raise Unproved('decoder input/accumulator not exact')
                from orbit.runtime.analysis_ioc_xor import decode
                text = decode(data.text, roles)
                return Value('string', text, (*data.links, *output.links, self.link('decoder', fn), self.link('provider', p), self.link('decode_call', node)))
            if category == 'return' and not args:
                for statement in fn.children[0].children[:-1]:
                    self.value(statement.children[0], depth + 1)
                result = self.value(fn.children[0].children[-1].children[0], depth + 1)
                return Value(result.kind, result.text, (*result.links, self.link('wrapper', fn), self.link('call', node)))
            raise Unproved('unsupported pure function invocation')
        if callee.kind != 'member':
            raise Unproved('indirect/computed call is not supported')
        root, method = callee.children[0], callee.value
        if root.kind == 'id' and root.value == 'document' and method == 'getElementById':
            if len(args) != 1 or self.value(args[0], depth + 1).kind != 'string':
                raise Unproved('DOM lookup argument is not exact')
            identifier = self.value(args[0], depth + 1).text
            origins = self.elements.get(identifier, [])
            if len(origins) != 1 or origins[0][1] >= node.start:
                raise Unproved('DOM lookup target is not uniquely present at this source point')
            links = ()
            if len(origins) == 1:
                _tag, lo, hi, _leaf = origins[0]
                links = (self.link('dom_origin', Node('element', None, (), lo, hi)), self.link('dom_lookup', node))
            return Value('dom', identifier, links)
        if (root.kind == 'member' and root.value == 'localStorage'
                and root.children[0].kind == 'id' and root.children[0].value == 'window'
                and method in ('getItem', 'setItem')):
            if len(args) != (1 if method == 'getItem' else 2):
                raise Unproved('invalid storage operation')
            if any(self.value(a, depth + 1).kind not in ('string', 'number', 'boolean') for a in args):
                raise Unproved('storage coercion is not a primitive')
            return Value('runtime_string')
        receiver = self.value(root, depth + 1)
        if receiver.kind in ('string', 'runtime_string'):
            if method == 'toLocaleLowerCase' and not args:
                return Value('runtime_string')
            if method == 'includes' and len(args) == 1 and self.value(args[0], depth + 1).kind == 'string':
                return Value('boolean')
        if receiver.kind == 'dom' and method == 'addEventListener' and len(args) == 2:
            if self.value(args[0], depth + 1).kind != 'string':
                raise Unproved('event name is not literal')
            self.handler(args[1], receiver)
            return Value('effect')
        raise Unproved('call effect is not in the closed supported set')

    def handler(self, fn, receiver):
        if fn.kind != 'function' or fn.value[0] or len(fn.value[1]) != 1:
            raise Unproved('unsupported event handler binding')
        scope = self.context[id(fn.children[0])]
        origin = self.elements.get(receiver.text, [])
        # event.target may name a descendant; only a leaf DOM element is proven.
        target = receiver.text if len(origin) == 1 and origin[0][3] else None
        scope.params[fn.value[1][0]] = Value('event', target, (*receiver.links, self.link('event_handler', fn)))
        self.check(fn.children[0].children)

    def check(self, nodes):
        for node in nodes:
            if node.kind == 'function':
                category, _roles = self.pure_family(node)
                if category == 'return':
                    for statement in node.children[0].children:
                        self.value(statement.children[0])
            elif node.kind == 'var':
                self.value(node.children[0])
            elif node.kind == 'block':
                self.check(node.children)
            elif node.kind == 'if':
                condition = self.value(node.children[0])
                # Check every branch for unsupported effects, including an
                # unreachable branch. Only a proven condition excludes sinks.
                self.check(node.children[1:])
                truth = self.truth(condition)
                if truth is not None:
                    dead = node.children[2:] if truth else node.children[1:2]
                    self.excluded_branches.extend((n.start, n.end) for n in dead)
                else:
                    self.discovery_issues.append('conditional reachability is not statically established')
                    if len(node.children) == 3:
                        self.ambiguous_branches.extend((n.start, n.end) for n in node.children[1:])
            elif node.kind == 'expr':
                expr = node.children[0]
                if expr.kind == 'binary' and expr.value == '=':
                    lhs, rhs = expr.children
                    if lhs.kind != 'member':
                        raise Unproved('unproven reassignment effect')
                    receiver = self.value(lhs.children[0])
                    if receiver.kind == 'dom' and lhs.value in ('onmouseover', 'onclick'):
                        self.handler(rhs, receiver)
                    elif receiver.kind == 'dom' and lhs.value in ('href', 'host', 'hostname'):
                        # A sink's RHS is an expression with effects too. It
                        # cannot be ignored while proving a later destination.
                        self.value(rhs)
                        origin = self.elements.get(receiver.text, [])
                        if receiver.text != '@location' and not (len(origin) == 1 and origin[0][0] in ('a', 'area')):
                            raise Unproved('receiver is not proven to be a URL-bearing element/location')
                        self.sinks.append((lhs, rhs))
                        self.sink_origins[id(lhs)] = receiver.links
                    elif ((receiver.kind == 'dom' and lhs.value == 'download')
                          or (receiver.kind == 'style' and lhs.value == 'display')):
                        if self.value(rhs).kind not in ('string', 'primitive'):
                            raise Unproved('property coercion is not a string')
                    else:
                        raise Unproved('property effect is not supported')
                else:
                    self.value(expr)
            else:
                raise Unproved('statement effect is outside the closed supported set')
        self.effects_checked = True


    @staticmethod
    def truth(value):
        if value.kind == 'boolean' and value.text in ('true', 'false'):
            return value.text == 'true'
        if value.kind == 'string' and value.text is not None:
            return bool(value.text)
        if value.kind == 'number' and value.text is not None:
            return int(value.text) != 0
        if value.kind == 'dom' and value.text is not None:
            return True
        return None


def scan_destinations(source):
    """Prove operands and independently certify complete supported discovery."""
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    if len(source) > MAX_INPUT_CHARS:
        return DestinationScan(source_hash, (), False, 'source exceeds static destination scan bound')
    view, external, elements, scripts, html_issues = _script_view(source)
    issues = list(html_issues)
    masked = _fcc_mask_code(view)
    candidates = list(re.finditer(r'\.\s*(href|host|hostname)\s*=(?!=)', masked))
    error = None
    proof = None
    try:
        if external:
            raise Unproved('external/inline script effects are not locally closed')
        if '\u2028' in view or '\u2029' in view:
            raise Unproved('unsupported JavaScript line separator')
        if len(scripts) > 128:
            raise Unproved('script count bound reached')
        nodes = tuple(node for lo, hi in scripts
                      for node in Parser(' ' * lo + view[lo:hi]).program())
        if sum(1 for node in nodes for _ in walk(node)) > MAX_NODES:
            raise Unproved('static syntax bound reached')
        proof = Proof(source, nodes, elements, scripts)
        proof.check(nodes)
        issues.extend(proof.discovery_issues)
    except (Unproved, RecursionError) as exc:
        error = str(exc) or 'static nesting bound reached'
        issues.append(error)
    result = []
    for m in candidates[:32]:
        if proof is not None and error is None and any(lo <= m.start() < hi for lo, hi in proof.excluded_branches):
            continue
        end = masked.find(';', m.end())
        end = len(view) if end < 0 else end
        value, links = None, ()
        state, reason = 'OPEN', error or 'local destination dependencies are not yet proven'
        if proof is not None and error is None:
            pair = next(((lhs, rhs) for lhs, rhs in proof.sinks if lhs.end == m.start() + len(m[0].rstrip().rstrip('=').rstrip())), None)
            # Match property token position rather than variable-name spelling.
            if pair is None:
                pair = next(((lhs, rhs) for lhs, rhs in proof.sinks if lhs.start <= m.start() < lhs.end and rhs.start >= m.end()), None)
            if pair:
                lhs, rhs = pair
                try:
                    if any(lo <= lhs.start < hi for lo, hi in proof.ambiguous_branches):
                        raise Unproved('alternative destination reachability is not established')
                    v = proof.value(rhs)
                    if v.kind == 'runtime_string' or v.kind == 'parameter':
                        state, reason = 'BLOCKED', 'destination depends on runtime input absent from the snapshot'
                    elif v.kind == 'string' and v.text is not None:
                        if lhs.value == 'href' and uris_in(v.text) != [v.text]:
                            if '://' not in v.text:
                                issues.append('relative destination is outside the closed value proof')
                                continue  # literal relative link, not network IoC
                            raise Unproved('absolute destination is incomplete')
                        if lhs.value != 'href':
                            import ipaddress
                            try:
                                ipaddress.ip_address(v.text)
                            except ValueError:
                                if not re.fullmatch(r'[A-Za-z0-9](?:[A-Za-z0-9.-]*\.)[A-Za-z]{2,63}', v.text):
                                    raise Unproved('host is not a complete value')
                        value, state, reason = v.text, 'RESOLVED_EXACT', ''
                        links = tuple(dict.fromkeys((*v.links, *proof.sink_origins[id(lhs)],
                            proof.link('effect_scope', Node('program', None, (), 0, len(source))),
                            proof.link('sink', Node('sink', lhs.value, (), lhs.start, rhs.end)))))
                    else:
                        raise Unproved('destination has no exact string value')
                except Unproved as exc:
                    reason = str(exc)
                    issues.append(reason)
            else:
                issues.append('candidate sink is not structurally classified')
        result.append(Destination(len(source[:m.start()].encode()), len(source[:end].encode()), m[1], state, reason, value, links))
    if len(candidates) > 32:
        issues.append('destination scan bound reached')
        m = candidates[32]
        result.append(Destination(len(source[:m.start()].encode()), len(source.encode()), 'scan', 'OPEN', 'destination scan bound reached'))
    return DestinationScan(source_hash, tuple(result), not issues, '; '.join(dict.fromkeys(issues)))


def destinations(source):
    """Compatibility view: per-sink proof is not a discovery certificate."""
    return list(scan_destinations(source).objectives)
