from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, FrozenSet

# Normalizacion y postfix
OPERATORS = {"|", ".", "*"}
PRECEDENCE = {"|": 1, ".": 2, "*": 3}
RIGHT_ASSOC = {"*"}  # operador unario postfix

def is_symbol(ch: str) -> bool:
    return ch.isalnum() or ch == "#"

def tokenize(regex: str) -> List[str]:
    tokens = []
    i = 0
    while i < len(regex):
        c = regex[i]
        if c.isspace():
            i += 1
            continue
        if c in {"(", ")", "|", "*"}:
            tokens.append(c)
        else:
            # Simbolos de un solo caracter
            if len(c) != 1:
                raise ValueError(f"Simbolo invalido: {c}")
            tokens.append(c)
        i += 1
    return tokens

def needs_concat(a: str, b: str) -> bool:
    # concatenacion implicita: (simbolo o ')' o '*') seguido de (simbolo o '(')
    left_ok = is_symbol(a) or a == ")" or a == "*"
    right_ok = is_symbol(b) or b == "("
    return left_ok and right_ok

def insert_concatenation(tokens: List[str]) -> List[str]:
    if not tokens:
        return tokens
    out = [tokens[0]]
    for t in tokens[1:]:
        prev = out[-1]
        if needs_concat(prev, t):
            out.append(".")
        out.append(t)
    return out

def to_postfix(tokens: List[str]) -> List[str]:
    output: List[str] = []
    stack: List[str] = []

    for t in tokens:
        if is_symbol(t):
            output.append(t)
        elif t == "(":
            stack.append(t)
        elif t == ")":
            while stack and stack[-1] != "(":
                output.append(stack.pop())
            if not stack:
                raise ValueError("Parentesis mal")
            stack.pop()  # cerrar paren
        elif t in OPERATORS:
            if t == "*":
                # Postfix con maxima precedencia
                while stack and stack[-1] in OPERATORS and PRECEDENCE[stack[-1]] > PRECEDENCE[t]:
                    output.append(stack.pop())
                stack.append(t)
            else:
                while stack and stack[-1] in OPERATORS:
                    top = stack[-1]
                    if (PRECEDENCE[top] > PRECEDENCE[t]) or (
                        PRECEDENCE[top] == PRECEDENCE[t] and t not in RIGHT_ASSOC
                    ):
                        output.append(stack.pop())
                    else:
                        break
                stack.append(t)
        else:
            raise ValueError(f"Token desconocido: {t}")

    while stack:
        op = stack.pop()
        if op in {"(", ")"}:
            raise ValueError("Parentesis mal")
        output.append(op)

    return output



# AST para construccion directa de AFD
@dataclass
class Node:
    kind: str  # 'leaf', 'or', 'cat', 'star'
    left: Optional["Node"] = None
    right: Optional["Node"] = None
    symbol: Optional[str] = None
    pos: Optional[int] = None

    nullable: bool = False
    firstpos: Set[int] = None
    lastpos: Set[int] = None

    def __post_init__(self):
        if self.firstpos is None:
            self.firstpos = set()
        if self.lastpos is None:
            self.lastpos = set()

def build_ast_from_postfix(postfix: List[str]) -> Node:
    stack: List[Node] = []
    pos_counter = 0

    for t in postfix:
        if is_symbol(t):
            pos_counter += 1
            stack.append(Node(kind="leaf", symbol=t, pos=pos_counter))
        elif t == "*":
            if not stack:
                raise ValueError("Estrella sin operando")
            child = stack.pop()
            stack.append(Node(kind="star", left=child))
        elif t == ".":
            if len(stack) < 2:
                raise ValueError("Concatenacion sin operandos")
            right = stack.pop()
            left = stack.pop()
            stack.append(Node(kind="cat", left=left, right=right))
        elif t == "|":
            if len(stack) < 2:
                raise ValueError("Alternancia sin operandos")
            right = stack.pop()
            left = stack.pop()
            stack.append(Node(kind="or", left=left, right=right))
        else:
            raise ValueError(f"Token postfix desconocido: {t}")

    if len(stack) != 1:
        raise ValueError("Postfix invalido")
    return stack[0]

def compute_nullable_first_last(n: Node) -> None:
    if n.kind == "leaf":
        n.nullable = False  # sin hojas epsilon
        n.firstpos = {n.pos}
        n.lastpos = {n.pos}
        return

    if n.kind == "or":
        compute_nullable_first_last(n.left)
        compute_nullable_first_last(n.right)
        n.nullable = n.left.nullable or n.right.nullable
        n.firstpos = n.left.firstpos | n.right.firstpos
        n.lastpos = n.left.lastpos | n.right.lastpos
        return

    if n.kind == "cat":
        compute_nullable_first_last(n.left)
        compute_nullable_first_last(n.right)
        # concatenacion: firstpos/lastpos dependen de la anulabilidad
        n.nullable = n.left.nullable and n.right.nullable
        if n.left.nullable:
            n.firstpos = n.left.firstpos | n.right.firstpos
        else:
            n.firstpos = set(n.left.firstpos)
        if n.right.nullable:
            n.lastpos = n.left.lastpos | n.right.lastpos
        else:
            n.lastpos = set(n.right.lastpos)
        return

    if n.kind == "star":
        compute_nullable_first_last(n.left)
        # estrella: siempre anulable
        n.nullable = True
        n.firstpos = set(n.left.firstpos)
        n.lastpos = set(n.left.lastpos)
        return

    raise ValueError(f"Nodo desconocido: {n.kind}")

def compute_followpos(n: Node, followpos: Dict[int, Set[int]]) -> None:
    if n.kind == "leaf":
        return

    if n.kind == "cat":
        # followpos: lastpos(izq) -> firstpos(der)
        for i in n.left.lastpos:
            followpos[i] |= n.right.firstpos
        compute_followpos(n.left, followpos)
        compute_followpos(n.right, followpos)
        return

    if n.kind == "star":
        # followpos en estrella: lastpos(hijo) a firstpos(hijo)
        for i in n.left.lastpos:
            followpos[i] |= n.left.firstpos
        compute_followpos(n.left, followpos)
        return

    if n.kind == "or":
        compute_followpos(n.left, followpos)
        compute_followpos(n.right, followpos)
        return

    raise ValueError(f"Nodo desconocido: {n.kind}")

def collect_pos_to_symbol(n: Node, mapping: Dict[int, str]) -> None:
    if n.kind == "leaf":
        mapping[n.pos] = n.symbol
        return
    if n.left:
        collect_pos_to_symbol(n.left, mapping)
    if n.right:
        collect_pos_to_symbol(n.right, mapping)


# Construccion directa de AFD
@dataclass
class DFA:
    states: List[FrozenSet[int]]
    start: FrozenSet[int]
    finals: Set[FrozenSet[int]]
    delta: Dict[Tuple[FrozenSet[int], str], FrozenSet[int]]
    alphabet: List[str]

def build_direct_dfa(regex_infix: str) -> Tuple[DFA, Dict[int, str], Dict[int, Set[int]]]:
    # Extender con # y concatenacion explicita
    raw = tokenize(regex_infix)
    # (r).# para marcar posicion final
    raw_extended = ["("] + raw + [")", "#"]
    tokens = insert_concatenation(raw_extended)

    # Infix a postfix
    postfix = to_postfix(tokens)

    # Postfix a AST
    ast = build_ast_from_postfix(postfix)

    # nullable/firstpos/lastpos
    compute_nullable_first_last(ast)

    # followpos
    pos_to_symbol: Dict[int, str] = {}
    collect_pos_to_symbol(ast, pos_to_symbol)

    followpos: Dict[int, Set[int]] = {p: set() for p in pos_to_symbol.keys()}
    compute_followpos(ast, followpos)

    # Alfabeto sin '#'
    alphabet = sorted({s for s in pos_to_symbol.values() if s != "#"})

    # Posicion de '#'
    hash_positions = [p for p, s in pos_to_symbol.items() if s == "#"]
    if len(hash_positions) != 1:
        raise ValueError("Se esperaba exactamente un '#' en la expresion")
    hash_pos = hash_positions[0]

    # Construccion de estados: mover por simbolo usando followpos
    start_state = frozenset(ast.firstpos)
    states: List[FrozenSet[int]] = [start_state]
    finals: Set[FrozenSet[int]] = set()
    delta: Dict[Tuple[FrozenSet[int], str], FrozenSet[int]] = {}

    if hash_pos in start_state:
        finals.add(start_state)

    i = 0
    while i < len(states):
        S = states[i]
        for a in alphabet:
            U: Set[int] = set()
            for p in S:
                if pos_to_symbol[p] == a:
                    U |= followpos[p]
            U_fs = frozenset(U)
            delta[(S, a)] = U_fs

            if U_fs and U_fs not in states:
                states.append(U_fs)
                if hash_pos in U_fs:
                    finals.add(U_fs)
        i += 1

    return DFA(states=states, start=start_state, finals=finals, delta=delta, alphabet=alphabet), pos_to_symbol, followpos

def simulate_dfa(dfa: DFA, w: str) -> bool:
    state = dfa.start
    for ch in w:
        if ch not in dfa.alphabet:
            return False
        state = dfa.delta.get((state, ch), frozenset())
        if not state:
            return False
    return state in dfa.finals

def format_state_label(state: FrozenSet) -> str:
    if not state:
        return "[]"
    sample = next(iter(state))
    if isinstance(sample, int):
        return str(sorted(state))
    inner = [sorted(list(s)) for s in state]
    inner.sort()
    return str(inner)

def print_dfa(dfa: DFA, title: str = "AFD") -> None:
    # Nombrar estados en orden 
    name: Dict[FrozenSet, str] = {st: f"S{i}" for i, st in enumerate(dfa.states)}

    print(title)
    print("Alfabeto:", "{" + ", ".join(dfa.alphabet) + "}")
    print("Estados:")
    for st in dfa.states:
        tag = []
        if st == dfa.start:
            tag.append("inicial")
        if st in dfa.finals:
            tag.append("final")
        extra = f" ({', '.join(tag)})" if tag else ""
        print(f"  {name[st]} = {format_state_label(st)}{extra}")

    print("Transiciones:")
    for st in dfa.states:
        for a in dfa.alphabet:
            to_st = dfa.delta.get((st, a), frozenset())
            to_name = name.get(to_st, "VACIO")
            print(f"  {name[st]} --{a}--> {to_name}")

def ensure_complete_dfa(dfa: DFA) -> DFA:
    dead = frozenset()
    states = list(dfa.states)
    delta = dict(dfa.delta)
    needs_dead = False

    # Completar transiciones hacia estado muerto
    for st in states:
        for a in dfa.alphabet:
            to_st = delta.get((st, a), frozenset())
            if not to_st:
                needs_dead = True
                delta[(st, a)] = dead

    if needs_dead and dead not in states:
        states.append(dead)
        for a in dfa.alphabet:
            delta[(dead, a)] = dead

    return DFA(states=states, start=dfa.start, finals=set(dfa.finals), delta=delta, alphabet=dfa.alphabet)

def minimize_dfa(dfa: DFA) -> DFA:
    complete = ensure_complete_dfa(dfa)
    states = set(complete.states)
    finals = set(complete.finals)
    nonfinals = states - finals

    # Particion inicial: finales y no finales
    partitions: List[Set[FrozenSet[int]]] = []
    if finals:
        partitions.append(set(finals))
    if nonfinals:
        partitions.append(set(nonfinals))

    worklist: List[Set[FrozenSet[int]]] = [p.copy() for p in partitions]

    while worklist:
        A = worklist.pop()
        for c in complete.alphabet:
            # X: estados que transitan con c hacia A
            X = {s for s in states if complete.delta[(s, c)] in A}
            if not X:
                continue
            new_partitions: List[Set[FrozenSet[int]]] = []
            for Y in partitions:
                # dividir Y segun interseccion con X
                inter = Y & X
                diff = Y - X
                if inter and diff:
                    new_partitions.append(inter)
                    new_partitions.append(diff)
                    if Y in worklist:
                        worklist.remove(Y)
                        worklist.append(inter)
                        worklist.append(diff)
                    else:
                        if len(inter) <= len(diff):
                            worklist.append(inter)
                        else:
                            worklist.append(diff)
                else:
                    new_partitions.append(Y)
            partitions = new_partitions

    # Mapear estado original a un bloque equivalente
    block_of: Dict[FrozenSet[int], FrozenSet[FrozenSet[int]]] = {}
    for block in partitions:
        block_fs = frozenset(block)
        for st in block:
            block_of[st] = block_fs

    # Mantener orden estable de bloques
    ordered_blocks: List[FrozenSet[FrozenSet[int]]] = []
    seen_blocks: Set[FrozenSet[FrozenSet[int]]] = set()
    for st in complete.states:
        b = block_of[st]
        if b not in seen_blocks:
            ordered_blocks.append(b)
            seen_blocks.add(b)

    start_block = block_of[complete.start]
    final_blocks = {block_of[s] for s in finals}

    min_delta: Dict[Tuple[FrozenSet[FrozenSet[int]], str], FrozenSet[FrozenSet[int]]] = {}
    for block in ordered_blocks:
        rep = next(iter(block))
        for c in complete.alphabet:
            target = complete.delta[(rep, c)]
            min_delta[(block, c)] = block_of[target]

    return DFA(
        states=ordered_blocks,
        start=start_block,
        finals=final_blocks,
        delta=min_delta,
        alphabet=complete.alphabet,
    )

def dfa_to_dot(dfa: DFA, title: str, file_path: Path) -> None:
    name: Dict[FrozenSet, str] = {st: f"S{i}" for i, st in enumerate(dfa.states)}
    lines: List[str] = []
    lines.append("digraph DFA {")
    lines.append("  rankdir=LR;")
    lines.append("  labelloc=\"t\";")
    lines.append(f"  label=\"{title}\";")
    lines.append("  node [shape=circle];")
    lines.append("  __start [shape=point];")
    lines.append(f"  __start -> {name[dfa.start]};")

    for st in dfa.states:
        if st in dfa.finals:
            lines.append(f"  {name[st]} [shape=doublecircle];")

    for st in dfa.states:
        for a in dfa.alphabet:
            to_st = dfa.delta.get((st, a))
            if to_st is None:
                continue
            to_name = name.get(to_st, "VACIO")
            if to_name == "VACIO":
                continue
            lines.append(f"  {name[st]} -> {to_name} [label=\"{a}\"];" )

    lines.append("}")
    file_path.write_text("\n".join(lines), encoding="utf-8")


# CLI
def main() -> None:
    while True:
        r = input("Ingrese expresion regular r (o 'exit'): ").strip()
        if r.lower() == "exit":
            break
        if not r:
            continue

        dfa, _, _ = build_direct_dfa(r)
        min_dfa = minimize_dfa(dfa)

        print_dfa(dfa, title="AFD (construccion directa)")
        print_dfa(min_dfa, title="AFD minimizado")

        dfa_to_dot(ensure_complete_dfa(dfa), "AFD (directo)", Path("afd_directa.dot"))
        dfa_to_dot(min_dfa, "AFD (minimizado)", Path("afd_min.dot"))
        print("Se generaron: afd_directa.dot y afd_min.dot")

        while True:
            w = input("Ingrese cadena w (o 'exit' / 'new'): ").strip()
            if w.lower() == "exit":
                return
            if w.lower() == "new":
                break
            ok = simulate_dfa(dfa, w)
            print("ACEPTADA" if ok else "RECHAZADA")

if __name__ == "__main__":
    main()