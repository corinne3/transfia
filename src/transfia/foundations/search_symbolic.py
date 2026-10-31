"""IA symbolique : recherche dans un espace d'états, et système expert à règles.

Avant l'apprentissage automatique, l'IA, c'était (et c'est encore, en partie) :
1. la RECHERCHE : trouver un chemin d'un état initial à un état but.
   - BFS (parcours en largeur) : plus court chemin en NOMBRE d'étapes (graphe non pondéré).
   - A* : f(n) = g(n) + h(n), coût réel depuis le départ + estimation HEURISTIQUE jusqu'au but.
     Si h ne surestime jamais (heuristique ADMISSIBLE, ex. distance de Manhattan sur une grille),
     A* trouve le chemin optimal en explorant bien moins d'états que BFS.
2. le RAISONNEMENT par RÈGLES (systèmes experts) : SI conditions ALORS conclusion.
   Chaînage AVANT : on part des faits et on applique les règles jusqu'à ne plus rien déduire.
   Explicable par construction (on sait quelle règle a conclu quoi).
"""
from __future__ import annotations

import heapq
from collections import deque

Grid = list[str]
Pos = tuple[int, int]


def neighbors(grid: Grid, p: Pos) -> list[Pos]:
    y, x = p
    out = []
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ny, nx = y + dy, x + dx
        if 0 <= ny < len(grid) and 0 <= nx < len(grid[0]) and grid[ny][nx] != "#":
            out.append((ny, nx))
    return out


def bfs(grid: Grid, start: Pos, goal: Pos) -> tuple[list[Pos] | None, int]:
    """Renvoie (chemin, nombre d'états explorés)."""
    parent: dict[Pos, Pos | None] = {start: None}
    q = deque([start])
    explored = 0
    while q:
        cur = q.popleft()
        explored += 1
        if cur == goal:
            break
        for nb in neighbors(grid, cur):
            if nb not in parent:
                parent[nb] = cur
                q.append(nb)
    if goal not in parent:
        return None, explored
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1], explored


def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: Grid, start: Pos, goal: Pos) -> tuple[list[Pos] | None, int]:
    open_heap = [(manhattan(start, goal), 0, start)]      # (f, g, état)
    g_cost = {start: 0}
    parent: dict[Pos, Pos | None] = {start: None}
    explored = 0
    while open_heap:
        f, g, cur = heapq.heappop(open_heap)
        if g > g_cost.get(cur, float("inf")):
            continue                                          # entrée périmée dans le tas
        explored += 1
        if cur == goal:
            path, node = [], goal
            while node is not None:
                path.append(node)
                node = parent[node]
            return path[::-1], explored
        for nb in neighbors(grid, cur):
            ng = g + 1
            if ng < g_cost.get(nb, float("inf")):
                g_cost[nb] = ng
                parent[nb] = cur
                heapq.heappush(open_heap, (ng + manhattan(nb, goal), ng, nb))
    return None, explored


# ------------------------------------------------------------- système expert
RULES = [
    ({"lait_grumeleux"}, "mammite_clinique"),
    ({"quartier_gonfle", "fievre"}, "mammite_clinique"),
    ({"cellules_somatiques_elevees"}, "mammite_subclinique"),
    ({"mammite_clinique"}, "appeler_veterinaire"),
    ({"mammite_subclinique", "litiere_humide"}, "ameliorer_litiere"),
    ({"mammite_subclinique"}, "verifier_machine_a_traire"),
]


def forward_chaining(facts: set[str], rules=RULES) -> tuple[set[str], list[str]]:
    """Applique les règles jusqu'au point fixe ; renvoie (faits, explication)."""
    facts = set(facts)
    trace = []
    changed = True
    while changed:
        changed = False
        for conditions, conclusion in rules:
            if conditions <= facts and conclusion not in facts:
                facts.add(conclusion)
                trace.append(f"SI {' ET '.join(sorted(conditions))} ALORS {conclusion}")
                changed = True
    return facts, trace


def demo() -> None:
    grid = ["S....#....",
            ".##..#.##.",
            ".#.......#",
            ".#.###.#..",
            "...#...#.G"]
    start, goal = (0, 0), (4, 9)
    p1, e1 = bfs(grid, start, goal)
    p2, e2 = astar(grid, start, goal)
    print(f"BFS : longueur {len(p1) - 1}, états explorés {e1}")
    print(f"A*  : longueur {len(p2) - 1}, états explorés {e2}  (même optimum, moins d'exploration)")
    facts, trace = forward_chaining({"cellules_somatiques_elevees", "litiere_humide"})
    print("Système expert :")
    for t in trace:
        print("  ", t)


if __name__ == "__main__":
    demo()
