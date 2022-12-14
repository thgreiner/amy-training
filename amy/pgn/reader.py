#!/usr/bin/env python3

import random
import re
from typing import Tuple

import chess.pgn
from prometheus_client import Counter

from amy.chess.representation import Repr2D
from amy.queueing import MAX_PRIO, PrioritizedItem

repr2d = Repr2D()

repetitions = 0


def randomize_item(item):
    item.priority = random.randint(0, MAX_PRIO)
    return item


# Counter for monitoring no. of games
game_counter = Counter("training_game_total", "Games seen by training", ["result"])


def pos_generator(filename, test_mode, queue):
    sample_rate = 100 if test_mode else 16

    cnt = 0
    with open(filename) as pgn:

        positions_created = 0
        while True:
            try:
                game = chess.pgn.read_game(pgn)
            except UnicodeDecodeError or ValueError:
                continue
            if game is None:
                break

            result = game.headers["Result"]
            date_of_game = game.headers["Date"]

            game_counter.labels(result=result).inc()

            cnt += 1
            if cnt % 100 == 0:
                print(
                    f"Parsing game #{cnt} {date_of_game}, {positions_created} positions (avg {positions_created / cnt:.1f} pos/game)",
                    end="\r",
                )

            positions_created += _traverse_game(
                game, game.board(), queue, result, sample_rate
            )

    print(
        f"Parsed {cnt} games, {positions_created} positions (avg {positions_created / cnt:.1f} pos/game)."
    )
    print(f"Repetitions suppressed: {repetitions}")

    queue.put(end_of_input_item())


def end_of_input_item():
    return PrioritizedItem(MAX_PRIO, None, None, None, None, None)


def _traverse_game(game, board, queue, result, sample_rate):
    global repetitions

    positions_created = 0
    pos_map = dict()

    moves_remaining = len([x for x in game.mainline()])

    for node in game.mainline():

        move = node.move

        if node.comment:

            q, policy = _parse_mcts_result(node.comment)
            q = q * 2 - 1.0
            z = _label_for_result(result, board.turn)

            # q = 0.5 * (q + z[0] - z[2])

            train_data_board = repr2d.board_to_array(board)
            train_labels1 = repr2d.policy_to_array(board, policy)

            item = PrioritizedItem(
                random.randint(0, MAX_PRIO),
                train_data_board,
                train_labels1,
                q,
                z,
                moves_remaining,
            )

            moves_remaining -= 1

            key = board._transposition_key()
            if key in pos_map:
                repetitions += 1

            pos_map[key] = item

        if move is not None:
            board.push(move)

    for item in pos_map.values():
        if random.randint(0, 99) < sample_rate:
            queue.put(item)
            positions_created += 1

    return positions_created


def _label_for_result(result: str, turn: bool) -> list:
    if result == "1-0":
        if turn:
            return [1, 0, 0]
        else:
            return [0, 0, 1]
    if result == "0-1":
        if turn:
            return [0, 0, 1]
        else:
            return [1, 0, 0]

    return [0, 1, 0]


_Q_AND_POLICY_RE = re.compile("q=(.*); p=\[(.*)\]")
_MOVE_AND_COUNT_RE = re.compile("(.*):(.*)")


def _parse_mcts_result(mcts_result: str) -> Tuple[float, dict]:
    """Parse the output of MCTS into a q value and a dictionary of moves and their visit counts."""
    m = _Q_AND_POLICY_RE.match(mcts_result)

    if m is None:
        return None, None

    q = float(m.group(1))

    variations = m.group(2).split(", ")

    v = {}
    for variation in variations:
        m2 = _MOVE_AND_COUNT_RE.match(variation)
        if m2 is not None:
            v[m2.group(1)] = float(m2.group(2))

    return q, v
