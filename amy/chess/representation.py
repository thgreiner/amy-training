from typing import Sequence, Tuple

import chess
from numpy import float32, int8, sum, zeros
from scipy.sparse import csr_matrix


class Repr2D:
    """A helper class to map chess boards and moves to two-dimensional matrix representations."""

    def __init__(self):
        queen_dirs = [-1, 1, -8, 8, -7, 7, -9, 9]
        knight_dirs = [-15, 15, -17, 17, -10, 10, -6, 6]
        pawn_dirs = [7, 8, 9]
        self.queen_indexes = dict()
        self.knight_indexes = dict()
        self.underpromo_indexes = {
            chess.KNIGHT: dict(),
            chess.BISHOP: dict(),
            chess.ROOK: dict(),
        }

        idx = 0

        for dir in queen_dirs:
            for i in range(1, 8):
                delta = i * dir
                self.queen_indexes[delta] = idx
                idx += 1

        for delta in knight_dirs:
            self.knight_indexes[delta] = idx
            idx += 1

        for delta in pawn_dirs:
            for piece in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
                self.underpromo_indexes[piece][delta] = idx
                idx += 1

        self.num_planes = 19

        # print(f"Generated {idx} indexes for moves.")

    def plane_index(self, move, xor):
        delta = (move.to_square ^ xor) - (move.from_square ^ xor)
        if move.promotion and move.promotion != chess.QUEEN:
            return self.underpromo_indexes[move.promotion][delta]
        elif _is_knight_move(move.from_square, move.to_square):
            return self.knight_indexes[delta]
        else:
            return self.queen_indexes[delta]

    def board_to_array(self, board):
        buf = zeros((8, 8, self.num_planes), int8)

        turn = board.turn

        _store_board(board, buf, turn)

        offset = 12

        if board.ep_square:
            xor = 0 if turn else 0x38
            rank, file = _coords(board.ep_square ^ xor)
            buf[rank, file, offset] = 1

        if board.has_kingside_castling_rights(board.turn):
            buf[:, :, offset + 1] = 1
        if board.has_queenside_castling_rights(board.turn):
            buf[:, :, offset + 2] = 1
        if board.has_kingside_castling_rights(not board.turn):
            buf[:, :, offset + 3] = 1
        if board.has_queenside_castling_rights(not board.turn):
            buf[:, :, offset + 4] = 1

        # One plane just ones so the network can detect the board edge
        buf[:, :, offset + 5] = 1

        buf[:, :, offset + 6] = board.halfmove_clock / 100.0

        return buf

    def move_to_array(self, b, move):
        buf = zeros((64, 73), int8)
        xor = 0 if b.turn else 0x38
        buf[move.to_square ^ xor, self.plane_index(move, xor)] = 1
        return buf.reshape(4672)

    def policy_to_array(self, b, policy):
        buf = zeros((64, 73), float32)

        xor = 0 if b.turn else 0x38

        for san, value in policy.items():
            move = b.parse_san(san)
            buf[move.to_square ^ xor, self.plane_index(move, xor)] = value

        buf = buf / sum(buf)
        return csr_matrix(buf)


def _is_knight_move(from_square: int, to_square: int) -> bool:
    file_dist = abs((from_square & 7) - (to_square & 7))
    rank_dist = abs((from_square >> 3) - (to_square >> 3))
    return (file_dist == 1 and rank_dist == 2) or (file_dist == 2 and rank_dist == 1)


def _coords(sq: int) -> Tuple:
    return (sq >> 3, sq & 7)


def _store_board(board: chess.Board, buf: Sequence, turn: bool) -> None:
    xor = 0 if turn else 0x38

    offset = 0
    for piece_type in range(1, 7):
        squares = board.pieces(piece_type, turn)
        for sq in squares:
            rank, file = _coords(sq ^ xor)
            buf[rank, file, offset] = 1

        offset += 1

        squares = board.pieces(piece_type, not turn)
        for sq in squares:
            rank, file = _coords(sq ^ xor)
            buf[rank, file, offset] = 1

        offset += 1
