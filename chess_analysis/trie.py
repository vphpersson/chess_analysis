from __future__ import annotations
from typing import Optional, Iterable, ClassVar, Union
from dataclasses import dataclass, field, asdict
from collections import Counter

from chess import Move, Color as ChessColor, Board
from chess.engine import Limit


@dataclass(unsafe_hash=True)
class BestMove(Move):
    centipawn_value: int = 0


@dataclass
class ChessPositionTrieNode:

    ANALYSIS_DEPTH: ClassVar[int] = 20
    MATE_SCORE: ClassVar[int] = 10_000

    root_node: Optional[ChessPositionTrieNode] = None
    turn_and_position_to_position_node: dict[tuple[ChessColor, str], ChessPositionTrieNode] = field(default_factory=dict)

    move_to_situation: dict[Move, ChessPositionTrieNode] = field(default_factory=dict)
    game_id_to_move: dict[int, Move] = field(default_factory=dict)
    game_id_to_parent: dict[int, ChessPositionTrieNode] = field(default_factory=dict)
    game_ids: set[int] = field(default_factory=set)

    move_counter: Counter = field(default_factory=Counter)

    centipawn_value: int = 0
    best_move: Optional[BestMove] = None

    def __post_init__(self):
        if not self.root_node:
            self.root_node = self

        self.turn_and_position_to_position_node = self.root_node.turn_and_position_to_position_node

    def is_best_move(self, parent: ChessPositionTrieNode) -> bool:

        if parent is None:
            return True

        best_move_dict = asdict(parent.best_move)
        best_move_dict.pop('centipawn_value')

        return parent.move_to_situation.get(Move(**best_move_dict)) is self

    def best_move_centipawn_difference(self, parent: ChessPositionTrieNode) -> int:
        return 0 \
            if self.is_best_move(parent=parent) \
            else parent.best_move.centipawn_value - self.centipawn_value

    async def add_game_moves(self, moves: Iterable[Move], game_id: int, engine) -> None:
        """
        Populate a Chess position Trie with moves of a game.

        Each position is analysed to find the best move. If the position has been observed before, there is no need to
        perform a new analysis as a previous result is used, already stored in the node.

        :param moves: An iterable of moves to be evaluated.
        :param engine: An engine with which to analyze the moves.
        :param game_id: The ID the game that the moves constituted.
        :return: None
        """

        board = Board()
        current_situation_trie_node = self.root_node

        starting_info = None

        for i, move in enumerate(moves, start=1):

            # print(f'{i=} {move=}')

            # Update the usage counters of the current position node. Only do this once for each game.
            if game_id not in current_situation_trie_node.game_ids:
                current_situation_trie_node.move_counter.update([move])

            # Find and analyse the best move.

            if current_situation_trie_node.best_move is None:
                # Find the best move for the current position. If the last move resulted in a new node, we can use the
                # analysis results of that position.
                best_move: Move = (starting_info or await engine.analyse(board, Limit(depth=self.ANALYSIS_DEPTH)))['pv'][0]

                board.push(best_move)

                # Assign the best move value to the current position node. If the position that the best move yields has
                # already been observed, use its centipawn value, otherwise perform an analysis with the best move.
                current_situation_trie_node.best_move = BestMove(
                    **asdict(best_move),
                    centipawn_value=(
                        best_move_position.centipawn_value
                        if (best_move_position := self.turn_and_position_to_position_node.get((board.turn, str(board))))
                        else (await engine.analyse(board=board, limit=Limit(depth=20)))['score'].white().score(mate_score=self.MATE_SCORE)
                    )
                )

                # Restore the to a state where only played move are on the stack.
                board.pop()

            # Obtain a node for the played position.

            board.push(move=move)

            played_move_position_node: ChessPositionTrieNode = self.turn_and_position_to_position_node.get((board.turn, str(board))) \
               or current_situation_trie_node.move_to_situation.get(move)

            if not played_move_position_node:
                played_move_info = await engine.analyse(board, Limit(depth=self.ANALYSIS_DEPTH))

                played_move_position_node = ChessPositionTrieNode(
                    root_node=self.root_node,
                    centipawn_value=played_move_info['score'].white().score(mate_score=self.MATE_SCORE)
                )

                starting_info = played_move_info
            else:
                starting_info = None

            # Add data to the current position node and the played position node.

            # Add the played position to the global lookup.
            self.turn_and_position_to_position_node[(board.turn, str(board))] = played_move_position_node

            # Add to Trie navigation lookup; child.
            current_situation_trie_node.move_to_situation[move] = played_move_position_node
            current_situation_trie_node.game_id_to_move[game_id] = move

            # Mark the current position as having been observed in the game with the provided ID.
            current_situation_trie_node.game_ids.add(game_id)

            # Add to Trie navigation lookup; parent.
            played_move_position_node.game_id_to_parent[game_id] = current_situation_trie_node

            # Set the played position node as the current position node, preparing for the next iteration.
            current_situation_trie_node = played_move_position_node

    def situation_from_moves(self, moves: Iterable[Union[str, Move]]) -> ChessPositionTrieNode:

        node = self
        for move in moves:
            node = node.move_to_situation[move if isinstance(move, Move) else Move.from_uci(move)]

        return node
