#!/usr/bin/env python

from asyncio import run as asyncio_run
from argparse import ArgumentError
from sys import stderr
from typing import Iterable, Optional, Type
from pathlib import Path
from json import dumps as json_dumps, loads as json_loads, JSONEncoder
from dataclasses import asdict, is_dataclass
from datetime import datetime
from itertools import chain
from contextlib import suppress, closing
from pickle import dumps as pickle_dumps, loads as pickle_loads

from httpx import AsyncClient
from chess import Color as ChessColor, WHITE, BLACK
from chess.engine import popen_uci
from chess_com_extractor import login, get_archived_game_entries, get_pgn_info
from chess_com_extractor.structures import PGNInfo
from graphviz import Digraph
from chess_analysis.trie import ChessPositionTrieNode
from terminal_utils.progressor import Progressor

from chess_analysis.cli import ChessAnalysisArgumentParser


class CustomJsonEncoder(JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        elif isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.astimezone().isoformat()
        return super().default(obj)


async def add_games_to_trie(
    trie_node: ChessPositionTrieNode,
    pgn_info_list: Iterable[PGNInfo],
    player_name: str,
    player_color: ChessColor,
    engine,
):
    """

    :param trie_node:
    :param pgn_info_list:
    :param player_name:
    :param player_color:
    :param engine:
    :return:
    """

    if player_color == WHITE:
        filter_func = lambda x: x.player1_name == player_name
    elif player_color == BLACK:
        filter_func = lambda x: x.player2_name == player_name
    else:
        raise ValueError

    with Progressor() as progressor:
        filtered_pgn_info_list = list(filter(filter_func, pgn_info_list))

        for i, pgn_info_entry in enumerate(filtered_pgn_info_list):
            progressor.print_progress(
                iteration=i,
                total=len(filtered_pgn_info_list)
            )
            await trie_node.add_game_moves(
                moves=pgn_info_entry.moves[:14],
                game_id=pgn_info_entry.game_id,
                engine=engine
            )


async def retrieve_chess_com_games(
    username: str,
    password: str,
    player_name: str,
    http_client: AsyncClient
):
    csrf_token: str = await login(
        username=username,
        password=password,
        http_client=http_client
    )

    return await get_pgn_info(
        archived_game_entries=await get_archived_game_entries(http_client=http_client, player_name=player_name),
        http_client=http_client,
        csrf_token=csrf_token
    )


def retrieve_saved_games_info(path: Path):

    def object_hook(obj: dict) -> PGNInfo:
        for key, value in obj.items():
            if isinstance(value, str):
                try:
                    obj[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass

        return PGNInfo(**obj)

    saved_pgn_info_list: list[PGNInfo] = []
    with suppress(OSError):
        saved_pgn_info_list: list[PGNInfo] = json_loads(path.read_text(), object_hook=object_hook)

    return saved_pgn_info_list


def make_graph(root, chess_color: ChessColor, game_ids: Optional[set[str]] = None):

    dot = Digraph()

    observed_node_ids: set[str] = set()

    def iter_node(node: ChessPositionTrieNode, depth: int):
        node_id: str = str(id(node))
        if node_id in observed_node_ids:
            return

        if game_ids and not game_ids.intersection(node.game_ids):
            return

        s = 1.5

        dot.node(
            name=node_id,
            label=f'{round(node.centipawn_value / 100, 3)}; {node.best_move}; {len(node.game_ids)}',
            fillcolor=('white' if depth % 2 == 0 else 'black'),
            style='filled',
            fontcolor=('white' if depth % 2 == 1 else 'black'),
            width=str(s),
            height=str(s),
            fixedsize='true'
        )
        observed_node_ids.add(node_id)

        for move, child_node in node.move_to_situation.items():

            if game_ids and not game_ids.intersection(child_node.game_ids):
                continue

            iter_node(node=child_node, depth=depth + 1)

            choice_ratio: float = node.move_counter[move] / len(list(node.move_counter.elements()))

            best_color, bad_color = ('green', 'red') if depth % 2 == (0 if chess_color == WHITE else 1) else ('blue', 'orange')

            dot.edge(
                tail_name=node_id,
                head_name=str(id(child_node)),
                label=f'{move} ({node.move_counter[move]}; {round(choice_ratio, 2)}); {round(abs(child_node.best_move_centipawn_difference(parent=node)) / 100, 2)}',
                color=(best_color if str(move) == str(node.best_move) else bad_color),
                penwidth=str(8 * choice_ratio),
                fontcolor=('red' if abs(child_node.best_move_centipawn_difference(parent=node) > 50) else 'black')
            )

    iter_node(node=root, depth=0)

    dot.render('/tmp/pls.gv', format='pdf')


async def get_root(
    player_name: str,
    player_color: ChessColor,
    trie_path: Path,
    saved_games_path: Path,
    username: Optional[str] = None,
    password: Optional[str] = None,
    num_most_recent: Optional[int] = None,
):

    # TODO: Use proper exception.
    root: ChessPositionTrieNode = ChessPositionTrieNode()
    with suppress(Exception):
        root = pickle_loads(trie_path.read_bytes())

    if username and password:
        async with AsyncClient(base_url='https://www.chess.com') as http_client:
            chess_com_games = await retrieve_chess_com_games(
                username=username,
                password=password,
                player_name=player_name,
                http_client=http_client
            )
    else:
        chess_com_games = []

    full_pgn_info_list: list[PGNInfo] = list({
        pgn_info_entry.game_id: pgn_info_entry
        for pgn_info_entry in chain(
            chess_com_games,
            retrieve_saved_games_info(path=saved_games_path)
        )
    }.values())

    saved_games_path.write_text(json_dumps(full_pgn_info_list, cls=CustomJsonEncoder))

    if num_most_recent is not None:
        full_pgn_info_list.sort(key=lambda x: x.end_time)
        full_pgn_info_list = full_pgn_info_list[-num_most_recent:]

    transport, engine = await popen_uci("/usr/bin/stockfish")
    with closing(transport):
        await add_games_to_trie(
            trie_node=root,
            pgn_info_list=full_pgn_info_list,
            player_name=player_name,
            player_color=player_color,
            engine=engine
        )

    trie_path.write_bytes(pickle_dumps(root))

    return root


async def main():
    try:
        args: Type[ChessAnalysisArgumentParser.Namespace] = ChessAnalysisArgumentParser().parse_args()
    except ArgumentError as e:
        print(e, file=stderr)
        exit(1)
        return

    if args.player_color == 'white':
        color = WHITE
    elif args.player_color == 'black':
        color = BLACK
    else:
        # TODO: Use proper exception.
        raise ValueError(f'Bad color: {args.player_color}')

    root = await get_root(
        player_name=args.player_name,
        player_color=color,
        trie_path=args.trie_path,
        saved_games_path=args.saved_games_path,
        username=args.username,
        password=args.password,
        num_most_recent=args.num_most_recent,
    )

    make_graph(root=root, chess_color=color)


if __name__ == '__main__':
    asyncio_run(main())
