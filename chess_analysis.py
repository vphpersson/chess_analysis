#!/usr/bin/env python

from asyncio import run as asyncio_run
from argparse import ArgumentError
from sys import stderr
from typing import Iterable, Type, Sequence
from pathlib import Path
from json import dumps as json_dumps, loads as json_loads, JSONEncoder
from dataclasses import asdict, is_dataclass
from datetime import datetime
from itertools import chain
from contextlib import suppress, closing
from pickle import dumps as pickle_dumps, loads as pickle_loads
from base64 import b64encode
from sys import setrecursionlimit
from collections import Counter, defaultdict

from httpx import AsyncClient
from chess import Color as ChessColor, WHITE, BLACK, square_name, Board
from chess.engine import popen_uci
from chess_com_extractor import login, get_archived_game_entries, get_pgn_info
from chess_com_extractor.structures import PGNInfo
from graphviz import Digraph
from chess_analysis.trie import ChessPositionTrieNode
from terminal_utils.progressor import Progressor

from chess_analysis.cli import ChessAnalysisArgumentParser


setrecursionlimit(3000)


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
    moves_depth: int = 30
):
    """

    :param trie_node:
    :param pgn_info_list:
    :param player_name:
    :param player_color:
    :param engine:
    :param moves_depth: The number of moves per game to include.
    :return:
    """

    if player_color == WHITE:
        filter_func = lambda x: x.player1_name == player_name
    elif player_color == BLACK:
        filter_func = lambda x: x.player2_name == player_name
    else:
        # TODO: Use Proper exception.
        raise ValueError

    with Progressor() as progressor:
        filtered_pgn_info_list = list(filter(filter_func, pgn_info_list))

        for i, pgn_info_entry in enumerate(filtered_pgn_info_list):
            progressor.print_progress(
                iteration=i,
                total=len(filtered_pgn_info_list)
            )
            await trie_node.add_game_moves(
                moves=pgn_info_entry.moves[:moves_depth],
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
        username='virrevvv',
        password='mamma123',
        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36',
        http_client=http_client
    )

    return await get_pgn_info(
        archived_game_entries=await get_archived_game_entries(
            http_client=http_client,
            player_name=player_name,
            game_type='live',
            game_sub_types=['blitz'],
            num_max_pages=10
        ),
        http_client=http_client,
        csrf_token=csrf_token,
        # as_dict=as_dict
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


def make_graph(
    root,
    chess_color: ChessColor,
    archived_game_entries,
    player_name: str,
    game_ids: set[int]
):

    dot = Digraph()
    dot.graph_attr['archived_game_entries'] = json_dumps(
        [asdict(entry) for entry in archived_game_entries],
        cls=CustomJsonEncoder
    )
    dot.graph_attr['player_name'] = player_name
    dot.graph_attr['player_color'] = 'white' if chess_color == WHITE else 'black'

    observed_node_ids: set[str] = set()

    def iter_node(node: ChessPositionTrieNode, depth: int):
        node_id: str = str(id(node))
        if node_id in observed_node_ids:
            return

        if not game_ids.intersection(node.game_ids):
            return

        move_counter = Counter()
        for game_id, move in node.game_id_to_move.items():
            if game_id in game_ids:
                move_counter.update([move])

        custom_node_attributes = dict(
            game_id_to_move=json_dumps({
                game_id: dict(
                    from_square=square_name(move.from_square),
                    promotion=move.promotion,
                    to_square=square_name(move.to_square),
                )
                for game_id, move in node.game_id_to_move.items()
                if game_id in game_ids
            }, separators=(',', ':')),
            game_id_to_parent=json_dumps({
                game_id: id(parent_node)
                for game_id, parent_node in node.game_id_to_parent.items()
                if game_id in game_ids
            }, separators=(',', ':')),
            game_ids=json_dumps(list(game_id for game_id in node.game_ids if game_id in game_ids), separators=(',', ':')),
            centipawn_value=str(node.centipawn_value),
            best_move=json_dumps(
                dict(
                    centipawn_value=node.best_move.centipawn_value,
                    from_square=square_name(node.best_move.from_square),
                    promotion=node.best_move.promotion,
                    to_square=square_name(node.best_move.to_square)
                ),
                separators=(',', ':')
            ) if node.best_move else None,
            move_counter=json_dumps({
                b64encode(
                    s=json_dumps(
                        dict(
                            from_square=square_name(move.from_square),
                            promotion=move.promotion,
                            to_square=square_name(move.to_square),
                        ),
                        separators=(',', ':')
                    ).encode()
                ).decode(): count
                for move, count in move_counter.items()
            })
        )

        dot.node(
            name=node_id,
            label=f'{custom_node_attributes["centipawn_value"]}',
            fillcolor=('white' if depth % 2 == 0 else 'black'),
            style='filled',
            fontcolor=('white' if depth % 2 == 1 else 'black'),
            width=str(s),
            height=str(s),
            fixedsize='true',
            **custom_node_attributes
        )
        observed_node_ids.add(node_id)

        for move, child_node in node.move_to_situation.items():

            if game_ids and not game_ids.intersection(child_node.game_ids):
                continue

            iter_node(node=child_node, depth=depth + 1)

            choice_ratio: float = node.move_counter[move] / len(list(node.move_counter.elements()))

            best_color, bad_color = ('green', 'red') if depth % 2 == (0 if chess_color == WHITE else 1) else ('blue', 'orange')

            custom_edge_attributes = dict(
                move=json_dumps(
                    dict(
                        from_square=square_name(move.from_square),
                        promotion=move.promotion,
                        to_square=square_name(move.to_square),
                    )
                ),
                move_count=str(node.move_counter[move]),
                choice_ratio=str(choice_ratio),
                best_move_centipawn_difference=str(round(abs(child_node.best_move_centipawn_difference(parent=node)) / 100, 2))
            )

            dot.edge(
                tail_name=node_id,
                head_name=str(id(child_node)),
                label=str(move),
                color=(best_color if str(move) == str(node.best_move) else bad_color),
                penwidth=str(8 * choice_ratio),
                fontcolor=('red' if abs(child_node.best_move_centipawn_difference(parent=node) > 50) else 'black'),
                **custom_edge_attributes
            )

    iter_node(node=root, depth=0)

    Path('/tmp/pls.dot').write_text(dot.source)


async def get_root(
    player_name: str,
    player_color: ChessColor,
    trie_path: Path,
    chess_com_games: list[PGNInfo]
):
    # TODO: Use proper exception.
    root: ChessPositionTrieNode = ChessPositionTrieNode()
    with suppress(Exception):
        root = pickle_loads(trie_path.read_bytes())

    transport, engine = await popen_uci("/usr/bin/stockfish")
    await engine.configure({
        'threads': 20,
        'hash': 2048
    })
    with closing(transport):
        await add_games_to_trie(
            trie_node=root,
            pgn_info_list=chess_com_games,
            player_name=player_name,
            player_color=player_color,
            engine=engine
        )

    trie_path.write_bytes(pickle_dumps(root))

    return root


def s(
    trie_root,
    pgn_info_list: Sequence[PGNInfo],
    player_name: str,
    color: ChessColor,
    occurrence_threshold: int = 3
):
    if color == WHITE:
        pgn_info_filter_func = lambda x: x.player1_name == player_name
        turn_mod = 0
    elif color == BLACK:
        pgn_info_filter_func = lambda x: x.player2_name == player_name
        turn_mod = 1
    else:
        # TODO: Use proper exception.
        raise ValueError

    game_ids = [pgn_info.game_id for pgn_info in pgn_info_list if pgn_info_filter_func(pgn_info)]

    bad_move_sequence_to_game_ids = defaultdict(set)

    for game_id in list(game_ids):
        turn = 0
        current_node = trie_root
        board = Board()

        while True:
            if move := current_node.game_id_to_move.get(game_id):
                board.push(move=move)
                next_node = current_node.move_to_situation.get(move)

                if turn % 2 == turn_mod:
                    if (diff := abs(next_node.best_move_centipawn_difference(parent=current_node))) >= 50:
                        bad_move_sequence_to_game_ids[(diff, board.fen())].add(game_id)
                        break

                current_node = next_node
                turn += 1
            else:
                break

    for (diff, fen), game_ids in bad_move_sequence_to_game_ids.items():
        if (occurrence := len(game_ids)) >= occurrence_threshold:
            print(f'{occurrence:2d} {diff:3d} {fen} https://www.chess.com/game/live/{list(game_ids)[0]}')


async def main():
    try:
        args: Type[ChessAnalysisArgumentParser.Namespace] = ChessAnalysisArgumentParser().parse_args()
    except ArgumentError as e:
        print(e, file=stderr)
        exit(1)

    if args.player_color == 'white':
        color = WHITE
    elif args.player_color == 'black':
        color = BLACK
    else:
        # TODO: Use proper exception.
        raise ValueError(f'Bad color: {args.player_color}')

    if args.username and args.password:
        async with AsyncClient(base_url='https://www.chess.com') as http_client:
            chess_com_games = await retrieve_chess_com_games(
                username=args.username,
                password=args.password,
                player_name=args.player_name,
                http_client=http_client
            )
    else:
        chess_com_games = []

    full_pgn_info_list: list[PGNInfo] = list({
        pgn_info_entry.game_id: pgn_info_entry
        for pgn_info_entry in chain(
            chess_com_games or [],
            retrieve_saved_games_info(path=args.saved_games_path)
        )
    }.values())

    args.saved_games_path.write_text(json_dumps(full_pgn_info_list, cls=CustomJsonEncoder))

    if args.num_most_recent is not None:
        full_pgn_info_list.sort(key=lambda x: x.end_time)
        full_pgn_info_list = full_pgn_info_list[-args.num_most_recent:]

    root = await get_root(
        player_name=args.player_name,
        player_color=color,
        trie_path=args.trie_path,
        chess_com_games=full_pgn_info_list
    )

    game_ids: set[int] = set(pgn_info.game_id for pgn_info in full_pgn_info_list)

    s(trie_root=root, pgn_info_list=full_pgn_info_list, player_name='virrevvv', color=color)

    make_graph(
        root=root,
        chess_color=color,
        player_name=args.player_name,
        archived_game_entries=full_pgn_info_list,
        game_ids=game_ids
    )


if __name__ == '__main__':
    asyncio_run(main())
