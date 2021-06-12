from __future__ import annotations
from argparse import ArgumentError
from pathlib import Path
from typing import Optional, Type

from pyutils.argparse.typed_argument_parser import TypedArgumentParser


class ChessAnalysisArgumentParser(TypedArgumentParser):

    class Namespace:
        player_name: Optional[str]
        player_color: str
        username: Optional[str]
        password: Optional[str]
        saved_games_path: Optional[Path]
        trie_path: Optional[Path]
        num_most_recent: Optional[int]

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **(
                dict(
                    description="Output a Trie of a player's chess games."
                ) | kwargs
            )
        )

        self.add_argument(
            'player_name',
            help='The name of the player from whose perspective the Trie is.'
        )

        self.add_argument(
            'player_color',
            help='The color of the player whose Trie to output.',
            choices=['black', 'white']
        )

        self.add_argument(
            '--username',
            help='An username with which to authenticate.'
        )

        self.add_argument(
            '--password',
            help='A password with which to authenticate.'
        )

        self.add_argument(
            '--saved-games-path',
            help='The file path where cached games are to be stored.',
            default=Path('saved_games.json')
        )

        self.add_argument(
            '--trie-path',
            help='The file path where the Trie is to be stored.',
        )

        self.add_argument(
            '--num-most-recent',
            help='The number of recent games to analyze',
            type=int
        )

    def parse_args(self, *args, **kwargs) -> Type[ChessAnalysisArgumentParser.Namespace]:
        namespace: Type[ChessAnalysisArgumentParser.Namespace] = super().parse_args(*args, **kwargs)

        if namespace.trie_path is None:
            setattr(namespace, 'trie_path', Path(f'chess_trie_{namespace.player_color}.pickle'))

        if namespace.username is not None and namespace.password is None:
            raise ArgumentError(
                argument=next(action for action in self._actions if action.dest == 'password'),
                message='A username has been provided but the password is unset.'
            )

        if namespace.password is not None and namespace.username is None:
            raise ArgumentError(
                argument=next(action for action in self._actions if action.dest == 'password'),
                message='A password has been provided but the username is unset.'
            )

        return namespace
