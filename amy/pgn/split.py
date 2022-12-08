import os

import chess.pgn


def split_pgn_by_date(file_name: str) -> None:
    """Split a PGN file into separate files based on the "Date" tag in their headers.

    @Args
        file_name: the name of the file to split
    """
    cnt = 0
    with open(file_name) as pgn:
        while True:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            date = game.headers["Date"]
            output_dir = os.path.join("PGN", date)

            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            try:
                with open(f"{output_dir}/games.pgn", "a") as output:
                    print(game, file=output, end="\n\n")

                cnt += 1
                print(f"{cnt}: {date}", end="\r")
            except IOError:
                pass
