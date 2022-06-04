import functools
import math
import time


class Piece():
    def __init__(self, team, has_moved):
        assert team in {-1, 1}
        assert type(has_moved) is bool
        self.team = team
        self.has_moved = has_moved

class Pawn(Piece):
    def __init__(self, *args):
        super().__init__(*args)

class Rook(Piece):
    def __init__(self, *args):
        super().__init__(*args)

class Knight(Piece):
    def __init__(self, *args):
        super().__init__(*args)

class Bishop(Piece):
    def __init__(self, *args):
        super().__init__(*args)

class Queen(Piece):
    def __init__(self, *args):
        super().__init__(*args)

class Prince(Piece):
    def __init__(self, *args):
        super().__init__(*args)

class King(Piece):
    def __init__(self, *args, castles = []):
        super().__init__(*args)
        self.castles = castles #list of rook starting positions, king move positions and rook move positions for castling




VALID_SQUARES = {Pawn, Rook, Knight, Bishop, Queen, Prince, King}



def generate_abstract_board_class(num_squares, flat_nbs, flat_opp, diag_nbs, diag_opp, pawn_moves, starting_layout):
    assert type(num_squares) == int and num_squares >= 0
    for piece, idx in starting_layout.items():
        assert type(piece) in VALID_SQUARES
        assert 0 <= idx < num_squares

    #flat_nbs : return the immediate neigbours of a square from the pov of a rook
    #flat_opp : return all possible next steps of a rook after moving from one square to a neigbour
    #diag_nbs : return the immediate neigbours of a square from the pov of a bishop
    #diag_opp : return all possible next steps of a bishop after moving from one square to a neigbour
    #pawn_moves : return a list of tuples (m1, [m2, ..., m2], [p, ... , p]) where
    #   m1 is a single step foward
    #   [m2, ..., m2] is a list of possible followup moves foward
    #starting_layout : return the piece at each square at the start of the game

    #for night moves - we need to know how to change to a perpendicular flat move
    def flat_nopp(i, j):
        opps = set(flat_opp(i, j))
        for k in flat_nbs(j):
            if not k in opps and k != i:
                yield k

    def knight_moves(a):
        #we must do f2s1 AND s1f2 to get all knight moves in wormhole chess
        #flat 2 side 1
        for b in flat_nbs(a):
            for c in flat_opp(a, b):
                for d in flat_nopp(b, c):
                    yield d
        #side 1 flat 2
        for b in flat_nbs(a):
            for c in flat_nopp(a, b):
                for d in flat_opp(b, c):
                    yield d

    def king_moves(idx):
        yield from flat_nbs(idx)
        yield from diag_nbs(idx)

    def pawn_attacks(team, idx):
        for m1, m2s in pawn_moves(team, idx):
            for a in flat_nopp(idx, m1):
                yield a
                
    #generate all possible sliding moves from a given square
    #note that there may be branching, for example in wormhole chess there can
    #be multiple continuations of the same initial slide
    #we must also take care to avoid infinite loops, for example round the wormhole
    #if an infinite loop occurs, we end with the starting point (so effectively a null move can be played)
    def gen_slide(idx, nbs, opp):
        def rest_slide(start, i, j):
            #yield all possible continuations of the slide [..., i] through j
            if j == start:
                #we found an loop
                yield [j]
            else:
                #not found a loop, so look at all possible next steps and all continuations of the slide from that next step
                none = True
                for k in opp(i, j):
                    none = False
                    for slide in rest_slide(start, j, k):
                        yield [j] + slide
                if none:
                    yield [j]
        for jdx in nbs(idx):
            for slide in rest_slide(idx, idx, jdx):
                yield tuple(slide)

    #information about whats defended and where things can move in theory
    class MovementInfo():
        def __init__(self, piece, at_idx, sees_idx):
            self.piece = piece
            self.at_idx = at_idx
            self.sees_idx = sees_idx
            
    #actual moves which can be made
    class ActualMove():
        def __init__(self, select_idx, perform_idx, from_board, to_board, is_capture):
            self.select_idx = select_idx
            self.perform_idx = perform_idx
            self.from_board = from_board
            self.to_board = to_board
            self.is_capture = is_capture

        @functools.cache
        def is_legal(self):
            #make this more efficient:
            #if in check, can only do 1) move king out of danger 2) block / take attacking piece
            #if not in check, can only not do 1) move a pinned piece 2) move king into check 3) do en passant and remove pinned enemy pawn
            #checking the above would be much faster than generating all moves on the next board to see if we are in check afer each move
            if self.to_board.is_checked(self.from_board.turn):
                return False
            return True
            

    class OutOfTime(Exception):
        pass


    class Board():
        #a slide is a tuple of moves a sliding piece can make assuming an empty board
        #in particuar, a slide does not contain the starting position (unless it is included as the end of a loop e.g. a loop round the wormhole ending at the starting position
        FLAT_SLIDE = {idx : tuple(gen_slide(idx, flat_nbs, flat_opp)) for idx in range(num_squares)} #tuple of slides
        DIAG_SLIDE = {idx : tuple(gen_slide(idx, diag_nbs, diag_opp)) for idx in range(num_squares)} #tuple of slides
        KNIGHT_MOVES = {idx : tuple(set(knight_moves(idx))) for idx in range(num_squares)} #tuple of knight moves
        KING_MOVES = {idx : tuple(set(king_moves(idx))) for idx in range(num_squares)} #tuple of king moves
        PAWN_MOVES = {team : {idx : tuple(tuple([m1, tuple(m2s)]) for m1, m2s in pawn_moves(team, idx)) for idx in range(num_squares)} for team in {-1, 1}}
        PAWN_ATTACKS = {team : {idx : tuple(set(pawn_attacks(team, idx))) for idx in range(num_squares)} for team in {-1, 1}}

        SEARCH_START_TIME = time.time()
        SEARCH_TIME_ALLOWED = 0.0

        @classmethod
        def starting_board(cls):
            return cls(0, starting_layout, 1)

        def __init__(self, num, pieces, turn):
            self.king_idx = {} #where is the king for each player
            self.num = num
            self.pieces = pieces
            self.idx_piece_lookup = {}
            for piece, idx in pieces.items():
                assert not idx in self.idx_piece_lookup
                self.idx_piece_lookup[idx] = piece
                if type(piece) == King:
                    assert not piece.team in self.king_idx
                    self.king_idx[piece.team] = idx
            self.turn = turn

            self.current_best_depth = 0
            self.current_best_move = None

        def cache_clear(self):
            self._get_move_info.cache_clear()
            self.get_moves.cache_clear()


        @functools.cache
        @lambda f : lambda self : tuple(f(self))
        def _get_move_info(self):
            actual_moves = []
            seen_by = {idx : set() for idx in range(num_squares)}
            movecount = {1 : 0, -1 : 0}

            def new_info(from_piece, from_idx, to_idx):
                info = MovementInfo(from_piece, from_idx, to_idx)
                seen_by[to_idx].add(info)
                movecount[from_piece.team] += 1

            def new_move(from_piece, from_idx, to_piece, to_idx):
                pieces = {piece : idx for piece, idx in self.pieces.items()}
                del pieces[from_piece]
                pieces[to_piece] = to_idx
                actual_moves.append(ActualMove(from_idx, to_idx, self, Board(self.num + 1, pieces, -self.turn), False))

            def new_take(from_piece, from_idx, to_piece, to_idx, take_piece):
                pieces = {piece : idx for piece, idx in self.pieces.items()}
                del pieces[from_piece]
                del pieces[take_piece]
                pieces[to_piece] = to_idx
                actual_moves.append(ActualMove(from_idx, to_idx, self, Board(self.num + 1, pieces, -self.turn), True))

            def do_slides(piece, idx, slides):
                for slide in slides:
                    for move_idx in slide:
                        new_info(piece, idx, move_idx)
                        if move_idx in self.idx_piece_lookup:
                            blocking_piece = self.idx_piece_lookup[move_idx]
                            if piece.team == self.turn and blocking_piece.team != piece.team:
                                new_take(piece, idx, type(piece)(piece.team, True), move_idx, blocking_piece)
                            break
                        else:
                            if piece.team == self.turn:
                                new_move(piece, idx, type(piece)(piece.team, True), move_idx)

            def do_teleports(piece, idx, move_idxs):
                for move_idx in move_idxs:
                    new_info(piece, idx, move_idx)
                    if move_idx in self.idx_piece_lookup:
                        blocking_piece = self.idx_piece_lookup[move_idx]
                        if piece.team == self.turn and blocking_piece.team != piece.team:
                            new_take(piece, idx, type(piece)(piece.team, True), move_idx, blocking_piece)
                    else:
                        if piece.team == self.turn:
                            new_move(piece, idx, type(piece)(piece.team, True), move_idx)

            for piece, idx in self.pieces.items():
                if type(piece) == Pawn:
                    if piece.team == self.turn:
                        for move1, move2s in self.PAWN_MOVES[piece.team][idx]:
                            #pawn move 1 foward
                            if not move1 in self.idx_piece_lookup:
                                new_move(piece, idx, Pawn(piece.team, True), move1)
                                #pawn move 2 foward
                                for move2 in move2s:
                                    if not move2 in self.idx_piece_lookup:
                                        new_move(piece, idx, Pawn(piece.team, True), move2)
                    #pawn attack
                    for move_idx in self.PAWN_ATTACKS[piece.team][idx]:
                        new_info(piece, idx, move_idx)
                        if move_idx in self.idx_piece_lookup:
                            blocking_piece = self.idx_piece_lookup[move_idx]
                            if piece.team == self.turn and blocking_piece.team != piece.team:
                                new_take(piece, idx, Pawn(piece.team, True), move_idx, blocking_piece)

                if type(piece) in {Rook, Queen}:
                    do_slides(piece, idx, self.FLAT_SLIDE[idx])
                if type(piece) in {Bishop, Queen}:
                    do_slides(piece, idx, self.DIAG_SLIDE[idx])
                if type(piece) in {Knight}:
                    do_teleports(piece, idx, self.KNIGHT_MOVES[idx])
                if type(piece) in {Prince, King}:
                    do_teleports(piece, idx, self.KING_MOVES[idx])
                                
            return actual_moves, seen_by, movecount

        def get_pseudo_moves(self):
            return self._get_move_info()[0]

        def seen_by(self, idx):
            return self._get_move_info()[1][idx]

        @functools.cache
        def is_checked(self, team):
            for info in self.seen_by(self.king_idx[team]):
                if info.piece.team != team:
                    return True
            return False

        @functools.cache
        def get_moves(self):
            return tuple(move for move in self.get_pseudo_moves() if move.is_legal())

        @functools.cache
        def static_eval(self, team):
            total = 0
            #piece value
            for piece in self.pieces:
                total += piece.team * {Pawn : 1, Rook : 5, Knight : 3, Bishop : 3, Queen : 9, King : 1000}[type(piece)]
            #avalable moves
            for t, c in self._get_move_info()[2].items():
                total += 0.1 * t * c
            return total * team

        def quiesce(self, alpha, beta):
            if time.time() - type(self).SEARCH_START_TIME > self.SEARCH_TIME_ALLOWED:
                raise OutOfTime()
            return self.static_eval(self.turn)

        @functools.cache
        def alpha_beta(self, alpha, beta, depth):            
            if depth == 0:
                return self.quiesce(alpha, beta)
            for move in sorted(self.get_moves(), key = lambda move : move.to_board.static_eval(move.to_board.turn)):
                score = -move.to_board.alpha_beta(-beta, -alpha, depth - 1)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            return alpha
        
        def alpha_beta_root(self, depth):
            assert depth >= 1
            alpha = -math.inf
            best_move = None
            for move in self.get_moves():
                score = -move.to_board.alpha_beta(-math.inf, -alpha, depth - 1)
                if score > alpha:
                    alpha = score
                    best_move = move
            return best_move

        def best_move_search(self, dt):
            type(self).SEARCH_START_TIME = time.time()
            type(self).SEARCH_TIME_ALLOWED = dt
            try:
                while True:
                    self.current_best_move = self.alpha_beta_root(self.current_best_depth + 1)
                    self.current_best_depth += 1
                    print(f"depth = {self.current_best_depth}")
            except OutOfTime:
                pass
                        

    return Board


    



if __name__ == "__main__":
    board = AbstractBoard([Pawn(-1, True)], 1)
    print(board)
    






















