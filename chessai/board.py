import functools
import math
import time
import random


class Piece():
    VALUE = None
    def __init__(self, team, has_moved):
        assert team in {-1, 1}
        assert type(has_moved) is bool
        self.team = team
        self.has_moved = has_moved

class Pawn(Piece):
    VALUE = 1
    def __init__(self, *args, adv = 0):
        super().__init__(*args)
        self.adv = 0 #how many steps have we taken

class Rook(Piece):
    VALUE = 5
    def __init__(self, *args):
        super().__init__(*args)

class Knight(Piece):
    VALUE = 3
    def __init__(self, *args):
        super().__init__(*args)

class Bishop(Piece):
    VALUE = 3
    def __init__(self, *args):
        super().__init__(*args)

class Queen(Piece):
    VALUE = 9
    def __init__(self, *args):
        super().__init__(*args)

class Prince(Piece):
    VALUE = 4
    def __init__(self, *args):
        super().__init__(*args)

class King(Piece):
    VALUE = 1000
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

    class MovementInfoSlide(MovementInfo):
        def __init__(self, *args, slide_idx, slide):
            super().__init__(*args)
            self.slide_idx = slide_idx
            self.slide = slide
            
    #actual moves which can be made
    class ActualMove():
        def __init__(self, select_idx, perform_idx, from_board, to_board):
            self.select_idx = select_idx
            self.perform_idx = perform_idx
            self.from_board = from_board
            self.to_board = to_board

        @property
        def is_capture(self):
            return False

        def is_legal_slow(self):
            #make this more efficient:
            #if in check, can only do 1) move king out of danger 2) block / take attacking piece
            #if not in check, can only not do 1) move a pinned piece 2) move king into check 3) do en passant and remove pinned enemy pawn
            #checking the above would be much faster than generating all moves on the next board to see if we are in check afer each move
            return not self.to_board.is_checked(self.from_board.turn)

        @functools.cache
        def is_legal(self):
            return self.is_legal_slow()
        
    class ActualNormalMove(ActualMove):
        def __init__(self, *args, moving_piece, from_idx, to_idx, take_piece = None):
            super().__init__(*args)
            self.moving_piece = moving_piece
            self.from_idx = from_idx
            self.to_idx = to_idx
            self.take_piece = take_piece
            
        @property
        def is_capture(self):
            return not self.take_piece is None

        @functools.cache
        def is_legal(self):
            verify = False
            
            #we want to determine if the move is legal without running the expensive self.to_board._board_info
            #we can however use self.from_board._board_info - this is usually enough
            if self.to_board.is_checked(self.from_board.turn):
                if type(self.moving_piece) == King:
                    #are we moving into check
                    for info in self.from_board.seen_by(self.to_idx):
                        if info.piece.team != self.from_board.turn:
                            ans = False
                            if verify:
                                assert ans == self.is_legal_slow()
                            return ans
                    #are we moving into a check which is obscured by ourself (the continuation of a checking sliding piece)
                    for info in self.from_board.seen_by(self.from_board.king_idx[self.from_board.turn]):
                        if info.piece.team != self.from_board.turn:
                            if isinstance(info, MovementInfoSlide):
                                if info.slide_idx < len(info.slide) - 1:
                                    danger_idx = info.slide[info.slide_idx + 1]
                                    if self.to_idx == danger_idx:
                                        ans = False
                                        if verify:
                                            assert ans == self.is_legal_slow()
                                        return ans
                    #we are safe otherwise
                    ans = True
                    if verify:
                        assert ans == self.is_legal_slow()
                    return ans
                else:
                    ans = self.is_legal_slow()
                    if verify:
                        assert ans == self.is_legal_slow()
                    return ans
            else:
                if type(self.moving_piece) == King:
                    #are we moving into check
                    for info in self.from_board.seen_by(self.to_idx):
                        if info.piece.team != self.from_board.turn:
                            ans = False
                            if verify:
                                assert ans == self.is_legal_slow()
                            return ans
                    #we are safe otherwise
                    ans = True
                    if verify:
                        assert ans == self.is_legal_slow()
                    return ans
                else:
                    #are we pinned
                    for info in self.from_board.seen_by(self.from_idx):
                        if isinstance(info, MovementInfoSlide):
                            slide_idx = info.slide_idx
                            while slide_idx < len(info.slide) - 1:
                                slide_idx += 1
                                danger_idx = info.slide[slide_idx]
                                if danger_idx in self.to_board.pieces:
                                    hit = self.to_board.pieces[danger_idx]
                                    if type(hit) == King and hit.team == self.from_board.turn:
                                        ans = False
                                        if verify:
                                            assert ans == self.is_legal_slow()
                                        return ans
                                    break
                                
                    #otherwise we are safe
                    ans = True
                    if verify:
                        assert ans == self.is_legal_slow()
                    return ans
        

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
        MAX_Q_DEPTH = 0
        LEAF_COUNT = 0
        NODE_COUNT = 0

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

            self.best_known_score = 0
            self.best_known_score_depth = -1
            
            self.current_best_depth = 0
            self.current_best_move = None


        def cache_clear(self, ignore):
            if not self in ignore:
                if "_board_info" in self.__dict__:
                    for move in self._board_info[0]:
                        move.to_board.cache_clear(ignore)
                    del self._board_info


        @functools.cached_property
        @lambda f : lambda self : tuple(f(self))
        def _board_info(self):
            actual_moves = []
            seen_by = {idx : set() for idx in range(num_squares)}
            movement_score = {1 : 0, -1 : 0}

            def new_movement_score(info):
                if info.sees_idx in self.idx_piece_lookup:
                    sees_piece = self.idx_piece_lookup[info.sees_idx]
                    movement_score[info.piece.team] += 0.05 * max(5 - type(info.piece).VALUE - type(sees_piece).VALUE, 0) #good to defend/attack low value stuff with low value stuff
                else:
                    m_score = 0.05
                    
                movement_score[info.piece.team] += 0.02 #movement score
                
            def new_info(from_piece, from_idx, to_idx):
                info = MovementInfo(from_piece, from_idx, to_idx)
                seen_by[to_idx].add(info)
                new_movement_score(info)

            def new_slide_info(from_piece, from_idx, to_idx, slide_idx, slide):
                info = MovementInfoSlide(from_piece, from_idx, to_idx, slide_idx = slide_idx, slide = slide)
                seen_by[to_idx].add(info)
                new_movement_score(info)

            def new_move(from_piece, from_idx, to_piece, to_idx):
                pieces = {piece : idx for piece, idx in self.pieces.items()}
                del pieces[from_piece]
                pieces[to_piece] = to_idx
                actual_moves.append(ActualNormalMove(from_idx, to_idx, self, Board(self.num + 1, pieces, -self.turn), moving_piece = piece, from_idx = from_idx, to_idx = to_idx, take_piece = None))

            def new_take(from_piece, from_idx, to_piece, to_idx, take_piece):
                pieces = {piece : idx for piece, idx in self.pieces.items()}
                del pieces[from_piece]
                del pieces[take_piece]
                pieces[to_piece] = to_idx
                actual_moves.append(ActualNormalMove(from_idx, to_idx, self, Board(self.num + 1, pieces, -self.turn), moving_piece = piece, from_idx = from_idx, to_idx = to_idx, take_piece = take_piece))

            def do_slides(piece, idx, slides):
                for slide in slides:
                    for slide_idx, move_idx in enumerate(slide):
                        new_slide_info(piece, idx, move_idx, slide_idx, slide)
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
                                new_move(piece, idx, Pawn(piece.team, True, adv = piece.adv + 1), move1)
                                #pawn move 2 foward
                                for move2 in move2s:
                                    if not move2 in self.idx_piece_lookup:
                                        new_move(piece, idx, Pawn(piece.team, True, adv = piece.adv + 2), move2)
                    #pawn attack
                    for move_idx in self.PAWN_ATTACKS[piece.team][idx]:
                        new_info(piece, idx, move_idx)
                        if move_idx in self.idx_piece_lookup:
                            blocking_piece = self.idx_piece_lookup[move_idx]
                            if piece.team == self.turn and blocking_piece.team != piece.team:
                                new_take(piece, idx, Pawn(piece.team, True, adv = piece.adv + 1), move_idx, blocking_piece)

                if type(piece) in {Rook, Queen}:
                    do_slides(piece, idx, self.FLAT_SLIDE[idx])
                if type(piece) in {Bishop, Queen}:
                    do_slides(piece, idx, self.DIAG_SLIDE[idx])
                if type(piece) in {Knight}:
                    do_teleports(piece, idx, self.KNIGHT_MOVES[idx])
                if type(piece) in {Prince, King}:
                    do_teleports(piece, idx, self.KING_MOVES[idx])
                                
            return actual_moves, seen_by, movement_score

        def get_pseudo_moves(self):
            return self._board_info[0]

        def seen_by(self, idx):
            return self._board_info[1][idx]

        def is_checked(self, team):
            for info in self.seen_by(self.king_idx[team]):
                if info.piece.team != team:
                    return True
            return False

        def get_moves(self):
            return tuple(move for move in self.get_pseudo_moves() if move.is_legal())

        def update_best_known_score(self, score, depth):
            if depth >= self.best_known_score_depth:
                self.best_known_score_depth = depth
                self.best_known_score = score

        def get_moves_sorted(self, only_captures = False):
            #deepest searches first
            #then those with the least opponent score on the next move
            return sorted([move for move in self.get_moves() if move.is_capture or not only_captures], key = lambda move : (-move.to_board.best_known_score_depth, move.to_board.best_known_score))

        @functools.cache
        def static_eval(self, team):
            def compute_score():
                if len(self.get_moves()) == 0:
                    for info in self.seen_by(self.king_idx[self.turn]):
                        if info.piece.team == -self.turn:
                            #no moves & in check => checkmate. We loose
                            return -math.inf * team
                            return total
                    #no moves & not in check => draw
                    return 0
                else:
                    total = 0
                    #piece value
                    for piece in self.pieces:
                        total += piece.team * type(piece).VALUE
                        #pawn adv
                        if type(piece) == Pawn:
                            total += piece.team * 0.1 * piece.adv
                    #avalable moves
                    for t, sc in self._board_info[2].items():
                        total += t * sc
                    return team * total

            score = compute_score()
            self.update_best_known_score(score, 0)
            type(self).NODE_COUNT += 1
            type(self).LEAF_COUNT += 1
            return score

        def quiesce(self, alpha, beta, depth):
            type(self).MAX_Q_DEPTH = max(type(self).MAX_Q_DEPTH, depth)
            stand_pat = self.static_eval(self.turn)
            if stand_pat >= beta:
                return beta
            if alpha < stand_pat:
                alpha = stand_pat
            for move in self.get_moves_sorted(True):
                if stand_pat + type(move.take_piece).VALUE < alpha:
                    #delta prune - we can already do better than taking this piece elsewhere
                    return alpha
                score = -move.to_board.quiesce(-beta, -alpha, depth + 1)
                if score >= beta:
                    return beta
                if score > alpha:
                   alpha = score
            type(self).NODE_COUNT += 1
            return alpha


        @functools.cache
        def alpha_beta(self, alpha, beta, depthleft, depth):
            if time.time() - type(self).SEARCH_START_TIME > self.SEARCH_TIME_ALLOWED:
                raise OutOfTime()
            if depthleft == 0:
                return self.quiesce(alpha, beta, depth)
            for move in self.get_moves_sorted():
                score = -move.to_board.alpha_beta(-beta, -alpha, depthleft - 1, depth + 1)
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
            self.update_best_known_score(alpha, depthleft)
            type(self).NODE_COUNT += 1
            return alpha
        
        def alpha_beta_root(self, depthleft, depth = 0):
            assert depthleft >= 1
            alpha = -math.inf
            best_move = random.choice(self.get_moves())
                
            for move in self.get_moves_sorted():
                score = -move.to_board.alpha_beta(-math.inf, -alpha, depthleft - 1, depth + 1)
                if score > alpha:
                    alpha = score
                    best_move = move
            self.update_best_known_score(alpha, depthleft)
            type(self).NODE_COUNT += 1
            return best_move, alpha

        def best_move_search(self, dt):
            if len(self.get_moves()) != 0:
                type(self).SEARCH_START_TIME = time.time()
                type(self).SEARCH_TIME_ALLOWED = dt
                try:
                    while self.current_best_depth < 3:
                        self.current_best_move, score = self.alpha_beta_root(self.current_best_depth + 1)
                        self.current_best_depth += 1
                        print(f"depth = {self.current_best_depth}q{type(self).MAX_Q_DEPTH}, nodes = {type(self).NODE_COUNT}, leaves = {type(self).LEAF_COUNT}, score = {round(self.turn * score, 2)}", "    ", self.current_best_move.select_idx, "->", self.current_best_move.perform_idx)
                        type(self).MAX_Q_DEPTH = 0
                        type(self).LEAF_COUNT = 0
                        type(self).NODE_COUNT = 0
                except OutOfTime:
                    pass
                        

    return Board


    



if __name__ == "__main__":
    board = AbstractBoard([Pawn(-1, True)], 1)
    print(board)
    






















