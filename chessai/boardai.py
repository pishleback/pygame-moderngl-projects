import functools
import math
import time
import random
import dataclasses
import multiprocessing
import os
import traceback
import sys
import ctypes

@dataclasses.dataclass(frozen = True)
class Piece():
    idx : int #where is it
    team : int
    has_moved : bool


##class Piece():
##    VALUE = None
##    def __init__(self, team, has_moved):
##        assert team in {-1, 1}
##        assert type(has_moved) is bool
##        self.team = team
##        self.has_moved = has_moved
    
@dataclasses.dataclass(frozen = True)
class Pawn(Piece):
    VALUE = 1

@dataclasses.dataclass(frozen = True)
class Rook(Piece):
    VALUE = 5

@dataclasses.dataclass(frozen = True)
class Knight(Piece):
    VALUE = 3

@dataclasses.dataclass(frozen = True)
class Bishop(Piece):
    VALUE = 3

@dataclasses.dataclass(frozen = True)
class Queen(Piece):
    VALUE = 9

@dataclasses.dataclass(frozen = True)
class Prince(Piece):
    VALUE = 4

@dataclasses.dataclass(frozen = True)
class King(Piece):
    castles : tuple = dataclasses.field(default = tuple([]))
    VALUE = 1000

VALID_SQUARES = {Pawn, Rook, Knight, Bishop, Queen, Prince, King}





class ShowBoardException(Exception):
    def __init__(self, msg, board):
        super().__init__(msg)
        self.board = board




class BoardSignature():
    def __init__(self, num_squares, flat_nbs, flat_opp, diag_nbs, diag_opp, pawn_moves, starting_layout):
        assert type(num_squares) == int and num_squares >= 0
        for piece in starting_layout:
            assert type(piece) in VALID_SQUARES
            assert 0 <= piece.idx < num_squares

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

        self.NUM_SQUARES = num_squares
        self.FLAT_SLIDE = {idx : tuple(gen_slide(idx, flat_nbs, flat_opp)) for idx in range(num_squares)} #tuple of slides
        self.DIAG_SLIDE = {idx : tuple(gen_slide(idx, diag_nbs, diag_opp)) for idx in range(num_squares)} #tuple of slides
        self.KNIGHT_MOVES = {idx : tuple(set(knight_moves(idx))) for idx in range(num_squares)} #tuple of knight moves
        self.KING_MOVES = {idx : tuple(set(king_moves(idx))) for idx in range(num_squares)} #tuple of king moves
        self.PAWN_MOVES = {team : {idx : tuple(tuple([m1, tuple(m2s)]) for m1, m2s in pawn_moves(team, idx)) for idx in range(num_squares)} for team in {-1, 1}}
        self.PAWN_ATTACKS = {team : {idx : tuple(set(pawn_attacks(team, idx))) for idx in range(num_squares)} for team in {-1, 1}}
        self.STARTING_LAYOUT = starting_layout

    def starting_board(self):
        return Board(self, 0, self.STARTING_LAYOUT, 1)





class Board():
    class MovementInfo():
        def __init__(self, piece, sees_idx):
            self.piece = piece
            self.sees_idx = sees_idx

    class MovementInfoTeleport(MovementInfo):
        def __init__(self, *args):
            super().__init__(*args)

    class MovementInfoSlide(MovementInfo):
        def __init__(self, *args, slide_idx, slide):
            super().__init__(*args)
            self.slide_idx = slide_idx
            self.slide = slide
            
    #actual moves which can be made
    class ActualMove():
        def __init__(self, select_idx, perform_idx, to_board):
            self.select_idx = select_idx
            self.perform_idx = perform_idx
            self.to_board = to_board

        @property
        def is_capture(self):
            return False

        def is_legal_slow(self, from_board, is_checked, seen_by):
            #make this more efficient:
            #if in check, can only do 1) move king out of danger 2) block / take attacking piece
            #if not in check, can only not do 1) move a pinned piece 2) move king into check 3) do en passant and remove pinned enemy pawn
            #checking the above would be much faster than generating all moves on the next board to see if we are in check afer each move
            return not self.to_board.is_checked()[from_board.turn]

        def is_legal(self, from_board, is_checked, seen_by):
            return self.is_legal_slow(from_board, is_checked, seen_by)


    #either a normal move or a take
    #does not include en passant or castling - for example, we assume that the taken piece is replaced by a piece for optimization algorithms
    class ActualNormalMove(ActualMove):
        def __init__(self, *args, moving_piece, to_piece, take_piece = None):
            super().__init__(*args)
            self.moving_piece = moving_piece
            self.to_piece = to_piece
            self.take_piece = take_piece
            
        @property
        def is_capture(self):
            return not self.take_piece is None

        def is_legal(self, from_board, is_checked, seen_by):
            def fast_is_legal():
                #we want to determine if the move is legal without running the expensive self.to_board._board_info
                #we can however use from_board._board_info - this is usually enough
                if type(self.moving_piece) == King:
                    if is_checked[from_board.turn]:
                        #are we moving into check
                        for info in seen_by[self.to_piece.idx]:
                            if info.piece.team != from_board.turn:
                                return False, "moving king into check from a checked position"
                        #are we moving into a check which is obscured by ourself (the continuation of a checking sliding piece)
                        for info in seen_by[from_board.king_idx[from_board.turn]]:
                            if info.piece.team != from_board.turn:
                                if isinstance(info, Board.MovementInfoSlide):
                                    if info.slide_idx < len(info.slide) - 1:
                                        danger_idx = info.slide[info.slide_idx + 1]
                                        if self.to_piece.idx == danger_idx:
                                            return False, "moving king into self-obscured check from a checked position"
                        #we are safe otherwise
                        return True, "not moving king into any check from a checked position"
                    else:
                        #are we moving into check
                        for info in seen_by[self.to_piece.idx]:
                            if info.piece.team != from_board.turn:
                                return False, "moving king into check from an unchecked position"
                        #we are safe otherwise
                        return True, "not moving king into check from an unchecked positino"
                else:
                    #are we pinned
                    taken_pieces = (set([]) if self.take_piece is None else {self.take_piece})
                    for info in seen_by[self.moving_piece.idx]: #look for all things attacking us with a sliding move
                        if isinstance(info, Board.MovementInfoSlide):
                            if info.piece.team == self.to_board.turn: #if it is an opponent piece
                                if not info.piece in taken_pieces: #and we are not taking it
                                    for danger_idx in info.slide: #we must make sure we are not pinned by it
                                        if danger_idx in self.to_board.idx_piece_lookup:
                                            hit = self.to_board.idx_piece_lookup[danger_idx]
                                            if type(hit) == King and hit.team == from_board.turn:
                                                return False, "pinned piece"
                                            break
                    #from now on, we know we are not pinned
                    if not is_checked[from_board.turn]:
                        return True, "not pinned and not in check"
                    else:
                        #we are in check but not pinned
                        #find all things putting us in check
                        checkers = []
                        for info in seen_by[from_board.king_idx[from_board.turn]]:
                            if info.piece.team != from_board.turn:
                                checkers.append(info)
                        assert len(checkers) >= 1                            
                        #NOTE that it is possible to block two sliding checkers at once in wormhole chess
                        #it is however not possible to block multiple checkers if one is a teleporter
                        if any(isinstance(info, Board.MovementInfoTeleport) for info in checkers):
                            #if any are teleporters, the move is only legal if there is exactly one and we are taking it
                            #for example it is not possible to both take a checking knight and block a checking rook at the same time
                            if self.take_piece is None:
                                return False, "checked by teleporting piece and not taking it"
                            if len(checkers) >= 2:
                                return False, "checked by 2 or more teleporting pieces"
                            checker = checkers[0].piece
                            #we are in check by a single teleporting piece and we are not pinned => legal iff we take it
                            return self.take_piece == checker, "taking the unique teleporting checking piece"
                            
                        assert all(isinstance(info, Board.MovementInfoSlide) for info in checkers)
                        #if there exists only sliding checkers are we are not pinned
                        #the move is legal iff (we block all sliding moves before they reach the king OR we take a unique checking sliding piece)

    ##                        for info in checkers:
    ##                            print(self.moving_piece, self.to_piece, self.take_piece, info.piece, info.slide, info.slide_idx)

                        if len(checkers) == 1 and not self.take_piece is None:
                                if self.take_piece == checkers[0].piece:
                                    return True, "a single sliding piece which gets taken by an unpinned piece"
                        #if theres 2 or more sliding checkers OR we arent taking one, then we must block them all
                        if all(self.to_piece.idx in info.slide[:info.slide_idx] for info in checkers):
                            return True, "liding checkers and we block them all with an unpinned piece"
                        else:
                            return False, "sliding checkers and not blocking them all"

                        #return self.is_legal_slow()

            verify = False
            ans, case = fast_is_legal()
            if verify:
                if ans != self.is_legal_slow(from_board, is_checked, seen_by):
                    raise Exception("ERROR in case: " + case + f"    Got {ans} expected {self.is_legal_slow(from_board, is_checked, seen_by)}", from_board)
            return ans
    
    
    SEARCH_START_TIME = time.time()
    SEARCH_TIME_ALLOWED = 0.0
    MAX_Q_DEPTH = 0
    LEAF_COUNT = 0
    NODE_COUNT = 0


    def __init__(self, sig, num, pieces, turn):
        self.sig = sig
        
        self.king_idx = {} #where is the king for each player
        self.num = num
        self.pieces = pieces
        self.idx_piece_lookup = {}
        for piece in pieces:
            assert not piece.idx in self.idx_piece_lookup
            self.idx_piece_lookup[piece.idx] = piece
            if type(piece) == King:
                assert not piece.team in self.king_idx
                self.king_idx[piece.team] = piece.idx
        self.turn = turn
        

    def cache_clear(self):
        if "_board_info" in self.__dict__:
            for move in self._board_info[0]:
                move.to_board.cache_clear()
                del move.to_board
            del self._board_info


    def board_info(self, query):
        assert type(query) == str
        assert query in {"moves", "checked", "score"}
        
        pseudo_moves = []
        seen_by = {idx : set() for idx in range(self.sig.NUM_SQUARES)}
        movement_score = {1 : 0, -1 : 0} #currently unused, delete it?

        def new_movement_score(info):
            if info.sees_idx in self.idx_piece_lookup:
                sees_piece = self.idx_piece_lookup[info.sees_idx]
                movement_score[info.piece.team] += 0.05 * max(5 - type(info.piece).VALUE - type(sees_piece).VALUE, 0) #good to defend/attack low value stuff with low value stuff
            else:
                m_score = 0.05
                
            movement_score[info.piece.team] += 0.02 #movement score
            
        def new_info_teleport(from_piece, to_idx):
            info = Board.MovementInfoTeleport(from_piece, to_idx)
            seen_by[to_idx].add(info)
            new_movement_score(info)

        def new_slide_info(from_piece, to_idx, slide_idx, slide):
            info = Board.MovementInfoSlide(from_piece, to_idx, slide_idx = slide_idx, slide = slide)
            seen_by[to_idx].add(info)
            new_movement_score(info)

        def new_move(from_piece, to_piece):
            pieces = set(self.pieces)
            pieces.remove(from_piece)
            pieces.add(to_piece)
            pseudo_moves.append(Board.ActualNormalMove(from_piece.idx, to_piece.idx, Board(self.sig, self.num + 1, pieces, -self.turn), moving_piece = piece, to_piece = to_piece, take_piece = None))

        def new_take(from_piece, to_piece, take_piece):
            pieces = set(self.pieces)
            pieces.remove(from_piece)
            pieces.remove(take_piece)
            pieces.add(to_piece)
            pseudo_moves.append(Board.ActualNormalMove(from_piece.idx, to_piece.idx, Board(self.sig, self.num + 1, pieces, -self.turn), moving_piece = piece, to_piece = to_piece, take_piece = take_piece))

        def do_slides(piece, idx, slides):
            for slide in slides:
                for slide_idx, move_idx in enumerate(slide):
                    new_slide_info(piece, move_idx, slide_idx, slide)
                    if move_idx in self.idx_piece_lookup:
                        blocking_piece = self.idx_piece_lookup[move_idx]
                        if piece.team == self.turn and blocking_piece.team != piece.team:
                            new_take(piece, type(piece)(move_idx, piece.team, True), blocking_piece)
                        break
                    else:
                        if piece.team == self.turn:
                            new_move(piece, type(piece)(move_idx, piece.team, True))

        def do_teleports(piece, idx, move_idxs):
            for move_idx in move_idxs:
                new_info_teleport(piece, move_idx)
                if move_idx in self.idx_piece_lookup:
                    blocking_piece = self.idx_piece_lookup[move_idx]
                    if piece.team == self.turn and blocking_piece.team != piece.team:
                        new_take(piece, type(piece)(move_idx, piece.team, True), blocking_piece)
                else:
                    if piece.team == self.turn:
                        new_move(piece, type(piece)(move_idx, piece.team, True))

        for piece in self.pieces:
            idx = piece.idx
            if type(piece) == Pawn:
                if piece.team == self.turn:
                    for move1, move2s in self.sig.PAWN_MOVES[piece.team][idx]:
                        #pawn move 1 foward
                        if not move1 in self.idx_piece_lookup:
                            new_move(piece, Pawn(move1, piece.team, True))
                            #pawn move 2 foward
                            for move2 in move2s:
                                if not move2 in self.idx_piece_lookup:
                                    new_move(piece, Pawn(move2, piece.team, True))
                #pawn attack
                for move_idx in self.sig.PAWN_ATTACKS[piece.team][idx]:
                    new_info_teleport(piece, move_idx)
                    if move_idx in self.idx_piece_lookup:
                        blocking_piece = self.idx_piece_lookup[move_idx]
                        if piece.team == self.turn and blocking_piece.team != piece.team:
                            new_take(piece, Pawn(move_idx, piece.team, True), blocking_piece)

            if type(piece) in {Rook, Queen}:
                do_slides(piece, idx, self.sig.FLAT_SLIDE[idx])
            if type(piece) in {Bishop, Queen}:
                do_slides(piece, idx, self.sig.DIAG_SLIDE[idx])
            if type(piece) in {Knight}:
                do_teleports(piece, idx, self.sig.KNIGHT_MOVES[idx])
            if type(piece) in {Prince, King}:
                do_teleports(piece, idx, self.sig.KING_MOVES[idx])

        def compute_is_checked(team):
            for info in seen_by[self.king_idx[team]]:
                if info.piece.team != team:
                    return True
            return False
        is_checked = {team : compute_is_checked(team) for team in {-1, 1}}
        if query == "checked":
            return is_checked

        legal_moves = tuple(move for move in pseudo_moves if move.is_legal(self, is_checked, seen_by))
        if query == "moves":
            #it is important that these come out in the same order each time for the ai to remember old move scores cross-process
            return legal_moves

        def compute_score():
            if len(legal_moves) == 0:
                for info in seen_by[self.king_idx[self.turn]]:
                    if info.piece.team == -self.turn:
                        #no moves & in check => checkmate. We loose
                        return -math.inf * self.turn
                #no moves & not in check => draw
                return 0
            else:
                total = 0
                #piece value
                for piece in self.pieces:
                    total += piece.team * type(piece).VALUE
                #avalable moves
                for t, sc in movement_score.items():
                    total += t * sc
                return self.turn * total
        score = compute_score()
        if query == "score":
            return score

        raise Exception(f"Unknown query \"{query}\"")
        
    def get_moves(self):
        return self.board_info("moves")

    def static_eval(self, team):
        return self.board_info("score")

    def is_checked(self):
        return self.board_info("checked")
    

class OutOfTime(Exception):
    pass



def long_func(x):
    for _ in range(10 ** 7):
        x += 1



class AiPlayer():
##    @classmethod
##    def get_moves_sorted(cls, board, only_captures = False):
##        return sorted([move for move in self.get_moves() if move.is_capture or not only_captures], key = lambda move : (-move.to_board.best_known_score_depth, move.to_board.best_known_score))


    @classmethod
    def quiesce(cls, board, path, move_score_queue, move_score_info, alpha, beta, depth, max_qdepth, node_count, leaf_count):
        max_qdepth.value = max(depth, max_qdepth.value)
        node_count.value += 1
        stand_pat = board.static_eval(board.turn)
        if stand_pat >= beta:
            leaf_count.value += 1
            move_score_queue.put((path, beta, depth, True))
            return beta
        if alpha < stand_pat:
            alpha = stand_pat            
##        for m_id, move in enumerate(board.get_moves()):
        for m_id, move in sorted(enumerate(board.get_moves()), key = lambda pair : move_score_info.get(tuple(path + [pair[0]]), 0)):
            if move.is_capture:
                if stand_pat + type(move.take_piece).VALUE < alpha:
                    leaf_count.value += 1
                    #delta prune - we can already do better than taking this piece elsewhere
                    return alpha
                score = -cls.quiesce(move.to_board, path + [m_id], move_score_queue, move_score_info, -beta, -alpha, depth + 1, max_qdepth, node_count, leaf_count)
                if score >= beta:
                    return beta
                if score > alpha:
                   alpha = score
        move_score_queue.put((path, alpha, depth, True))
        return alpha

    @classmethod
    def alpha_beta(cls, board, path, move_score_queue, move_score_info, alpha, beta, depthleft, depth, max_qdepth, node_count, leaf_count):
        max_qdepth.value = max(depth, max_qdepth.value)
        node_count.value += 1
        if depthleft == 0:
            score = cls.quiesce(board, path, move_score_queue, move_score_info, alpha, beta, depth, max_qdepth, node_count, leaf_count)
            move_score_queue.put((path, score, depth, True))
            return score
##        for m_id, move in enumerate(board.get_moves()):
        for m_id, move in sorted(enumerate(board.get_moves()), key = lambda pair : move_score_info.get(tuple(path + [pair[0]]), 0)):
            score = -cls.alpha_beta(move.to_board, path + [m_id], move_score_queue, move_score_info, -beta, -alpha, depthleft - 1, depth + 1, max_qdepth, node_count, leaf_count)
            if score >= beta:
                return beta
            if score > alpha:
                alpha = score
        move_score_queue.put((path, alpha, depth, False))
        return alpha

    @classmethod
    def alpha_beta_process_root(cls, ans_queue, board, path, move_score_queue, move_score_info, alpha, beta, depthleft, depth, max_qdepth, node_count, leaf_count):
        ans_queue.put(cls.alpha_beta(board, path, move_score_queue, move_score_info, alpha, beta, depthleft, depth, max_qdepth, node_count, leaf_count))

    class Process(multiprocessing.Process):
        def __init__(self, error_queue, ans_queue, move_score_queue, max_qdepth, node_count, leaf_count, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error_queue = error_queue
            self.ans_queue = ans_queue
            self.move_score_queue = move_score_queue
            
            self.max_qdepth = max_qdepth
            self.node_count = node_count
            self.leaf_count = leaf_count
            
        def run(self, *args, **kwargs):
            try:
                super().run(*args, **kwargs)
            except Exception as e:
                self.error_queue.put("".join(traceback.format_exception(None, e, e.__traceback__)) + f"\nError occured in subprocess pid={os.getpid()} with parent pid={os.getppid()}")
                raise e
        
    def __init__(self, board, num = 4):
        self.board = board
        self.num = num #how many processes
        self.error_queue = multiprocessing.Queue()
        self.processes = {} #move -> process

        self.move_score_info = {}
        self.alpha = None
        self.qdepth = 0
        self.node_count = 0
        self.leaf_count = 0
        self.search_depth = 1
        self.moves_to_try = []
        self.best_move = self.alpha_beta_root()
        self.best_move_current_search = self.best_move
        self.start_search(2)

    def start_search(self, search_depth):
        assert type(search_depth) == int and search_depth >= 1
        self.qdepth = 0
        self.node_count = 0
        self.leaf_count = 0
        self.alpha = -math.inf
        self.moves_to_try = list(enumerate(self.board.get_moves()))
        self.search_depth = search_depth

    def print_done_message(self):
        print(f"\nDone at depth {self.search_depth}+{self.qdepth}={self.search_depth + self.qdepth} boards={self.node_count} leaves={self.leaf_count}")
        
    def new_move_score_info(self, path, score, depth, capture_only):
        #print(path, score, depth, capture_only)
        self.move_score_info[tuple(path)] = score

    def alpha_beta_root(self):
        depthleft = self.search_depth
        assert depthleft >= 1
        max_qdepth = multiprocessing.Value(ctypes.c_int64, 0)
        node_count = multiprocessing.Value(ctypes.c_int64, 0)
        leaf_count = multiprocessing.Value(ctypes.c_int64, 0)
        move_score_queue = multiprocessing.Queue()
        board = self.board
        alpha = -math.inf
        best_move = random.choice(board.get_moves())
        for m_id, move in enumerate(board.get_moves()):
            score = -type(self).alpha_beta(move.to_board, [m_id], move_score_queue, self.move_score_info, -math.inf, -alpha, depthleft - 1, 1, max_qdepth, node_count, leaf_count)
            if score > alpha:
                alpha = score
                best_move = move
        self.qdepth = int(max_qdepth.value)
        self.node_count = int(node_count.value)
        self.leaf_count = int(leaf_count.value)
        while not move_score_queue.empty():
            self.new_move_score_info(*move_score_queue.get())
        self.print_done_message()
        return best_move

    def tick(self):
        was_progress = False
        for move, p in list(self.processes.items()):
            if not p.is_alive():
                p.terminate()
                p.close()
                score = -p.ans_queue.get()
                if score > self.alpha:
                    self.alpha = score
                    self.best_move_current_search = move
                del self.processes[move]
                self.qdepth = max(self.qdepth, int(p.max_qdepth.value))
                self.node_count += int(p.node_count.value)
                self.leaf_count += int(p.leaf_count.value)
                was_progress = True
            while not p.move_score_queue.empty():
                self.new_move_score_info(*p.move_score_queue.get())
        if was_progress:
            print(len(self.moves_to_try) + len(self.processes), end = " ")

        while len(self.moves_to_try) > 0 and len(self.processes) < 12:
            m_id, move = self.moves_to_try.pop()
            ans_queue = multiprocessing.Queue()
            max_qdepth = multiprocessing.Value(ctypes.c_int64, 0)
            node_count = multiprocessing.Value(ctypes.c_int64, 0)
            leaf_count = multiprocessing.Value(ctypes.c_int64, 0)
            move_score_queue = multiprocessing.Queue()
            p = AiPlayer.Process(self.error_queue,
                                 ans_queue,
                                 move_score_queue,
                                 max_qdepth,
                                 node_count,
                                 leaf_count,
                                 target = self.alpha_beta_process_root,
                                 args = (ans_queue,
                                         move.to_board,
                                         [m_id],
                                         move_score_queue,
                                         self.move_score_info,
                                         -math.inf,
                                         -self.alpha,
                                         self.search_depth - 1,
                                         1,
                                         max_qdepth,
                                         node_count,
                                         leaf_count))
            self.processes[move] = p
            p.start()

        if len(self.moves_to_try) == 0 and len(self.processes) == 0:
            self.print_done_message()
            self.best_move = self.best_move_current_search
            self.start_search(self.search_depth + 1)

        while not self.error_queue.empty():
            e = self.error_queue.get()
            raise Exception(e)

    def __del__(self):
        for move, p in self.processes.items():
            p.terminate()
        



    



















