import functools
import math
import time
import random
import dataclasses
import multiprocessing
import queue
import os
import traceback
import sys
import ctypes



@dataclasses.dataclass(frozen = True)
class Castles():
    king_to : int #where the king moves to
    piece_from : int #where the rook starts
    piece_to : int #where the rook ends
    nocheck : tuple #in-between places where there cannot be check
    empty : tuple #in-between places which must be empty



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
    VALUE = 1000
    castles : tuple = dataclasses.field(default = tuple([]))


@dataclasses.dataclass(frozen = True)
class EnPassant():
    take_idx : int
    pawn : Pawn




@dataclasses.dataclass
class MoveSelectionPoint():
    INVIS = 0
    REGULAR = 1
    CAPTURE = 2
    SPECIAL = 3
    idx : int
    kind : int = dataclasses.field(default = REGULAR)

    
    

VALID_PIECES = {Pawn, Rook, Knight, Bishop, Queen, Prince, King}





class ShowBoardException(Exception):
    def __init__(self, msg, board):
        super().__init__(msg)
        self.board = board




class BoardSignature():
    def __init__(self, num_squares, flat_nbs, flat_opp, diag_nbs, diag_opp, pawn_moves, pawn_promotions, starting_layout):
        assert type(num_squares) == int and num_squares >= 0
        for piece in starting_layout:
            assert type(piece) in VALID_PIECES
            assert 0 <= piece.idx < num_squares

        for team in {-1, 1}:
            for idx, prom in pawn_promotions(team):
                for piece, sel_idx in prom:
                    assert piece.idx == idx

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
        self.PAWN_PROMS = {}
        for team in {-1, 1}:
            self.PAWN_PROMS[team] = {}
            for idx, prom in pawn_promotions(team):
                self.PAWN_PROMS[team][idx] = tuple((piece, sel_idx) for piece, sel_idx in prom)

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
        VERIFY_LEGAL = False
        
        def __init__(self, select_points, to_board):
            assert len(select_points) >= 1
            for select_point in select_points:
                assert isinstance(select_point, MoveSelectionPoint)
            self.select_points = select_points
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


    class ActualCastleMove(ActualMove):
        def __init__(self, *args, kingp, castle_info):
            super().__init__(*args)
            self.kingp = kingp
            self.castle_info = castle_info

        def is_legal(self, from_board, is_checked, seen_by):
            if type(self.kingp) == King:
                #musn't be in check, end up in check, or move through check
                for kingp_intermediate_idx in list(self.castle_info.nocheck) + [self.castle_info.king_to] + [self.kingp.idx]:
                    for info in seen_by[kingp_intermediate_idx]:
                        if info.piece.team != from_board.turn:
                            return False
                #otherwise we can castle
                if type(self).VERIFY_LEGAL:
                    assert super().is_legal(from_board, is_checked, seen_by)
                return True
            else: #e.g. prince
                return super().is_legal(from_board, is_checked, seen_by)


    class ActualEnPassantMove(ActualMove):
        @property
        def is_capture(self):
            return True
##        def __init__(self, *args):
##            super().__init__(*args)


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

            ans, case = fast_is_legal()
            if type(self).VERIFY_LEGAL:
                if ans != self.is_legal_slow(from_board, is_checked, seen_by):
                    raise Exception("ERROR in case: " + case + f"    Got {ans} expected {self.is_legal_slow(from_board, is_checked, seen_by)}", from_board)
            return ans

    def __init__(self, sig, num, pieces, turn, en_passant = None):
        if en_passant is None:
            en_passant = set()
        self.sig = sig        
        self.king_idx = {} #where is the king for each player
        self.en_passant = en_passant #pawns which can be taken via en_passant next move
        self.num = num
        #its important that the pieces are sorted so that two boards given the same pieces yield the same moves in the same order
        self.pieces = pieces
        self.ordered_pieces = sorted(pieces, key = lambda piece : piece.idx)
        self.idx_piece_lookup = {}
        for piece in pieces:
            assert not piece.idx in self.idx_piece_lookup
            self.idx_piece_lookup[piece.idx] = piece
            if type(piece) == King:
                assert not piece.team in self.king_idx
                self.king_idx[piece.team] = piece.idx
        self.turn = turn

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

        def new_move(from_piece, to_piece, sel_idxs = []):
            pieces = set(self.pieces)
            pieces.remove(from_piece)
            pieces.add(to_piece)
            selection_points = [MoveSelectionPoint(from_piece.idx, MoveSelectionPoint.INVIS), MoveSelectionPoint(to_piece.idx, MoveSelectionPoint.REGULAR)]
            for sel_idx in sel_idxs:
                selection_points.append(MoveSelectionPoint(sel_idx, MoveSelectionPoint.SPECIAL))
            pseudo_moves.append(Board.ActualNormalMove(selection_points, Board(self.sig, self.num + 1, pieces, -self.turn), moving_piece = piece, to_piece = to_piece, take_piece = None))

        def new_take(from_piece, to_piece, take_piece, sel_idxs = []):
            pieces = set(self.pieces)
            pieces.remove(from_piece)
            pieces.remove(take_piece)
            pieces.add(to_piece)
            selection_points = [MoveSelectionPoint(from_piece.idx, MoveSelectionPoint.INVIS), MoveSelectionPoint(to_piece.idx, MoveSelectionPoint.CAPTURE)]
            for sel_idx in sel_idxs:
                selection_points.append(MoveSelectionPoint(sel_idx, MoveSelectionPoint.SPECIAL))
            pseudo_moves.append(Board.ActualNormalMove(selection_points, Board(self.sig, self.num + 1, pieces, -self.turn), moving_piece = piece, to_piece = to_piece, take_piece = take_piece))

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

        for piece in self.ordered_pieces:
            idx = piece.idx
            if type(piece) == Pawn:
                if piece.team == self.turn:
                    for move1, move2s in self.sig.PAWN_MOVES[piece.team][idx]:
                        #pawn move 1 foward
                        if not move1 in self.idx_piece_lookup:
                            if move1 in self.sig.PAWN_PROMS[piece.team]:
                                for prom_piece, sel_idx in self.sig.PAWN_PROMS[piece.team][move1]:
                                    new_move(piece, prom_piece, sel_idxs = [sel_idx])
                            else:
                                new_move(piece, Pawn(move1, piece.team, True))
                                #pawn move 2 foward
                                for move2 in move2s:
                                    if not move2 in self.idx_piece_lookup:
                                        to_pawn = Pawn(move2, piece.team, True)
                                        pieces = set(self.pieces)
                                        pieces.remove(piece)
                                        pieces.add(to_pawn)
                                        move_board = Board(self.sig, self.num + 1, pieces, -self.turn, en_passant = {EnPassant(move1, to_pawn)})
                                        pseudo_moves.append(Board.ActualNormalMove([MoveSelectionPoint(piece.idx, MoveSelectionPoint.INVIS), MoveSelectionPoint(to_pawn.idx, MoveSelectionPoint.REGULAR)], move_board, moving_piece = piece, to_piece = to_pawn, take_piece = None))

                                    
                #pawn attack
                for move_idx in self.sig.PAWN_ATTACKS[piece.team][idx]:
                    new_info_teleport(piece, move_idx)
                    if move_idx in self.idx_piece_lookup:
                        blocking_piece = self.idx_piece_lookup[move_idx]
                        if piece.team == self.turn and blocking_piece.team != piece.team:
                            if move_idx in self.sig.PAWN_PROMS[piece.team]:
                                for prom_piece, sel_idx in self.sig.PAWN_PROMS[piece.team][move_idx]:
                                    new_take(piece, prom_piece, blocking_piece, sel_idxs = [sel_idx])
                            else:
                                new_take(piece, Pawn(move_idx, piece.team, True), blocking_piece)
                    else:
                        for en_passant_info in self.en_passant:
                            if move_idx == en_passant_info.take_idx and en_passant_info.pawn.team != piece.team:
                                to_piece = Pawn(move_idx, piece.team, True)
                                pieces = set(self.pieces)
                                pieces.remove(piece)
                                pieces.remove(en_passant_info.pawn)
                                pieces.add(to_piece)
                                move_board = Board(self.sig, self.num + 1, pieces, -self.turn)
                                pseudo_moves.append(Board.ActualEnPassantMove([MoveSelectionPoint(piece.idx, MoveSelectionPoint.INVIS), MoveSelectionPoint(move_idx, MoveSelectionPoint.SPECIAL)], move_board))
                            
            if type(piece) in {Rook, Queen}:
                do_slides(piece, idx, self.sig.FLAT_SLIDE[idx])
            if type(piece) in {Bishop, Queen}:
                do_slides(piece, idx, self.sig.DIAG_SLIDE[idx])
            if type(piece) in {Knight}:
                do_teleports(piece, idx, self.sig.KNIGHT_MOVES[idx])
            if type(piece) in {Prince, King}:
                do_teleports(piece, idx, self.sig.KING_MOVES[idx])
                kingp = piece
                for castle_info in kingp.castles:
                    if castle_info.piece_from in self.idx_piece_lookup:
                        piece = self.idx_piece_lookup[castle_info.piece_from]
                        if not piece.has_moved:
                            if not castle_info.piece_to in self.idx_piece_lookup and not castle_info.king_to in self.idx_piece_lookup:
                                if not any(i in self.idx_piece_lookup for i in castle_info.empty):
                                    moved_king = type(kingp)(castle_info.king_to, kingp.team, True)
                                    moved_piece = type(piece)(castle_info.piece_to, piece.team, True)
                                    pieces = set(self.pieces)
                                    pieces.remove(kingp)
                                    pieces.remove(piece)
                                    pieces.add(moved_king)
                                    pieces.add(moved_piece)
                                    pseudo_moves.append(Board.ActualCastleMove([MoveSelectionPoint(kingp.idx, MoveSelectionPoint.INVIS), MoveSelectionPoint(moved_king.idx, MoveSelectionPoint.SPECIAL)], Board(self.sig, self.num + 1, pieces, -self.turn), kingp = kingp, castle_info = castle_info))
                            

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
                if is_checked[self.turn]:
                    return -math.inf
                #no moves & not in check => draw
                return 0
            else:
                total = 0
                #piece value
                for piece in self.pieces:
                    total += piece.team * (type(piece).VALUE + (0.02 if piece.has_moved and type(piece) in {Bishop, Knight} else 0))
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

    def static_eval(self):
        return self.board_info("score")

    def is_checked(self):
        return self.board_info("checked")
    

class OutOfTime(Exception):
    pass





class AiPlayer():
##    @classmethod
##    def get_moves_sorted(cls, board, only_captures = False):
##        return sorted([move for move in self.get_moves() if move.is_capture or not only_captures], key = lambda move : (-move.to_board.best_known_score_depth, move.to_board.best_known_score))


    @classmethod
    def sort_moves(cls, moves, path, maxpath, move_scores):
##        #we want to try the most promising moves first so that alpha-beta pruning is most effective
##        for m_id, move in enumerate(moves):
##            if not move in move_scores:
##                score = move.to_board.static_eval()
##                info = (path, maxpath, score, -1, False)
##                move_scores[tuple(path + [m_id])] = score
####                move_score_queue.put((path + [m_id], maxpath + [len(moves) - 1], score, -1, False))
        return sorted(enumerate(moves), key = lambda pair : (move_scores[tuple(path + [pair[0]])] if tuple(path + [pair[0]]) in move_scores else pair[1].to_board.static_eval()))
        


    @classmethod
    def quiesce(cls, board, path, maxpath, move_score_queue, move_scores, alpha, beta, depth, max_qdepth, leaf_count):
        max_qdepth.value = max(depth, max_qdepth.value)
        stand_pat = board.static_eval()
        if stand_pat >= beta:
            leaf_count.value += 1
            move_score_queue.put((path, maxpath, beta, depth, True))
            return beta
        if alpha < stand_pat:
            alpha = stand_pat
        moves = board.get_moves()
        if len(moves) == 0:
            leaf_count.value += 1
        for m_id, move in cls.sort_moves(moves, path, maxpath, move_scores):
            if move.is_capture:
                if stand_pat + type(move.take_piece).VALUE < alpha:
                    #delta prune - we can already do better than taking this piece elsewhere
                    move_score_queue.put((path, maxpath, alpha, depth, True))
                    return alpha
                score = -cls.quiesce(move.to_board, path + [m_id], maxpath + [len(moves) - 1], move_score_queue, move_scores, -beta, -alpha, depth + 1, max_qdepth, leaf_count)
                if score >= beta:
                    move_score_queue.put((path, maxpath, beta, depth, True))
                    return beta
                if score > alpha:
                   alpha = score
        move_score_queue.put((path, maxpath, alpha, depth, True))
        return alpha

    @classmethod
    def alpha_beta(cls, board, path, maxpath, move_score_queue, move_scores, alpha, beta, depthleft, depth, max_qdepth, leaf_count):
        max_qdepth.value = max(depth, max_qdepth.value)
        if depthleft == 0:
            score = cls.quiesce(board, path, maxpath, move_score_queue, move_scores, alpha, beta, depth, max_qdepth, leaf_count)
            move_score_queue.put((path, maxpath, score, depthleft, False))
            return score
        moves = board.get_moves()
        for m_id, move in cls.sort_moves(moves, path, maxpath, move_scores):
            score = -cls.alpha_beta(move.to_board, path + [m_id], maxpath + [len(moves) - 1], move_score_queue, move_scores, -beta, -alpha, depthleft - 1, depth + 1, max_qdepth, leaf_count)
            if score >= beta:
                move_score_queue.put((path, maxpath, beta, depthleft, False))
                return beta
            if score > alpha:
                alpha = score
        move_score_queue.put((path, maxpath, alpha, depthleft, False))
        return alpha

    @classmethod
    def alpha_beta_process_root(cls, ans_queue, board, path, maxpath, move_score_queue, move_scores, alpha, beta, depthleft, depth, max_qdepth, leaf_count):
        ans_queue.put(cls.alpha_beta(board, path, maxpath, move_score_queue, move_scores, alpha, beta, depthleft, depth, max_qdepth, leaf_count))
        
    class Process(multiprocessing.Process):
        def __init__(self, error_queue, ans_queue, move_score_queue, max_qdepth, leaf_count, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error_queue = error_queue
            self.ans_queue = ans_queue
            self.move_score_queue = move_score_queue
            
            self.max_qdepth = max_qdepth
            self.leaf_count = leaf_count
            
        def run(self, *args, **kwargs):
            try:
                super().run(*args, **kwargs)
            except Exception as e:
                self.error_queue.put("".join(traceback.format_exception(None, e, e.__traceback__)) + f"\nError occured in subprocess pid={os.getpid()} with parent pid={os.getppid()}")
                raise e
        
    def __init__(self, board, num_proc = 12, move_score_info = None):
        #move_score_info is any information we already have on this move, for example from previous searches earlier in the game
        assert type(num_proc) == int and num_proc >= 1
        if move_score_info is None:
            move_score_info = {}
        assert len(board.get_moves()) > 0
                
        self.board = board
        self.num_proc = num_proc #how many processes
        self.error_queue = multiprocessing.Queue()
        self.processes = {} #move -> process

        self.move_score_info = move_score_info
        self.alpha = None
        self.qdepth = 0
        self.leaf_count = 0
        self.moves_to_try = []
        self.search_depth = None
        self.best_move_score = None
        self.best_move_idx = None
        self.best_move_idx_current_search = None
        self.alpha_beta_root(0)
        self.start_search(1)

    @property
    def best_move(self):
        return self.board.get_moves()[self.best_move_idx]
    def sub_move_score_info(self, m_id):
        return {path[1:] : tuple([info[0][1:]] + list(info[1:])) for path, info in self.move_score_info.items() if path[0] == m_id and len(path) >= 2}

    @property
    def move_scores(self):
        return {path : self.move_score_info[path][1] for path in self.move_score_info}

    def start_search(self, search_depth):
        assert type(search_depth) == int and search_depth >= 1
        self.qdepth = 0
        self.leaf_count = 0
        self.alpha = -math.inf
                
        self.moves_to_try = list(reversed(type(self).sort_moves(self.board.get_moves(), [], [], self.move_scores))) #reversed becasue we pop from the right of th list
        self.search_depth = search_depth
        print(len(self.moves_to_try), end = " ")

    def print_done_message(self):
        print(f"\nDone at depth {self.search_depth}+{self.qdepth}={self.search_depth + self.qdepth} boards={len(self.move_score_info)} leaves={self.leaf_count} score={round(self.board.turn * self.alpha, 2)}")
        
    def new_move_score_info(self, path, maxpath, score, depth, capture_only):
        assert (n := len(path)) == len(maxpath)
        for i in range(n):
            assert 0 <= path[i] <= maxpath[i]
        path = tuple(path)
        if path in self.move_score_info:
            old_maxpath, old_score, old_depth, old_capture_only = self.move_score_info[path]
            if (depth, not capture_only) > (old_depth, not old_capture_only):
                self.move_score_info[path] = (maxpath, score, depth, capture_only)
        else:
            self.move_score_info[path] = (maxpath, score, depth, capture_only)

    def alpha_beta_root(self, depthleft):
        self.search_depth = depthleft
        if depthleft == 0:
            self.best_move_idx_current_search = random.choice(range(len(self.board.get_moves())))
            self.alpha = 0
        else:
            assert depthleft >= 1
            max_qdepth = multiprocessing.Value(ctypes.c_int64, 0)
            leaf_count = multiprocessing.Value(ctypes.c_int64, 0)
            move_score_queue = queue.Queue()
            board = self.board
            self.alpha = -math.inf
            best_move_id = random.choice(range(len(board.get_moves())))
            moves = board.get_moves()
            for m_id, move in enumerate(moves):
                score = -type(self).alpha_beta(move.to_board, [m_id], [len(moves) - 1], move_score_queue, self.move_scores, -math.inf, -self.alpha, depthleft - 1, 1, max_qdepth, leaf_count)
                if score > self.alpha:
                    self.alpha = score
                    best_move_id = m_id
            self.qdepth = int(max_qdepth.value)
            self.leaf_count = int(leaf_count.value)
            while not move_score_queue.empty():
                self.new_move_score_info(*move_score_queue.get())
            self.best_move_idx_current_search = best_move_id
        self.best_move_idx = self.best_move_idx_current_search
        self.best_move_score = self.alpha
        self.print_done_message()


    def tick(self):    
        was_progress = False
        for m_id, p in list(self.processes.items()):
            if not p.is_alive():
                p.terminate()
                p.close()
                score = -p.ans_queue.get()
                if score > self.alpha:
                    self.alpha = score
                    self.best_move_idx_current_search = m_id
                    if self.alpha > self.best_move_score:
                        self.best_move_idx = self.best_move_idx_current_search
                        self.best_move_score = self.alpha
                del self.processes[m_id]
                self.qdepth = max(self.qdepth, int(p.max_qdepth.value))
                self.leaf_count += int(p.leaf_count.value)
                was_progress = True
            while not p.move_score_queue.empty():
                self.new_move_score_info(*p.move_score_queue.get())
        if was_progress:
            print(len(self.moves_to_try) + len(self.processes), end = " ")

        if len(self.moves_to_try) > 0 and len(self.processes) < self.num_proc:
            m_id, move = self.moves_to_try.pop()
            ans_queue = multiprocessing.Queue()
            max_qdepth = multiprocessing.Value(ctypes.c_int64, 0)
            leaf_count = multiprocessing.Value(ctypes.c_int64, 0)
            move_score_queue = multiprocessing.Queue()
            p = AiPlayer.Process(self.error_queue,
                                 ans_queue,
                                 move_score_queue,
                                 max_qdepth,
                                 leaf_count,
                                 target = self.alpha_beta_process_root,
                                 args = (ans_queue,
                                         move.to_board,
                                         [m_id],
                                         [len(self.board.get_moves()) - 1],
                                         move_score_queue,
                                         self.move_scores,
                                         -math.inf,
                                         -self.alpha,
                                         self.search_depth - 1,
                                         1,
                                         max_qdepth,
                                         leaf_count),
                                 daemon = True)
            self.processes[m_id] = p
            p.start()

        if len(self.moves_to_try) == 0 and len(self.processes) == 0:
            self.print_done_message()
            self.best_move_idx = self.best_move_idx_current_search
            self.best_move_score = self.alpha
            self.start_search(self.search_depth + 1)

        while not self.error_queue.empty():
            e = self.error_queue.get()
            raise Exception(e)


    def __del__(self):
        for move, p in self.processes.items():
            p.terminate()
        



    



















