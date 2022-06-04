import functools



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
    def __init__(self, *args, castles):
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


    class Move():
        def __init__(self, select_idx, perform_idx, from_board, to_board, is_capture):
            self.select_idx = select_idx
            self.perform_idx = perform_idx
            self.from_board = from_board
            self.to_board = to_board
            self.is_capture = is_capture


    class Board():
        #a slide is a tuple of moves a sliding piece can make assuming an empty board
        #in particuar, a slide does not contain the starting position (unless it is included as the end of a loop e.g. a loop round the wormhole ending at the starting position
        FLAT_SLIDE = {idx : tuple(gen_slide(idx, flat_nbs, flat_opp)) for idx in range(num_squares)} #tuple of slides
        DIAG_SLIDE = {idx : tuple(gen_slide(idx, diag_nbs, diag_opp)) for idx in range(num_squares)} #tuple of slides
        KNIGHT_MOVES = {idx : tuple(set(knight_moves(idx))) for idx in range(num_squares)} #tuple of knight moves
        KING_MOVES = {idx : tuple(set(king_moves(idx))) for idx in range(num_squares)} #tuple of king moves
        PAWN_MOVES = {team : {idx : tuple(tuple([m1, tuple(m2s)]) for m1, m2s in pawn_moves(team, idx)) for idx in range(num_squares)} for team in {-1, 1}}
        PAWN_ATTACKS = {team : {idx : tuple(set(pawn_attacks(team, idx))) for idx in range(num_squares)} for team in {-1, 1}}

        @classmethod
        def starting_board(cls):
            return cls(0, starting_layout, 1)

        def __init__(self, num, pieces, turn):
            self.num = num
            self.pieces = pieces
            self.idx_piece_lookup = {}
            for piece, idx in pieces.items():
                assert not idx in self.idx_piece_lookup
                self.idx_piece_lookup[idx] = piece
            self.turn = turn

        @functools.cache
        @lambda f : lambda self : tuple(f(self))
        def get_moves(self):
            def move_board(move_pieces):
                return Board(self.num + 1, move_pieces, -self.turn)
            
            for piece, idx in self.pieces.items():
                if piece.team == self.turn:
                    if type(piece) == Pawn:
                        for move1, move2s in self.PAWN_MOVES[self.turn][idx]:
                            #pawn move 1 foward
                            if not move1 in self.idx_piece_lookup:
                                move_pieces = {piece : idx for piece, idx in self.pieces.items()}
                                del move_pieces[piece]
                                move_pieces[Pawn(piece.team, True)] = move1
                                yield Move(idx, move1, self, move_board(move_pieces), False)
                                #pawn move 2 foward
                                for move2 in move2s:
                                    if not move2 in self.idx_piece_lookup:
                                        move_pieces = {piece : idx for piece, idx in self.pieces.items()}
                                        del move_pieces[piece]
                                        move_pieces[Pawn(piece.team, True)] = move2
                                        yield Move(idx, move2, self, move_board(move_pieces), False)
                        #pawn attack
                        for take in self.PAWN_ATTACKS[self.turn][idx]:
                            if take in self.idx_piece_lookup:
                                take_piece = self.idx_piece_lookup[take]
                                if take_piece.team == -piece.team:
                                    move_pieces = {piece : idx for piece, idx in self.pieces.items()}
                                    del move_pieces[piece]
                                    del move_pieces[take_piece]
                                    move_pieces[Pawn(piece.team, True)] = take
                                    yield Move(idx, take, self, move_board(move_pieces), True)
                    #flat slides
                    if type(piece) in {Rook, Queen}:
                        for slide in self.FLAT_SLIDE[idx]:
                            for move_idx in slide:
                                if move_idx in self.idx_piece_lookup:
                                    blocking_piece = self.idx_piece_lookup[move_idx]
                                    if blocking_piece.team != self.turn:
                                        #we can take a piece
                                        move_pieces = {piece : idx for piece, idx in self.pieces.items()}
                                        del move_pieces[piece]
                                        del move_pieces[blocking_piece]
                                        move_pieces[type(piece)(piece.team, True)] = move_idx
                                        yield Move(idx, move_idx, self, move_board(move_pieces), True)
                                    break
                                else:
                                    #generic move
                                    move_pieces = {piece : idx for piece, idx in self.pieces.items()}
                                    del move_pieces[piece]
                                    move_pieces[type(piece)(piece.team, True)] = move_idx
                                    yield Move(idx, move_idx, self, move_board(move_pieces), False)
                    #diagonal slides   
                    if type(piece) in {Bishop, Queen}:
                        for slide in self.DIAG_SLIDE[idx]:
                            for move_idx in slide:
                                if move_idx in self.idx_piece_lookup:
                                    blocking_piece = self.idx_piece_lookup[move_idx]
                                    if blocking_piece.team != self.turn:
                                        #we can take a piece
                                        move_pieces = {piece : idx for piece, idx in self.pieces.items()}
                                        del move_pieces[piece]
                                        del move_pieces[blocking_piece]
                                        move_pieces[type(piece)(piece.team, True)] = move_idx
                                        yield Move(idx, move_idx, self, move_board(move_pieces), True)
                                    break
                                else:
                                    #generic move
                                    move_pieces = {piece : idx for piece, idx in self.pieces.items()}
                                    del move_pieces[piece]
                                    move_pieces[type(piece)(piece.team, True)] = move_idx
                                    yield Move(idx, move_idx, self, move_board(move_pieces), False)
                                    
                                    
            return
            yield
                

    return Board


    



if __name__ == "__main__":
    board = AbstractBoard([Pawn(-1, True)], 1)
    print(board)
    






















