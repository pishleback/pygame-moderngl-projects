import pgbase
import pygame
import moderngl
import math
from chessai import boardai
import sys
import os
import functools
import time






def sq_to_idx(sq):
    return sq[0] + 8 * sq[1]

def idx_to_sq(idx):
    return [idx % 8, idx // 8]

for i in range(64):
    assert i == sq_to_idx(idx_to_sq(i))
for x in range(8):
    for y in range(8):
        assert [x, y] == idx_to_sq(sq_to_idx([x, y]))

def flat_nbs(idx):
    x, y = idx_to_sq(idx)
    for dx, dy in [[1, 0], [-1, 0], [0, 1], [0, -1]]:
        ax, ay = x + dx, y + dy
        if 0 <= ax < 8 and 0 <= ay < 8:
            yield sq_to_idx([ax, ay])

def diag_nbs(idx):
    x, y = idx_to_sq(idx)
    for dx, dy in [[1, 1], [-1, -1], [-1, 1], [1, -1]]:
        ax, ay = x + dx, y + dy
        if 0 <= ax < 8 and 0 <= ay < 8:
            yield sq_to_idx([ax, ay])

def opp(i, j):
    xi, yi = idx_to_sq(i)
    xj, yj = idx_to_sq(j)
    xk, yk = 2 * xj - xi, 2 * yj - yi
    if 0 <= xk < 8 and 0 <= yk < 8:
        yield sq_to_idx([xk, yk])

def pawn_moves(team, idx):
    x, y = idx_to_sq(idx)
    if y in {0, 7}:
        #invalid position
        return []
    if team == 1:
        if y == 1:
            #starting move
            return [[sq_to_idx([x, y + 1]), [sq_to_idx([x, y + 2])]]]
        elif y <= 6:
            #generic move
            return [[sq_to_idx([x, y + 1]), []]]
        else:
            assert False
    elif team == -1:
        if y == 6:
            #starting move
            return [[sq_to_idx([x, y - 1]), [sq_to_idx([x, y - 2])]]]
        elif y >= 1:
            #generic move
            return [[sq_to_idx([x, y - 1]), []]]
        else:
            assert False
    else:
        assert False


STARTING_LAYOUT = set()
for x in range(8):
    STARTING_LAYOUT.add(boardai.Pawn(sq_to_idx([x, 1]), 1, False))
    STARTING_LAYOUT.add(boardai.Pawn(sq_to_idx([x, 6]), -1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx([0, 0]), 1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx([1, 0]), 1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx([2, 0]), 1, False))
STARTING_LAYOUT.add(boardai.King(sq_to_idx([3, 0]), 1, False, castles = (boardai.Castles(sq_to_idx([1, 0]), sq_to_idx([0, 0]), sq_to_idx([2, 0]), (sq_to_idx([2, 0]),), tuple()),
                                                                         boardai.Castles(sq_to_idx([5, 0]), sq_to_idx([7, 0]), sq_to_idx([4, 0]), (sq_to_idx([4, 0]),), (sq_to_idx([6, 0]),)))))
STARTING_LAYOUT.add(boardai.Queen(sq_to_idx([4, 0]), 1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx([5, 0]), 1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx([6, 0]), 1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx([7, 0]), 1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx([0, 7]), -1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx([1, 7]), -1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx([2, 7]), -1, False))
STARTING_LAYOUT.add(boardai.King(sq_to_idx([3, 7]), -1, False, castles = (boardai.Castles(sq_to_idx([1, 7]), sq_to_idx([0, 7]), sq_to_idx([2, 7]), (sq_to_idx([2, 7]),), tuple()),
                                                                          boardai.Castles(sq_to_idx([5, 7]), sq_to_idx([7, 7]), sq_to_idx([4, 7]), (sq_to_idx([4, 7]),), (sq_to_idx([6, 7]),)))))
STARTING_LAYOUT.add(boardai.Queen(sq_to_idx([4, 7]), -1, False))
STARTING_LAYOUT.add(boardai.Bishop(sq_to_idx([5, 7]), -1, False))
STARTING_LAYOUT.add(boardai.Knight(sq_to_idx([6, 7]), -1, False))
STARTING_LAYOUT.add(boardai.Rook(sq_to_idx([7, 7]), -1, False))

BOARD_SIGNATURE = boardai.BoardSignature(8 * 8, flat_nbs, opp, diag_nbs, opp, pawn_moves, STARTING_LAYOUT)



class BoardView(pgbase.canvas2d.Window2D):
    LIGHT_SQ_COLOUR = (255, 206, 158)
    DARK_SQ_COLOUR = (209, 139, 71)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.xsq = [math.floor(x * self.width / 8) for x in range(9)]
        self.ysq = [math.floor(y * self.height / 8) for y in range(9)]

        self.bg_surf = pygame.Surface([self.width, self.height], flags = pygame.SRCALPHA)
        for x in range(8):
            for y in range(8):
                pygame.draw.rect(self.bg_surf, [self.LIGHT_SQ_COLOUR, self.DARK_SQ_COLOUR][(x + y) % 2], self.get_sq_rect(x, y))

        self.board = None
        self.moves = []
        self.ai_player = None

        self.selected_piece = None
        self.last_move = None
        self.last_interact_time = time.time()
        self.draw_board = None


        self.set_board(BOARD_SIGNATURE.starting_board())


    def make_move(self, move, m_id = None):
        self.set_board(move.to_board, m_id = m_id)
        self.last_move = move

    def set_board(self, board, m_id = None):
        if not m_id is None:
            sub_move_score_info = self.ai_player.sub_move_score_info(m_id)
        else:
            sub_move_score_info = None
            
        print("new board", board.num, round(board.static_eval(), 2))
        self.last_move_time = time.time()
        self.board = board
        self.moves = tuple(board.get_moves())
        self.selected_piece = None
        if len(self.moves) == 0:
           self.ai_player = None
        else:
            self.ai_player = boardai.AiPlayer(self.board, move_score_info = sub_move_score_info)

##        print("========")
##        for x in range(8):
##            for y in range(8):
##                idx = sq_to_idx([x, y])
##                print([x, y], [idx_to_sq(info.at_idx) for info in self.board.seen_by(idx)])
##        print(self.board.is_checked(1), self.board.is_checked(-1))
##        print(self.board.static_eval(1))

##        for idx, obs in self.board.seenby_lookup.items():
##            print(idx, obs)
        

    def get_sq_rect(self, x, y):
        return (self.xsq[x], self.ysq[y], self.xsq[x + 1] - self.xsq[x], self.ysq[y + 1] - self.ysq[y])
        
    def set_rect(self, rect):
        super().set_rect(rect)

    def tick(self, tps):
        pass
##        if not self.ai_player is None:
##            self.ai_player.tick()
##            if self.ai_player.search_depth > 3:
##                self.make_move(self.ai_player.best_move, m_id = self.ai_player.best_move_idx)

##        try:
##            if time.time() - self.last_interact_time > 1:
##                self.board.best_move_search(0.5)
##        except boardai.ShowBoardException as e:
##            print(e)
##            self.draw_board = e.board
##        if time.time() - self.last_interact_time > 2 and time.time() - self.last_move_time > 3 and not self.board.current_best_move is None:
##            self.make_move(self.board.current_best_move)

            
        
    def event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self.last_interact_time = time.time()
                
                self.last_move = None
                
                sqx, sqy = [math.floor(8 * (event.pos[i] - self.rect[i]) / self.rect[i + 2]) for i in [0, 1]]
                if 0 <= sqx < 8 and 0 <= sqy < 8:
                    idx = sq_to_idx([sqx, sqy])
                    
                    if not self.selected_piece is None:
                        for m_id, move in enumerate(self.moves):
                            if move.select_idx == self.selected_piece.idx and move.perform_idx == idx:
                                self.make_move(move, m_id = m_id)
                                return
                            
                    if idx in self.board.idx_piece_lookup:
                        piece = self.board.idx_piece_lookup[idx]
                        if piece.team == self.board.turn:
                            self.selected_piece = piece
                        else:
                            self.selected_piece = None
                    else:
                        self.selected_piece = None
                    
                    
                
    @functools.cache
    def piece_surf(self, piece):
        assert type(piece) in boardai.VALID_PIECES
        if piece.team == 1:
            if type(piece) == boardai.Pawn:
                img = "white pawn.png"
            elif type(piece) == boardai.Rook:
                img = "white rook.png"
            elif type(piece) == boardai.Knight:
                img = "white knight.png"
            elif type(piece) == boardai.Bishop:
                img = "white bishop.png"
            elif type(piece) == boardai.Queen:
                img = "white queen.png"
            elif type(piece) == boardai.King:
                img = "white king.png"
            else:
                raise NotImplementedError(f"no icon for {piece}")
        elif piece.team == -1:
            if type(piece) == boardai.Pawn:
                img = "black pawn.png"
            elif type(piece) == boardai.Rook:
                img = "black rook.png"
            elif type(piece) == boardai.Knight:
                img = "black knight.png"
            elif type(piece) == boardai.Bishop:
                img = "black bishop.png"
            elif type(piece) == boardai.Queen:
                img = "black queen.png"
            elif type(piece) == boardai.King:
                img = "black king.png"
            else:
                raise NotImplementedError(f"no icon for {piece}")
        else:
            assert False
        path = os.path.join("chessai", "icons", img)
        surf = pygame.image.load(path)
        w, h = self.width / 8, self.height / 8
        surf = pygame.transform.smoothscale(surf, (w, h))
        return surf
        
            
    def draw(self):        
        self.ctx.clear(0, 0, 0, 0, 0)
        self.ctx.enable_only(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        draw_surf = pygame.Surface([self.width, self.height], flags = pygame.SRCALPHA)
        draw_surf.blit(self.bg_surf, (0, 0))
        for piece in (self.board.pieces if self.draw_board is None else self.draw_board.pieces):
            sqx, sqy = idx_to_sq(piece.idx)
            x, y, _, _ = self.get_sq_rect(sqx, sqy)
            draw_surf.blit(self.piece_surf(piece), (x, y))

        
        if not self.ai_player is None:   
            best_move = self.ai_player.best_move
            for idx in [best_move.select_idx, best_move.perform_idx]:
                sqx, sqy = idx_to_sq(idx)
                rect = self.get_sq_rect(sqx, sqy)
                pygame.draw.rect(draw_surf, (255, 128, 0), rect, 8)

        if not self.last_move is None:
             for idx in [self.last_move.select_idx, self.last_move.perform_idx]:
                sqx, sqy = idx_to_sq(idx)
                rect = self.get_sq_rect(sqx, sqy)
                pygame.draw.rect(draw_surf, (255, 255, 0), rect, 8)

        if not self.selected_piece is None:
            sqx, sqy = idx_to_sq(self.selected_piece.idx)
            rect = self.get_sq_rect(sqx, sqy)
            pygame.draw.rect(draw_surf, (0, 128, 255), rect, 8)

            for move in self.moves:
                if move.select_idx == self.selected_piece.idx:
                    sqx, sqy = idx_to_sq(move.perform_idx)
                    rect = self.get_sq_rect(sqx, sqy)
                    colour = (0, 255, 0)
                    if move.is_capture:
                        colour = (255, 0, 0)
                    pygame.draw.rect(draw_surf, colour, rect, 8)
        
        tex = pgbase.tools.np_uint8_to_tex(self.ctx, pgbase.tools.pysurf_to_np_uint8(draw_surf).transpose([1, 0, 2]))
        pgbase.tools.render_tex(tex)
        tex.release() #to avoid memory issues lol

    def end(self, e):
        #terminate ai subprocesses
        if not self.ai_player:
            del self.ai_player
        self.ai_player = None
        





def run():
    pgbase.core.Window.setup(size = [1000, 1000])
    pgbase.core.run(BoardView())
    pygame.quit()
    sys.exit()




















    
