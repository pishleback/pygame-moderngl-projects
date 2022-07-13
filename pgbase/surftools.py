import pygame
import os
import math

FONT_NOTO_REGULAR = os.path.join("pgbase", "fonts", "NotoSerif-Regular.ttf")
FONT_NOTO_SYMBOLS = os.path.join("pgbase", "fonts", "NotoSansSymbols-Regular.ttf")
FONT_NOTO_SYMBOLS2 = os.path.join("pgbase", "fonts", "NotoSansSymbols2-Regular.ttf")
FONT_NOTO_MATHS = os.path.join("pgbase", "fonts", "NotoSansMath-Regular.ttf")

FONT_DEFAULT = FONT_NOTO_REGULAR

def make_raw_text_surface(text, height, colour, font = None):
    if font is None:
        font = FONT_DEFAULT
    def mask_to_output(text_mask):
        text_width, text_height = text_mask.get_width(), text_mask.get_height()
        reduc_mult = height / text_height
        text_surf = pygame.Surface(text_mask.get_size()).convert_alpha()
        text_surf.fill(colour)
        text_surf.blit(text_mask, (0, 0), special_flags = pygame.BLEND_RGBA_MULT)
        text_surf_fitted = pygame.transform.smoothscale(text_surf, [math.floor(text_width * reduc_mult), math.floor(text_height * reduc_mult)])
        return text_surf_fitted
    assert len(colour) == 4
    font = pygame.font.Font(font, 2 * height)
    text_mask = font.render(text, True, [255, 255, 255]).convert_alpha()
    return mask_to_output(text_mask)


def write(surf, text, rect, colour, align = (0.5, 0.5), font = None):
    #rect should be given as fractions of surf NOT IN PIXELS
    #write text onto surf inside rect using new lines if needed
    
    pix_rect = [math.floor(surf.get_width() * rect[0]),
                math.floor(surf.get_height() * rect[1]),
                math.floor(surf.get_width() * rect[2]),
                math.floor(surf.get_height() * rect[3])]
    
    text_surf = make_raw_text_surface(text, 2 * pix_rect[3], colour, font)

    width_scaledown = text_surf.get_width() / pix_rect[2]
    height_scaledown = text_surf.get_height() / pix_rect[3]

    if width_scaledown > height_scaledown:
        small_text_surf = pygame.transform.smoothscale(text_surf, [pix_rect[2], math.floor(text_surf.get_height() / width_scaledown)])
        surf.blit(small_text_surf, [pix_rect[0], pix_rect[1] + math.floor(align[1] * (pix_rect[3] - small_text_surf.get_height()))])
    else:
        small_text_surf = pygame.transform.smoothscale(text_surf, [math.floor(text_surf.get_width() / height_scaledown), pix_rect[3]])
        surf.blit(small_text_surf, [pix_rect[0] + math.floor(align[0] * (pix_rect[2] - small_text_surf.get_width())), pix_rect[1]])
        


def join_vert(surfs, *, pad = 0, f = 0.5):
    assert 0 <= f <= 1
    for surf in surfs:
        assert type(surf) == pygame.Surface
    width = max([surf.get_width() for surf in surfs])
    height = sum([surf.get_height() for surf in surfs]) + (len(surfs) - 1) * pad
    surface = pygame.Surface([width, height]).convert_alpha()
    surface.fill([0, 0, 0, 0])
    y_offset = 0
    for i, surf in enumerate(surfs):
        if i != 0:
            y_offset += pad
        surface.blit(surf, [int(f * (width - surf.get_width())), y_offset], special_flags = pygame.BLEND_RGBA_ADD)
        y_offset += surf.get_height()
    return surface


def join_horz(surfs, *, pad = 0, f = 0.5):
    assert 0 <= f <= 1
    for surf in surfs:
        assert type(surf) == pygame.Surface
    width = sum([surf.get_width() for surf in surfs]) + (len(surfs) - 1) * pad
    height = max([surf.get_height() for surf in surfs])
    surface = pygame.Surface([width, height]).convert_alpha()
    surface.fill([0, 0, 0, 0])
    x_offset = 0
    for i, surf in enumerate(surfs):
        if i != 0:
            x_offset += pad
        surface.blit(surf, [x_offset, int(f * (height - surf.get_height()))], special_flags = pygame.BLEND_RGBA_ADD)
        x_offset += surf.get_width()
    return surface
