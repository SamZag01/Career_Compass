import sys
import time
from enum import Enum

from main import GameStatus
from src.components.optionbar import Bar #from src.components.hand import Hand Chk changes

class BarSide(Enum):
    RIGHT = 0
    LEFT = 1

from src.components.player import Player  #build
from src.components.displayboard import Displayboard
from src.global_state import GlobalState
from src.services.music_service import MusicService
from src.services.visualization_service import VisualizationService
from src.utils.tools import update_background_using_scroll, update_press_key, is_close_app_event

GlobalState.load_main_screen()
VisualizationService.load_main_game_displays()

displayboard = Displayboard()  #check changes
# Sprite Setup
P1 = Player()
H1 = Bar(BarSide.RIGHT)
H2 = Bar(BarSide.LEFT)
# Sprite Groups
bars = pygame.sprite.Group()
bars.add(H1)
bars.add(H2)
all_sprites = pygame.sprite.Group()
all_sprites.add(P1)
all_sprites.add(H1)
all_sprites.add(H2)

def main_menu_phase():
    displayboard.reset_current_trait()

    events = pygame.event.get()

    for event in events:
        if is_close_app_event(event):
            GlobalState.GAME_STATE = GameStatus.GAME_END
            return

        if event.type == pygame.KEYDOWN:
            GlobalState.GAME_STATE = GameStatus.GAMEPLAY

    GlobalState.SCROLL = update_background_using_scroll(GlobalState.SCROLL)
    VisualizationService.draw_background_with_scroll(GlobalState.SCREEN, GlobalState.SCROLL)
    GlobalState.PRESS_Y = update_press_key(GlobalState.PRESS_Y)
    VisualizationService.draw_main_menu(GlobalState.SCREEN, displayboard.get_max_score(), GlobalState.PRESS_Y)


def gameplay_phase():
    events = pygame.event.get()

    for event in events:
        if is_close_app_event(event):
            game_over()
            return

    P1.update()
    H1.move(displayboard, P1.player_position)
    H2.move(displayboard, P1.player_position)

    GlobalState.SCROLL = update_background_using_scroll(GlobalState.SCROLL)
    VisualizationService.draw_background_with_scroll(GlobalState.SCREEN, GlobalState.SCROLL)

    P1.draw(GlobalState.SCREEN)
    H1.draw(GlobalState.SCREEN)
    H2.draw(GlobalState.SCREEN)
    displayboard.draw(GlobalState.SCREEN)

    if pygame.sprite.spritecollide(P1, bars, False, pygame.sprite.collide_mask):
        displayboard.update_max_score()
        MusicService.play_slap_sound()
        time.sleep(0.5)
        #game_over()
def exit_game_phase():
    pygame.quit()
    sys.exit()
def game_over():
    P1.reset()
    H1.reset()
    H2.reset()
    GlobalState.GAME_STATE = GameStatus.MAIN_MENU
    time.sleep(0.5)