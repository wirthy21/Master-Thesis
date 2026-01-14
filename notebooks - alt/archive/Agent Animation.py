# Image: https://www.researchgate.net/figure/Comparison-of-real-fish-motion-top-Blake-1983-and-simulated-fish-using-the_fig1_312591550

import pygame, sys

pygame.init()
screen = pygame.display.set_mode((800, 600))

# 1) Lade alle 8 Frames in eine Liste
pred_frames = []
for i in range(1, 9):
    surf_pred = pygame.image.load(f'OneDrive\Dokumente\Privat\Bildung\M. Sc. Social and Economic Data Science/4. Semester\Master Thesis\Code\data\images/fish_movement/pred_{i}.png').convert_alpha()
    pred_frames.append(surf_pred)

# 1) Lade alle 8 Frames in eine Liste
prey_frames = []
for i in range(1, 9):
    surf_prey = pygame.image.load(f'OneDrive\Dokumente\Privat\Bildung\M. Sc. Social and Economic Data Science/4. Semester\Master Thesis\Code\data\images/fish_movement/prey_{i}.png').convert_alpha()
    scaled = pygame.transform.scale(surf_prey, (surf_prey.get_width() // 4, surf_prey.get_height() // 4))
    prey_frames.append(scaled)

frame_idx = 0
clock     = pygame.time.Clock()
FPS       = 8

while True:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    # 2) Frame‐Index aktualisieren
    frame_idx = (frame_idx + 1) % 8 #frames

    # 3) Rendern
    screen.fill((30, 30, 30))
    img_pred = pred_frames[frame_idx]
    screen.blit(img_pred, (100, 100))

    img_prey = prey_frames[frame_idx]
    screen.blit(img_prey, (400, 100))
    pygame.display.flip()
    clock.tick(FPS)