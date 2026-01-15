import math
import pyautogui
import numpy as np
from collections import deque

class Solver:
    """
    An expert AI solver that uses predictive aiming and graph-based combo detection.
    """
    def __init__(self, frog_pos, screen_rect, shot_speed=1500):
        self.frog_pos = frog_pos
        self.screen_rect = screen_rect
        self.shot_speed = shot_speed # pixels per second
        self.candidate_cache = {}
        self.cache_max_size = 500
        pyautogui.PAUSE = 0.01
        pyautogui.FAILSAFE = True

    def _get_candidate_shots(self, frog_ball_color, path_balls):
        """
        Finds all locations where inserting the frog_ball would create a match.
        Uses a cache to avoid re-computing for the same game state.
        """
        path_key = "".join([b['color'][0] for b in path_balls])
        cache_key = f"{frog_ball_color}-{path_key}"
        if cache_key in self.candidate_cache:
            return self.candidate_cache[cache_key]

        candidates = []
        if len(path_balls) < 2:
            self.candidate_cache[cache_key] = []
            return []

        for i in range(len(path_balls) - 1):
            b1 = path_balls[i]
            b2 = path_balls[i+1]
            if b1['color'] == frog_ball_color and b2['color'] == frog_ball_color:
                if i > 0:
                    prev_ball = path_balls[i-1]
                    candidates.append({'x': (prev_ball['x'] + b1['x'])//2, 'y': (prev_ball['y'] + b1['y'])//2, 'match_size': 2, 'idx': i})
                if i < len(path_balls) - 2:
                    next_ball = path_balls[i+2]
                    candidates.append({'x': (b2['x'] + next_ball['x'])//2, 'y': (b2['y'] + next_ball['y'])//2, 'match_size': 2, 'idx': i})
        
        for i in range(len(path_balls) - 2):
            b1 = path_balls[i]
            b2 = path_balls[i+2]
            if b1['color'] == frog_ball_color and b2['color'] == frog_ball_color:
                gap_ball = path_balls[i+1]
                candidates.append({'x': gap_ball['x'], 'y': gap_ball['y'], 'match_size': 1, 'idx': i})

        if len(self.candidate_cache) > self.cache_max_size:
            # Evict the oldest item
            self.candidate_cache.pop(next(iter(self.candidate_cache)))
        self.candidate_cache[cache_key] = candidates
        return candidates

    def _score_shot(self, shot, path_balls):
        """
        Scores a shot based on survival, points, safety, and strategic bonuses.
        """
        num_balls = len(path_balls)
        if num_balls == 0 or shot['idx'] >= num_balls:
            return 0

        survival_score = math.exp(shot['idx'] / num_balls * 5)
        points_score = shot['match_size']

        gap_before = 1000
        if shot['idx'] > 0:
            p1 = np.array((path_balls[shot['idx']-1]['x'], path_balls[shot['idx']-1]['y']))
            p2 = np.array((shot['x'], shot['y']))
            gap_before = np.linalg.norm(p1 - p2)

        gap_after = 1000
        if shot['idx'] < num_balls:
            p1 = np.array((shot['x'], shot['y']))
            p2 = np.array((path_balls[shot['idx']]['x'], path_balls[shot['idx']]['y']))
            gap_after = np.linalg.norm(p1 - p2)
        
        safety_score = min(gap_before, gap_after)

        # --- Strategic Bonuses ---
        gap_bonus = 0
        chain_bonus = 0
        if shot['idx'] < len(path_balls):
            # Use the color of the ball being shot for the simulation
            shot_color = path_balls[shot['idx']]['color']
            future_path = self._simulate_shot(path_balls, shot['idx'], shot_color)
            if len(future_path) < num_balls - 3:
                chain_bonus = 500
            # Check for gap creation
            if shot['idx'] > 0 and shot['idx'] < len(path_balls) -1:
                if path_balls[shot['idx']-1]['color'] != path_balls[shot['idx']]['color'] and path_balls[shot['idx']]['color'] != path_balls[shot['idx']+1]['color']:
                    gap_bonus = 1000

        return survival_score * 1000 + points_score * 100 + safety_score + gap_bonus + chain_bonus

    def _is_obstructed(self, target_pos, path_balls):
        frog_pt = np.array(self.frog_pos)
        target_pt = np.array(target_pos)
        for ball in path_balls:
            ball_pt = np.array((ball['x'], ball['y']))
            ball_radius = ball['radius']
            if np.linalg.norm(target_pt - ball_pt) < ball_radius * 2:
                continue
            if np.linalg.norm(target_pt - frog_pt) == 0: return False
            d = np.linalg.norm(np.cross(target_pt - frog_pt, frog_pt - ball_pt)) / np.linalg.norm(target_pt - frog_pt)
            if d < ball_radius:
                return True
        return False

    def _predict_target_pos(self, target_ball):
        if 'vx' not in target_ball or 'vy' not in target_ball:
            return (target_ball['x'], target_ball['y'])
        dist = np.linalg.norm(np.array(self.frog_pos) - np.array((target_ball['x'], target_ball['y'])))
        if self.shot_speed == 0: return (target_ball['x'], target_ball['y'])
        time_to_impact = dist / self.shot_speed
        future_x = target_ball['x'] + target_ball['vx'] * time_to_impact
        future_y = target_ball['y'] + target_ball['vy'] * time_to_impact
        return (future_x, future_y)

    def _simulate_shot(self, path_balls, shot_idx, ball_color):
        if shot_idx >= len(path_balls):
            return path_balls
        sim_path = list(path_balls)
        sim_path.insert(shot_idx, {'color': ball_color, 'x': 0, 'y': 0, 'radius': 0})
        q = deque([shot_idx])
        visited = {shot_idx}
        match_group = {shot_idx}
        while q:
            curr_idx = q.popleft()
            for offset in [-1, 1]:
                neighbor_idx = curr_idx + offset
                if 0 <= neighbor_idx < len(sim_path) and neighbor_idx not in visited:
                    if sim_path[neighbor_idx]['color'] == ball_color:
                        visited.add(neighbor_idx)
                        match_group.add(neighbor_idx)
                        q.append(neighbor_idx)
        if len(match_group) >= 3:
            return [ball for i, ball in enumerate(sim_path) if i not in match_group]
        else:
            return sim_path

    def find_best_shot(self, current_ball, next_ball, path_balls):
        if not current_ball or not path_balls:
            return None
        current_candidates = self._get_candidate_shots(current_ball['color'], path_balls)
        if not current_candidates:
            return None
        best_overall_shot = None
        max_score = -1
        for shot1 in current_candidates:
            if shot1['idx'] >= len(path_balls):
                continue
            predicted_pos = self._predict_target_pos(path_balls[shot1['idx']])
            game_w = self.screen_rect[2]
            game_h = self.screen_rect[3]
            clamped_x = np.clip(predicted_pos[0], 1, game_w - 1)
            clamped_y = np.clip(predicted_pos[1], 1, game_h - 1)
            shot1['target_pos'] = (clamped_x, clamped_y)
            if self._is_obstructed(shot1['target_pos'], path_balls):
                continue
            future_path = self._simulate_shot(path_balls, shot1['idx'], current_ball['color'])
            score1 = self._score_shot(shot1, path_balls)
            score2 = 0
            if next_ball and future_path:
                next_candidates = self._get_candidate_shots(next_ball['color'], future_path)
                if next_candidates:
                    best_future_score = max(self._score_shot(shot2, future_path) for shot2 in next_candidates)
                    score2 = best_future_score
            total_score = score1 + score2 * 0.5
            if total_score > max_score:
                max_score = total_score
                best_overall_shot = shot1
        return best_overall_shot

    def execute_shot(self, target_pos):
        if not target_pos:
            return

        screen_w, screen_h = pyautogui.size()
        screen_x = self.screen_rect[0] + int(target_pos[0])
        screen_y = self.screen_rect[1] + int(target_pos[1])

        # Final, absolute safety check against screen corners
        if (screen_x <= 0 or screen_y <= 0 or 
            screen_x >= screen_w - 1 or screen_y >= screen_h - 1):
            print(f"SAFETY: Shot at ({screen_x}, {screen_y}) is too close to screen edge. Aborting.")
            return

        print(f"Shooting at: ({screen_x}, {screen_y})")
        pyautogui.moveTo(screen_x, screen_y, duration=0.1)
        pyautogui.click()
