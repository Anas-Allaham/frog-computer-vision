from __future__ import annotations
import win32gui, win32con, win32process
import psutil

class WindowNotFound(Exception):
    pass

def _enum_windows():
    out = []
    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if title:
            out.append((hwnd, title))
    win32gui.EnumWindows(cb, None)
    return out

def find_window_hwnd(title_contains: str | None = None,
                     process_name: str | None = None) -> int:
    wins = _enum_windows()

    if title_contains:
        t = title_contains.lower()
        wins = [(h, ttl) for (h, ttl) in wins if t in ttl.lower()]

    if process_name:
        p = process_name.lower()
        filtered = []
        for hwnd, ttl in wins:
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                if psutil.Process(pid).name().lower() == p:
                    filtered.append((hwnd, ttl))
            except Exception:
                pass
        wins = filtered

    if not wins:
        raise WindowNotFound(f"No window found for title_contains={title_contains}, process_name={process_name}")

    fg = win32gui.GetForegroundWindow()
    for hwnd, _ in wins:
        if hwnd == fg:
            return hwnd
    return wins[0][0]

def get_client_bbox(hwnd: int) -> dict:
    l, t, r, b = win32gui.GetClientRect(hwnd)
    tl = win32gui.ClientToScreen(hwnd, (l, t))
    br = win32gui.ClientToScreen(hwnd, (r, b))
    x1, y1 = tl
    x2, y2 = br
    return {"left": x1, "top": y1, "width": x2 - x1, "height": y2 - y1}

import time
import win32gui, win32con, win32process
import ctypes

def focus_window(hwnd: int, click_fallback: bool = False):
    # restore if minimized
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

    # try normal foreground
    try:
        win32gui.SetForegroundWindow(hwnd)
        time.sleep(0.05)
        if win32gui.GetForegroundWindow() == hwnd:
            return True
    except Exception:
        pass

    # stronger method: attach input to foreground thread
    try:
        fg = win32gui.GetForegroundWindow()
        fg_tid = win32process.GetWindowThreadProcessId(fg)[0]
        this_tid = win32process.GetWindowThreadProcessId(hwnd)[0]

        user32 = ctypes.windll.user32
        user32.AttachThreadInput(fg_tid, this_tid, True)
        win32gui.BringWindowToTop(hwnd)
        win32gui.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(fg_tid, this_tid, False)

        time.sleep(0.05)
        if win32gui.GetForegroundWindow() == hwnd:
            return True
    except Exception:
        pass

    # optional fallback: click inside window (most reliable)
    if click_fallback:
        try:
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            cx, cy = (l + r) // 2, (t + b) // 2
            user32 = ctypes.windll.user32
            user32.SetCursorPos(cx, cy)
            user32.mouse_event(2, 0, 0, 0, 0)  # left down
            user32.mouse_event(4, 0, 0, 0, 0)  # left up
            time.sleep(0.05)
            return win32gui.GetForegroundWindow() == hwnd
        except Exception:
            pass

    return False

def is_alive(hwnd: int) -> bool:
    return win32gui.IsWindow(hwnd) and win32gui.IsWindowVisible(hwnd)
