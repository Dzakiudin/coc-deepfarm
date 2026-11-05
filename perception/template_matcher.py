import logging
import os

import cv2

logger = logging.getLogger(__name__)


class TemplateMatcher:
    """
    Template matching engine for Smart Click (Macro Recorder style).
    Matches cropped button reference images against game screen with multi-scale support.
    """

    def __init__(self, template_dir="templates"):
        self.template_dir = template_dir
        self._cache = {}
        self.reload_templates()

    def reload_templates(self):
        """Load and cache all templates from template_dir."""
        self._cache.clear()
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir, exist_ok=True)
            return

        for filename in os.listdir(self.template_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                key = os.path.splitext(filename)[0]
                path = os.path.join(self.template_dir, filename)
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    self._cache[key] = {
                        "color": img,
                        "gray": cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                        "shape": img.shape[:2],  # (h, w)
                    }
                    logger.debug("Loaded template '%s' from %s shape=%s", key, path, img.shape)
                else:
                    logger.warning("Failed to read template image: %s", path)

    def has_template(self, name: str) -> bool:
        return name in self._cache

    def available_templates(self) -> list:
        return sorted(list(self._cache.keys()))

    def find_template(
        self,
        screen_frame,
        template_name: str,
        threshold: float = 0.80,
        search_roi: tuple | None = None,
        scales: tuple = (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 1.05, 1.1, 1.15),
    ) -> dict | None:
        """
        Locate template image in screen_frame.
        
        Args:
            screen_frame: BGR numpy array of full game screen.
            template_name: Key name in templates directory (e.g. 'attack_btn_home').
            threshold: Minimum correlation score (0.0 - 1.0), default 0.80.
            search_roi: Optional (x, y, w, h) bounding box to restrict search area for speed.
            scales: Multi-scale factors to account for slight emulator resolution differences.
            
        Returns:
            dict with:
                "found": True,
                "center": (cx, cy),
                "bbox": (x1, y1, x2, y2),
                "confidence": float,
                "scale": float
            or None if no match reaches threshold.
        """
        if template_name not in self._cache:
            return None

        template_data = self._cache[template_name]
        template_gray = template_data["gray"]
        orig_th, orig_tw = template_data["shape"]

        if screen_frame is None or screen_frame.size == 0:
            return None

        screen_h, screen_w = screen_frame.shape[:2]

        # Crop ROI if specified
        roi_offset_x = 0
        roi_offset_y = 0
        if search_roi:
            rx, ry, rw, rh = [int(v) for v in search_roi]
            x1 = max(0, min(screen_w - 1, rx))
            y1 = max(0, min(screen_h - 1, ry))
            x2 = max(x1 + 1, min(screen_w, rx + rw))
            y2 = max(y1 + 1, min(screen_h, ry + rh))
            sub_frame = screen_frame[y1:y2, x1:x2]
            roi_offset_x = x1
            roi_offset_y = y1
        else:
            sub_frame = screen_frame

        sub_gray = cv2.cvtColor(sub_frame, cv2.COLOR_BGR2GRAY)
        sub_h, sub_w = sub_gray.shape[:2]

        best_score = -1.0
        best_match = None

        for scale in scales:
            if scale == 1.0:
                scaled_tpl = template_gray
                tw, th = orig_tw, orig_th
            else:
                tw = int(round(orig_tw * scale))
                th = int(round(orig_th * scale))
                if tw < 8 or th < 8 or tw > sub_w or th > sub_h:
                    continue
                scaled_tpl = cv2.resize(template_gray, (tw, th), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR)

            if tw > sub_w or th > sub_h:
                continue

            # Standard Normalized Cross-Correlation
            res = cv2.matchTemplate(sub_gray, scaled_tpl, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_score:
                best_score = max_val
                best_match = {
                    "top_left": (max_loc[0] + roi_offset_x, max_loc[1] + roi_offset_y),
                    "size": (tw, th),
                    "confidence": float(max_val),
                    "scale": scale,
                }

            # Fast exit if very confident
            if best_score >= 0.94:
                break

        if best_match and best_score >= threshold:
            tl_x, tl_y = best_match["top_left"]
            tw, th = best_match["size"]
            cx = int(round(tl_x + tw / 2.0))
            cy = int(round(tl_y + th / 2.0))
            return {
                "found": True,
                "center": (cx, cy),
                "bbox": (tl_x, tl_y, tl_x + tw, tl_y + th),
                "confidence": round(best_score, 3),
                "scale": best_match["scale"],
            }

        return None
