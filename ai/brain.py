import json
import logging
import os
import time
import urllib.error
import urllib.request

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_FALLBACK_MODEL = "gemini-2.5-flash-lite"
_MODEL_ALIASES = {
    "gemini-3.5-flash": "gemini-2.5-flash-lite",
    "gemini-3.8-flash": "gemini-2.5-flash",
}


def normalize_model_name(name: str | None, default: str = DEFAULT_MODEL) -> str:
    raw = str(name or "").strip()
    if not raw:
        return default
    return _MODEL_ALIASES.get(raw.lower(), raw)


def evaluate_loot_targets(
    loot: dict,
    targets: dict,
    mode: str = "and",
    dead_base: bool = False,
) -> bool:
    """
    Evaluate whether base loot satisfies target thresholds under:
    - 'and': all resources meet their respective targets
    - 'or': at least one resource with a positive target meets its target
    - 'smart': standard AND, OR combined Gold+Elixir >= 1.15x target sum,
      OR single jackpot >= 1.6x target, plus a 15% threshold discount on Dead Bases.
    """
    mode_norm = str(mode or "and").strip().lower()
    gold = int(loot.get("gold", 0) or 0)
    elixir = int(loot.get("elixir", 0) or 0)
    delixir = int(loot.get("delixir", 0) or 0)

    g_req = int(targets.get("gold", 0) or 0)
    e_req = int(targets.get("elixir", 0) or 0)
    d_req = int(targets.get("delixir", 0) or 0)

    if mode_norm == "or":
        checks = []
        if g_req > 0:
            checks.append(gold >= g_req)
        if e_req > 0:
            checks.append(elixir >= e_req)
        if d_req > 0:
            checks.append(delixir >= d_req)
        return any(checks) if checks else True

    if mode_norm == "smart":
        discount = 0.85 if dead_base else 1.0
        sg = int(g_req * discount)
        se = int(e_req * discount)
        sd = int(d_req * discount)
        if gold >= sg and elixir >= se and delixir >= sd:
            return True
        if (sg + se) > 0 and (gold + elixir) >= int((sg + se) * 1.15):
            if sd == 0 or delixir >= int(sd * 0.7):
                return True
        if sg > 0 and gold >= int(sg * 1.6) and (se == 0 or elixir >= int(se * 0.5)):
            return True
        if se > 0 and elixir >= int(se * 1.6) and (sg == 0 or gold >= int(sg * 0.5)):
            return True
        return False

    return gold >= g_req and elixir >= e_req and delixir >= d_req


def infer_target_side(base_geom: dict) -> str:
    """
    Choose 'left' or 'right' if >= 72% of detected stores cluster on one side, else 'all'.
    """
    if not isinstance(base_geom, dict):
        return "all"
    storages = base_geom.get("storages") or []
    if len(storages) < 4:
        return "all"
    left_count = sum(1 for s in storages if float(s.get("x", 0.5)) <= 0.48)
    right_count = sum(1 for s in storages if float(s.get("x", 0.5)) >= 0.52)
    total = len(storages)
    if left_count / total >= 0.72:
        return "left"
    if right_count / total >= 0.72:
        return "right"
    return "all"


def should_consult_ai(state: dict) -> bool:
    """
    Pre-filter to avoid burning Gemini API RPM quota on obviously empty bases while scouting.
    Consults AI when the base meets targets, or reaches at least 65% of target loot, or is a dead base.
    """
    if not isinstance(state, dict):
        return False
    loot = state.get("loot") or {}
    targets = state.get("targets") or {}
    base_geom = state.get("base") or {}
    attack_cfg = state.get("attack") or {}
    dead_base = bool(base_geom.get("dead_base", False))
    target_mode = str(attack_cfg.get("target_mode") or "and")

    if evaluate_loot_targets(loot, targets, mode=target_mode, dead_base=dead_base):
        return True
    if dead_base:
        return True

    relaxed_targets = {
        k: int(float(targets.get(k, 0) or 0) * 0.65)
        for k in ("gold", "elixir", "delixir")
    }
    return evaluate_loot_targets(loot, relaxed_targets, mode="or", dead_base=False)
GENERATE_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
INTERACTIONS_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions"
RETRY_STATUS = (429, 503)

PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "skip_base": {
            "type": "boolean",
            "description": (
                "True if this base is below the loot targets and should be skipped."
            ),
        },
        "deploy_order": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Troop slot names from state.troop_slots, in the order to deploy them."
            ),
        },
        "surrender_after_s": {
            "type": "number",
            "description": "Seconds to keep attacking before surrendering.",
        },
        "target_side": {
            "type": "string",
            "enum": ["all", "left", "right"],
            "description": (
                "Deploy toward the side where the stores cluster. Vertical "
                "splits are not offered because a top/bottom split leaves the "
                "other half of the base unreachable."
            ),
        },
    },
    "required": ["skip_base", "deploy_order", "surrender_after_s", "target_side"],
}

SYSTEM_INSTRUCTION = (
    "You are a Clash of Clans farm-bot strategist. The bot scans a base, reads "
    "the available loot, then deploys troops to drain resources. Only four "
    "decisions are yours: skip the base, troop deployment order, seconds to keep "
    "attacking before surrendering, and which side of the base to deploy on. "
    "Every coordinate is chosen elsewhere, so never return pixel positions.\n"
    "State fields: `loot` is what the base offers now, `targets` is the minimum "
    "worth attacking, `base.storages` lists detected stores with normalised "
    "coordinates (x 0 = left, y 0 = top), `history` shows the last raids and what "
    "they gained, and `session.totals` accumulates this run.\n"
    "Prefer greed for loot over stars: skip bases that cannot beat `targets`, "
    "deploy cheap fast troops first when the loot is shallow, and spread the "
    "deploy side toward where the stores cluster. `target_side` is all, left, or "
    "right; vertical splits are excluded because a top/bottom split leaves the "
    "other half of the base unreachable. `deploy_order` may only contain "
    "names present in state.troop_slots. Answer only with the requested JSON."
)


class Brain:

    name = "base"

    def plan(self, state: dict) -> dict:
        raise NotImplementedError


class RuleBrain(Brain):

    name = "rule"

    def plan(self, state: dict) -> dict:
        try:
            loot = state["loot"]
            targets = state["targets"]
            attack_cfg = state.get("attack") if isinstance(state.get("attack"), dict) else {}
            base_geom = state.get("base") if isinstance(state.get("base"), dict) else {}
            target_mode = attack_cfg.get("target_mode", "and")
            dead_base = bool(base_geom.get("dead_base", False))
            meets = evaluate_loot_targets(
                loot, targets, mode=target_mode, dead_base=dead_base
            )
            skip_base = not meets
            return {
                "skip_base": skip_base,
                "deploy_order": list(state["troop_slots"]),
                "surrender_after_s": float(state["attack"]["time_limit"]),
                "target_side": infer_target_side(base_geom),
            }
        except Exception as e:
            logger.debug(f"RuleBrain could not plan from state: {e}")
            return {}


class GeminiBrain(Brain):

    name = "gemini"

    def __init__(
        self,
        model=DEFAULT_MODEL,
        api_key=None,
        api_key_env="GEMINI_API_KEY",
        timeout_s=8.0,
        endpoint=None,
        thinking_level="low",
        fallback_model=None,
    ):
        self.model = normalize_model_name(model, default=DEFAULT_MODEL)
        self.fallback_model = (
            normalize_model_name(fallback_model, default="") or None
            if fallback_model
            else None
        )
        self.api_key = api_key or os.environ.get(api_key_env)
        self.api_key_env = api_key_env
        self.timeout_s = float(timeout_s)
        self._custom_endpoint = bool(endpoint)
        self.endpoint = endpoint or GENERATE_ENDPOINT.format(model=self.model)
        self.thinking_level = thinking_level
        self.thinking_budget = (
            0
            if str(thinking_level).strip().lower() in ("low", "none", "minimal", "0")
            else None
        )
        self.enabled = bool(self.api_key)
        self._blocked_until = 0.0
        self.last_error = ""
        if not self.enabled:
            logger.warning(
                "GeminiBrain disabled: no API key (pass api_key or set "
                f"{api_key_env})"
            )

    def describe(self) -> str:
        return f"gemini/{self.model} enabled={self.enabled}"

    def _build_body(self, state: dict) -> dict:
        payload = json.dumps(state, ensure_ascii=False)
        if "interactions" in self.endpoint:
            return {
                "model": self.model,
                "input": payload,
                "system_instruction": SYSTEM_INSTRUCTION,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": PLAN_SCHEMA,
                },
                "generation_config": {"thinking_level": self.thinking_level},
            }

        generation_config = {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseSchema": PLAN_SCHEMA,
        }
        if self.thinking_budget is not None:
            generation_config["thinkingConfig"] = {
                "thinkingBudget": self.thinking_budget
            }
        return {
            "systemInstruction": {"parts": [{"text": SYSTEM_INSTRUCTION}]},
            "contents": [{"role": "user", "parts": [{"text": payload}]}],
            "generationConfig": generation_config,
        }

    def plan(self, state: dict) -> dict:
        if not self.enabled:
            return {}
        if time.time() < self._blocked_until:
            remaining = self._blocked_until - time.time()
            logger.debug(f"gemini cooling down for {remaining:.0f}s")
            return {}

        api_key = str(self.api_key or "")
        base_body = self._build_body(state)
        models = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            models.append(self.fallback_model)

        raw = None
        for model in models:
            endpoint = self.endpoint
            if model != self.model and not self._custom_endpoint:
                endpoint = GENERATE_ENDPOINT.format(model=model)
            body = json.loads(json.dumps(base_body))
            stripped_thinking = False
            if "lite" in model and "thinkingConfig" in (
                body.get("generationConfig") or {}
            ):
                body["generationConfig"].pop("thinkingConfig", None)
                stripped_thinking = True
            for attempt in range(3):
                request = urllib.request.Request(
                    endpoint,
                    data=json.dumps(body).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "x-goog-api-key": api_key,
                    },
                    method="POST",
                )
                try:
                    with urllib.request.urlopen(
                        request, timeout=self.timeout_s
                    ) as response:
                        raw = response.read().decode("utf-8", errors="replace")
                    if model != self.model:
                        logger.info(f"gemini answered with fallback model {model}")
                        self.model = model
                        self.endpoint = endpoint
                    self.last_error = ""
                    break
                except urllib.error.HTTPError as e:
                    try:
                        body_snippet = e.read().decode("utf-8", errors="replace")
                    except Exception:
                        body_snippet = ""
                    logger.warning(f"gemini http {e.code}: {body_snippet[:300]}")
                    self.last_error = f"http {e.code} ({model})"
                    generation = body.get("generationConfig") or {}
                    if e.code == 400 and "thinkingConfig" in generation and (
                        not stripped_thinking
                    ):
                        generation.pop("thinkingConfig", None)
                        stripped_thinking = True
                        logger.debug(
                            f"{model} rejected thinkingConfig; retrying without it"
                        )
                        continue
                    if e.code == 429:
                        if model != models[-1]:
                            break
                        wait_s = 60.0
                        retry_after = ""
                        try:
                            retry_after = (e.headers or {}).get("Retry-After") or ""
                        except Exception:
                            retry_after = ""
                        if str(retry_after).strip().isdigit():
                            wait_s = float(retry_after)
                        self._blocked_until = time.time() + wait_s
                        logger.info(
                            f"gemini rate limited; pausing requests for {wait_s:.0f}s"
                        )
                        return {}
                    if e.code in RETRY_STATUS and attempt == 0:
                        time.sleep(1.0)
                        continue
                    break
                except Exception as e:
                    logger.warning(f"gemini request failed: {e}")
                    self.last_error = str(e)
                    break
            if raw:
                break

        if not raw:
            return {}

        try:
            response_json = json.loads(raw)
        except Exception as e:
            logger.warning(f"gemini returned non-JSON: {e}")
            return {}

        text = self._extract_text(response_json)
        if not text:
            return {}

        try:
            parsed = json.loads(text)
        except Exception as e:
            logger.warning(f"gemini plan was not JSON: {e}")
            return {}

        return self._sanitize(parsed, state)

    def _extract_text(self, resp):
        if not isinstance(resp, dict):
            return ""

        if "skip_base" in resp:
            try:
                return json.dumps(resp)
            except Exception:
                pass

        output_text = resp.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text

        steps = resp.get("steps")
        if isinstance(steps, list):
            for step in reversed(steps):
                if not isinstance(step, dict):
                    continue
                if step.get("type") not in ("model_output", "model_output_step"):
                    continue
                content = step.get("content")
                if not isinstance(content, list):
                    continue
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                joined = "".join(parts)
                if joined.strip():
                    return joined

        try:
            parts = resp["candidates"][0]["content"]["parts"]
            if isinstance(parts, list):
                joined = "".join(
                    part["text"]
                    for part in parts
                    if isinstance(part, dict) and isinstance(part.get("text"), str)
                )
                if joined.strip():
                    return joined
        except (KeyError, IndexError, TypeError):
            pass

        outputs = resp.get("outputs")
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, str) and first.strip():
                return first
            if isinstance(first, dict):
                text = first.get("text") or first.get("output_text")
                if isinstance(text, str) and text.strip():
                    return text

        responses = resp.get("responses")
        if isinstance(responses, list) and responses:
            first = responses[0]
            if isinstance(first, dict):
                text = first.get("output_text") or first.get("text")
                if isinstance(text, str) and text.strip():
                    return text

        text = resp.get("text")
        if isinstance(text, str) and text.strip():
            return text

        return ""

    def _sanitize(self, parsed, state):
        if not isinstance(parsed, dict):
            return {}

        result = {}

        if "skip_base" in parsed:
            result["skip_base"] = bool(parsed["skip_base"])

        slots = state.get("troop_slots", []) if isinstance(state, dict) else []
        if not isinstance(slots, (list, tuple)):
            slots = []
        raw_order = parsed.get("deploy_order")
        if isinstance(raw_order, list):
            order = []
            for name in raw_order:
                if isinstance(name, str) and name in slots and name not in order:
                    order.append(name)
            if order:
                result["deploy_order"] = order

        try:
            value = float(parsed["surrender_after_s"])
            result["surrender_after_s"] = max(10.0, min(180.0, value))
        except (KeyError, TypeError, ValueError):
            pass

        side = parsed.get("target_side")
        if "target_side" in parsed:
            result["target_side"] = (
                side if side in ("all", "left", "right") else "all"
            )

        required = ("skip_base", "deploy_order", "surrender_after_s", "target_side")
        if not all(key in result for key in required):
            return {}
        return result


DEFAULT_LOCAL_MODEL = "qwen2.5vl:3b"
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434/api/generate"


class LocalLLMBrain(GeminiBrain):
    """
    Local LLM / VLM strategic brain powered by Ollama or a local OpenAI-compatible server.
    Designed to run locally on an NVIDIA RTX 4060 (8GB VRAM) with zero rate limits.
    Supports both structured state JSON and direct base screenshot vision (VLM).
    """

    name = "local_llm"

    def __init__(
        self,
        model: str = DEFAULT_LOCAL_MODEL,
        endpoint: str | None = None,
        timeout_s: float = 12.0,
        vision: bool = True,
    ):
        self.model = str(model or DEFAULT_LOCAL_MODEL).strip()
        self.endpoint = str(endpoint or DEFAULT_OLLAMA_ENDPOINT).strip()
        self.timeout_s = float(timeout_s)
        self.vision = bool(vision)
        self.enabled = True
        self.last_error = ""

    def describe(self) -> str:
        mode = "vlm+json" if self.vision else "json"
        return f"local_llm/{self.model} ({mode}) endpoint={self.endpoint}"

    @staticmethod
    def _encode_frame_base64(frame) -> str | None:
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        try:
            import base64
            import cv2

            h, w = frame.shape[:2]
            if w > 960:
                scale = 960.0 / float(w)
                frame = cv2.resize(
                    frame,
                    (960, max(1, int(h * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                return None
            return base64.b64encode(buf.tobytes()).decode("ascii")
        except Exception as e:
            logger.debug(f"local_llm frame encode failed: {e}")
            return None

    def plan(self, state: dict, frame=None) -> dict:
        if not self.enabled:
            return {}

        payload_text = json.dumps(state, ensure_ascii=False)
        body: dict = {
            "model": self.model,
            "system": SYSTEM_INSTRUCTION,
            "prompt": payload_text,
            "stream": False,
            "format": PLAN_SCHEMA,
            "keep_alive": "15m",
            "options": {
                "temperature": 0.15,
                "num_predict": 256,
            },
        }

        if self.vision and frame is not None:
            encoded_img = self._encode_frame_base64(frame)
            if encoded_img:
                body["images"] = [encoded_img]

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            self.last_error = ""
        except Exception as e:
            logger.warning(f"local_llm request failed ({self.endpoint}): {e}")
            self.last_error = str(e)
            return {}

        try:
            resp_json = json.loads(raw)
        except Exception as e:
            logger.warning(f"local_llm returned non-JSON: {e}")
            self.last_error = "invalid json response"
            return {}

        text = resp_json.get("response") or self._extract_text(resp_json)
        if not text:
            self.last_error = "empty model response"
            return {}

        try:
            parsed = json.loads(text) if isinstance(text, str) else text
        except Exception as e:
            logger.warning(f"local_llm plan was not valid JSON: {e}")
            self.last_error = "invalid plan json"
            return {}

        return self._sanitize(parsed, state)

