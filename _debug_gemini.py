import traceback

from services.gemini_service import _call_gemini


try:
    response_text = _call_gemini("Return strict JSON only with key ok and boolean true")
    print("OK", response_text[:300])
except Exception as exc:  # noqa: BLE001
    print(type(exc).__name__)
    print(str(exc))
    traceback.print_exc()
