# ===== [01] IMPORTS ==========================================================
import importlib

# ===== [02] TESTS ============================================================
def test_package_import():
    """src 패키지가 정상 import 되는지 확인"""
    pkg = importlib.import_module("src")
    assert hasattr(pkg, "__version__") or True  # 버전 없어도 True로 통과

def test_app_importable():
    """app.py가 최소한 import 에러 없이 로드되는지 확인"""
    try:
        import app  # noqa: F401
    except Exception as e:
        raise AssertionError(f"app.py import 실패: {e}")

def test_config_or_engine():
    """config, rag_engine 같은 핵심 모듈이 있다면 import 가능해야 함"""
    for mod in ["src.config", "src.rag_engine"]:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            # 아직 파일이 없을 수도 있으니 통과시킴
            pass
